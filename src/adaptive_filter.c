/*
 * INF560 - Adaptive Image Filtering Project
 *
 * This adaptive driver detects hardware resources and input image characteristics,
 * then automatically (or via a user override) selects the best parallelization approach.
 *
 * It uses our in–house gif_lib to load and store GIF images.
 *
 * Compile (for example):
 *   mpicc -o adaptive_filter adaptive_filter.c dgif_lib.c egif_lib.c gif_err.c gif_font.c gif_hash.c gifalloc.c openbsd-reallocarray.c quantize.c -lm -fopenmp -lcuda
 *
 * Run with:
 *   ./adaptive_filter input.gif output.gif [mode]
 * where [mode] can be "auto" (default), "cuda", "mpi", or "omp".
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "gif_lib.h"

/* Data structure for one pixel */
typedef struct pixel {
    int r;
    int g;
    int b;
} pixel;

/* Data structure for an animated GIF */
typedef struct animated_gif {
    int n_images;      /* Number of images */
    int *width;        /* Width of each image */
    int *height;       /* Height of each image */
    pixel **p;         /* Pixels of each image (flat array, row–major) */
    GifFileType *g;    /* Internal GIF representation (DO NOT MODIFY) */
} animated_gif;

/* ======================================================================
   load_pixels() – (Same as your sequential code.)
   ====================================================================== */
animated_gif * load_pixels( char * filename ) 
{
    GifFileType * g;
    ColorMapObject * colmap;
    int error;
    int n_images;
    int * width;
    int * height;
    pixel ** p;
    int i, j;
    animated_gif * image;

    g = DGifOpenFileName( filename, &error );
    if ( g == NULL ) {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename );
        return NULL;
    }
    error = DGifSlurp( g );
    if ( error != GIF_OK ) {
        fprintf( stderr, "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) );
        return NULL;
    }
    n_images = g->ImageCount;
    width = (int *)malloc( n_images * sizeof( int ) );
    height = (int *)malloc( n_images * sizeof( int ) );
    p = (pixel **)malloc( n_images * sizeof( pixel * ) );
    for ( i = 0; i < n_images; i++ ) {
        width[i] = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;
        p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) );
    }
    
    colmap = g->SColorMap;
    if ( colmap == NULL ) {
        fprintf( stderr, "Error global colormap is NULL\n" );
        return NULL;
    }
    for ( i = 0; i < n_images; i++ ) {
        for ( j = 0; j < width[i] * height[i]; j++ ) {
            int c = g->SavedImages[i].RasterBits[j];
            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }
    image = (animated_gif *)malloc( sizeof(animated_gif) );
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;
    return image;
}

/* ======================================================================
   output_modified_read_gif() and store_pixels() – (Same as your sequential code)
   ====================================================================== */
int output_modified_read_gif( char * filename, GifFileType * g ) 
{
    GifFileType * g2;
    int error2;
    g2 = EGifOpenFileName( filename, false, &error2 );
    if ( g2 == NULL ) {
        fprintf( stderr, "Error EGifOpenFileName %s\n", filename );
        return 0;
    }
    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;
    error2 = EGifSpew( g2 );
    if ( error2 != GIF_OK ) {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", error2, GifErrorString(g2->Error) );
        return 0;
    }
    return 1;
}
int store_pixels( char * filename, animated_gif * image )
{
    return output_modified_read_gif( filename, image->g );
}

/* ======================================================================
   detect_hardware_resources():
   Determine the number of MPI ranks, OpenMP threads, and CUDA GPUs.
   ====================================================================== */
void detect_hardware_resources(int *mpi_ranks, int *omp_threads, int *cuda_gpus)
{
    int flag;
    MPI_Initialized(&flag);
    if(flag) {
        MPI_Comm_size(MPI_COMM_WORLD, mpi_ranks);
    } else {
        *mpi_ranks = 1;
    }
    *omp_threads = omp_get_max_threads();
    cudaError_t cudaStatus = cudaGetDeviceCount(cuda_gpus);
    if(cudaStatus != cudaSuccess)
        *cuda_gpus = 0;
}

/* ======================================================================
   evaluate_input():
   Loads the input GIF (using load_pixels), then computes:
      - the number of frames
      - the average width and height.
   ====================================================================== */
void evaluate_input(const char *filename, int *num_frames, double *avg_width, double *avg_height)
{
    animated_gif *img = load_pixels((char*)filename);
    if(!img) {
        fprintf(stderr, "Error evaluating input: cannot load %s\n", filename);
        exit(1);
    }
    *num_frames = img->n_images;
    long sum_w = 0, sum_h = 0;
    for (int i = 0; i < img->n_images; i++) {
        sum_w += img->width[i];
        sum_h += img->height[i];
    }
    *avg_width = (double)sum_w / img->n_images;
    *avg_height = (double)sum_h / img->n_images;
    // Free the loaded image if not needed further (or keep it for processing).
    // For this evaluation, we free the pixel arrays and header data.
    for (int i = 0; i < img->n_images; i++) {
        free(img->p[i]);
    }
    free(img->p);
    free(img->width);
    free(img->height);
    // Note: We are not freeing g because it might be used later if you integrate.
    free(img);
}

/* ======================================================================
   main():
   - Initialize MPI.
   - Detect hardware resources.
   - Evaluate input GIF.
   - Select the best parallelization approach (or use user override).
   ====================================================================== */
int main(int argc, char **argv)
{
    int mpi_ranks = 1, omp_threads = 1, cuda_gpus = 0;
    MPI_Init(&argc, &argv);
    detect_hardware_resources(&mpi_ranks, &omp_threads, &cuda_gpus);

    printf("Hardware Resources Detected:\n");
    if(mpi_ranks > 1)
         printf("  MPI: %d ranks\n", mpi_ranks);
    else
         printf("  MPI: Not in use (single rank)\n");
    printf("  OpenMP threads: %d\n", omp_threads);
    printf("  CUDA GPUs: %d\n", cuda_gpus);

    if(argc < 3) {
         fprintf(stderr, "Usage: %s input.gif output.gif [mode]\n", argv[0]);
         MPI_Finalize();
         return 1;
    }
    char *input_filename = argv[1];
    char *output_filename = argv[2];

    int num_frames;
    double avg_w, avg_h;
    evaluate_input(input_filename, &num_frames, &avg_w, &avg_h);
    printf("Input GIF has %d frame(s) with average dimensions %.0f x %.0f\n",
           num_frames, avg_w, avg_h);

    // Determine mode: auto or user override (cuda, mpi, omp)
    char *mode = "auto";
    if(argc > 3) {
         mode = argv[3];
    }
    
    // Simple adaptive logic:
    if(strcmp(mode, "auto") == 0)
    {
         if(num_frames == 1 && (avg_w * avg_h) > 1000000 && cuda_gpus > 0) {
              printf("Auto-selected: CUDA (large single image, GPU available)\n");
              // Call your CUDA filtering function here.
         } else if(num_frames > 1) {
              printf("Auto-selected: MPI (multiple images, embarrassingly parallel)\n");
              // Call your MPI-based filtering function.
         } else {
              printf("Auto-selected: OpenMP/sequential (small image)\n");
              // Call your OpenMP or sequential filtering function.
         }
    }
    else if(strcmp(mode, "cuda") == 0)
    {
         printf("User selected: CUDA approach.\n");
         // Call your CUDA filtering function.
    }
    else if(strcmp(mode, "mpi") == 0)
    {
         printf("User selected: MPI approach.\n");
         // Call your MPI filtering function.
    }
    else if(strcmp(mode, "omp") == 0)
    {
         printf("User selected: OpenMP approach.\n");
         // Call your OpenMP (or sequential) filtering function.
    }
    else
    {
         printf("Unknown mode: %s. Defaulting to auto selection.\n", mode);
         // Use auto selection.
    }

    MPI_Finalize();
    return 0;
}
