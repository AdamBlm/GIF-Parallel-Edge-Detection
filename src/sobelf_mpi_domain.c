/*
 * INF560 - Image Filtering Project (MPI Domain Decomposition)
 * Modified to produce timing messages consistent with your test scripts.
 *
 * Single-pass filters with 3×3 neighborhoods (grayscale, blur, Sobel):
 *   - Domain-decompose the image by rows among ranks, each rank gets local_real_rows.
 *   - Each rank allocates +1 ghost row on top and +1 on bottom (even rank=0 and rank=size−1).
 *   - Exchange ghost rows with neighbors (or replicate if no neighbor).
 *   - Apply filters on [ghost_top .. ghost_top+local_real_rows−1].
 *   - Gather final pixels at rank 0, re-encode, and write out GIF.
 *
 * Compile:
 *   mpicc -o sobelf_mpi_domain sobelf_mpi_domain.c -lm -lgif -fopenmp
 *
 * Run:
 *   mpirun -np 4 ./sobelf_mpi_domain input.gif output.gif
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "gif_lib.h"

#ifndef SOBELF_DEBUG
#define SOBELF_DEBUG 0
#endif

/* ------------------------------------------------------------------
 *                   Structures and I/O helpers
 * ------------------------------------------------------------------ */

typedef struct pixel {
    int r, g, b;
} pixel;

/* Holds all frames (images) from one GIF */
typedef struct animated_gif {
    int n_images;
    int *width;
    int *height;
    pixel **p;        /* For each image i, p[i] is a flat array of W×H pixels in row-major order */
    GifFileType *g;   /* Internal GIF structure for rewriting (do not modify it outside store_pixels) */
} animated_gif;

/* -------------- Loading the GIF into animated_gif -------------- */

animated_gif * load_pixels(char *filename)
{
    int error;
    GifFileType *g = DGifOpenFileName(filename, &error);
    if (g == NULL) {
        fprintf(stderr, "Error opening input GIF %s\n", filename);
        return NULL;
    }
    error = DGifSlurp(g);
    if (error != GIF_OK) {
        fprintf(stderr, "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
        return NULL;
    }

    int n_images = g->ImageCount;
    int *width   = (int*) malloc(n_images * sizeof(int));
    int *height  = (int*) malloc(n_images * sizeof(int));
    pixel **p    = (pixel**) malloc(n_images * sizeof(pixel*));

    /* Convert color indices -> RGB in p[i] */
    ColorMapObject *colmap = g->SColorMap;
    if (!colmap) {
        fprintf(stderr, "Error: global colormap is NULL\n");
        return NULL;
    }

    for (int i = 0; i < n_images; i++) {
        width[i]  = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;
        p[i] = (pixel*) malloc(width[i] * height[i] * sizeof(pixel));

        for (int j = 0; j < width[i] * height[i]; j++) {
            int c = g->SavedImages[i].RasterBits[j];
            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }

    animated_gif *image = (animated_gif*) malloc(sizeof(animated_gif));
    image->n_images = n_images;
    image->width    = width;
    image->height   = height;
    image->p        = p;
    image->g        = g;
    return image;
}

/* -------------- Writing out the final GIF -------------- */

int output_modified_read_gif(char *filename, GifFileType *g)
{
    int error2;
    GifFileType *g2 = EGifOpenFileName(filename, false, &error2);
    if (!g2) {
        fprintf(stderr, "Error EGifOpenFileName %s\n", filename);
        return 0;
    }
    g2->SWidth          = g->SWidth;
    g2->SHeight         = g->SHeight;
    g2->SColorResolution= g->SColorResolution;
    g2->SBackGroundColor= g->SBackGroundColor;
    g2->AspectByte      = g->AspectByte;
    g2->SColorMap       = g->SColorMap;
    g2->ImageCount      = g->ImageCount;
    g2->SavedImages     = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks     = g->ExtensionBlocks;

    error2 = EGifSpew(g2);
    if (error2 != GIF_OK) {
        fprintf(stderr, "Error after EGifSpew: %d <%s>\n", error2, GifErrorString(g2->Error));
        return 0;
    }
    return 1;
}

/* Re-encode final pixel arrays -> RasterBits with a new color map. */
int store_pixels(char *filename, animated_gif *image)
{
    int n_colors = 0;
    GifColorType *colormap = (GifColorType*) malloc(256 * sizeof(GifColorType));
    if (!colormap) {
        fprintf(stderr, "Cannot allocate 256-colormap\n");
        return 0;
    }
    /* White by default */
    for (int i = 0; i < 256; i++) {
        colormap[i].Red   = 255;
        colormap[i].Green = 255;
        colormap[i].Blue  = 255;
    }

    /* Put background color at index 0 (just as an example) */
    {
        int bg = image->g->SBackGroundColor;
        int r  = image->g->SColorMap->Colors[bg].Red;
        int g  = image->g->SColorMap->Colors[bg].Green;
        int b  = image->g->SColorMap->Colors[bg].Blue;
        int moy = (r + g + b) / 3;
        if (moy < 0)   moy = 0;
        if (moy > 255) moy = 255;
        colormap[0].Red   = moy;
        colormap[0].Green = moy;
        colormap[0].Blue  = moy;
        image->g->SBackGroundColor = 0;
    }
    n_colors++;

    /* Gather all final colors */
    for (int i = 0; i < image->n_images; i++) {
        int W = image->width[i];
        int H = image->height[i];
        for (int j = 0; j < W * H; j++) {
            pixel px = image->p[i][j];
            int found = -1;
            for (int c = 0; c < n_colors; c++) {
                if (px.r == colormap[c].Red &&
                    px.g == colormap[c].Green &&
                    px.b == colormap[c].Blue) {
                    found = c;
                    break;
                }
            }
            if (found < 0) {
                if (n_colors >= 256) {
                    fprintf(stderr, "Error: too many colors\n");
                    return 0;
                }
                colormap[n_colors].Red   = px.r;
                colormap[n_colors].Green = px.g;
                colormap[n_colors].Blue  = px.b;
                n_colors++;
            }
        }
    }

    /* Round up to next power-of-2 if needed */
    int real_n = (1 << GifBitSize(n_colors));
    if (real_n != n_colors) {
        n_colors = real_n;
    }

    ColorMapObject *cmo = GifMakeMapObject(n_colors, colormap);
    if (!cmo) {
        fprintf(stderr, "Error building ColorMapObject\n");
        return 0;
    }
    image->g->SColorMap = cmo;

    /* Rewrite each frame's RasterBits from p[i] */
    for (int i = 0; i < image->n_images; i++) {
        int W = image->width[i];
        int H = image->height[i];
        for (int j = 0; j < W * H; j++) {
            pixel px = image->p[i][j];
            int idx_found = -1;
            for (int c = 0; c < n_colors; c++) {
                if (px.r == cmo->Colors[c].Red &&
                    px.g == cmo->Colors[c].Green &&
                    px.b == cmo->Colors[c].Blue) {
                    idx_found = c;
                    break;
                }
            }
            if (idx_found < 0) {
                fprintf(stderr, "Error: pixel not found in new colormap\n");
                return 0;
            }
            image->g->SavedImages[i].RasterBits[j] = idx_found;
        }
    }

    /* Finally call EGifSpew to write out */
    if (!output_modified_read_gif(filename, image->g)) {
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------
 *                 Single-Pass Filter Kernels
 * ------------------------------------------------------------------ */

/* Convert each pixel to grayscale in [ghost_top..ghost_top+local_real_rows-1]. */
void apply_gray_filter_local(pixel *block,
                             int ghost_top,
                             int local_real_rows,
                             int width)
{
    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            int moy = (block[idx].r + block[idx].g + block[idx].b) / 3;
            if (moy < 0)   moy = 0;
            if (moy > 255) moy = 255;
            block[idx].r = moy;
            block[idx].g = moy;
            block[idx].b = moy;
        }
    }
}

/* Single-pass blur with a 3×3 kernel (size=1). */
void apply_blur_filter_local_once(pixel *block,
                                  int ghost_top,
                                  int local_real_rows,
                                  int width,
                                  int size)  /* size=1 => 3×3 */
{
    int total_rows = ghost_top + local_real_rows;
    pixel *newBlock = (pixel*) malloc(total_rows * width * sizeof(pixel));
    if (!newBlock) return;

    /* Copy the entire portion first */
    memcpy(newBlock, block, total_rows * width * sizeof(pixel));

    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        /* skip j=0 and j=width-1 to avoid out-of-bounds on j±1 */
        for (int j = 1; j < width - 1; j++) {
            int sum_r = 0, sum_g = 0, sum_b = 0, count = 0;
            for (int di = -size; di <= size; di++) {
                for (int dj = -size; dj <= size; dj++) {
                    int idx = (i + di) * width + (j + dj);
                    sum_r += block[idx].r;
                    sum_g += block[idx].g;
                    sum_b += block[idx].b;
                    count++;
                }
            }
            int idx = i * width + j;
            newBlock[idx].r = sum_r / count;
            newBlock[idx].g = sum_g / count;
            newBlock[idx].b = sum_b / count;
        }
    }

    memcpy(block, newBlock, total_rows * width * sizeof(pixel));
    free(newBlock);
}

/* Single-pass Sobel with a 3×3 neighborhood. */
void apply_sobel_filter_local_once(pixel *block,
                                   int ghost_top,
                                   int local_real_rows,
                                   int width)
{
    int total_rows = ghost_top + local_real_rows;
    pixel *result = (pixel*) malloc(local_real_rows * width * sizeof(pixel));
    if (!result) return;
    memset(result, 0, local_real_rows * width * sizeof(pixel));

    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (int j = 1; j < width - 1; j++) {
            int b_no = block[(i-1)*width + (j-1)].b;
            int b_n  = block[(i-1)*width + j    ].b;
            int b_ne = block[(i-1)*width + (j+1)].b;
            int b_o  = block[i*width + (j-1)].b;
            int b_e  = block[i*width + (j+1)].b;
            int b_so = block[(i+1)*width + (j-1)].b;
            int b_s  = block[(i+1)*width + j    ].b;
            int b_se = block[(i+1)*width + (j+1)].b;

            float deltaX = -b_no + b_ne - 2*b_o + 2*b_e - b_so + b_se;
            float deltaY =  b_se + 2*b_s + b_so - b_ne - 2*b_n - b_no;
            float mag = sqrtf(deltaX*deltaX + deltaY*deltaY) / 4.0f;

            int ridx = (i - ghost_top) * width + j;
            if (mag > 50.0f)
                result[ridx].r = result[ridx].g = result[ridx].b = 255;
            else
                result[ridx].r = result[ridx].g = result[ridx].b = 0;
        }
    }

    /* Write results back */
    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx  = i * width + j;
            int ridx = (i - ghost_top) * width + j;
            block[idx].r = result[ridx].r;
            block[idx].g = result[ridx].g;
            block[idx].b = result[ridx].b;
        }
    }
    free(result);
}

/* ------------------------------------------------------------------
 *                     Helper: Exchange Ghost Rows
 * ------------------------------------------------------------------ */
void exchange_ghost_rows(pixel *local_block, int ghost_top, int local_real_rows, int width, int rank, int size) {
    // Exchange top ghost row: send our first real row and receive from upper neighbor.
    if (rank > 0) {
        MPI_Sendrecv(local_block + ghost_top * width, 
                     width * sizeof(pixel), MPI_BYTE,
                     rank - 1, 200,
                     local_block, 
                     width * sizeof(pixel), MPI_BYTE,
                     rank - 1, 300,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(local_block, local_block + ghost_top * width, width * sizeof(pixel));
    }
    // Exchange bottom ghost row: send our last real row and receive from lower neighbor.
    if (rank < size - 1) {
        MPI_Sendrecv(local_block + (ghost_top + local_real_rows - 1) * width, 
                     width * sizeof(pixel), MPI_BYTE,
                     rank + 1, 300,
                     local_block + (ghost_top + local_real_rows) * width, 
                     width * sizeof(pixel), MPI_BYTE,
                     rank + 1, 200,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(local_block + (ghost_top + local_real_rows) * width,
               local_block + (ghost_top + local_real_rows - 1) * width,
               width * sizeof(pixel));
    }
}

/* ------------------------------------------------------------------
 *                              MAIN
 * ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* We'll measure times for 3 stages: load, filter, export */
    double load_time   = 0.0;
    double filter_time = 0.0;
    double export_time = 0.0;

    if (rank == 0 && argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char *input_filename  = NULL;
    char *output_filename = NULL;
    animated_gif *image   = NULL;
    int n_images          = 0;

    /* ========== LOAD STAGE ========== */
    struct timeval t1, t2;
    if (rank == 0) {
        input_filename  = argv[1];
        output_filename = argv[2];

        gettimeofday(&t1, NULL);  /* Start load timer */
        image = load_pixels(input_filename);
        if (!image) {
            fprintf(stderr, "Error loading GIF %s\n", input_filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n_images = image->n_images;
        gettimeofday(&t2, NULL);  /* End load timer */
        load_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;

        /* Print the “loaded” line that your test scripts expect */
        printf("GIF loaded from file %s with %d image(s) in %.3f s\n",
               input_filename, n_images, load_time);
        fflush(stdout);
    }

    /* Let all ranks know how many frames we have. */
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* ========== FILTER STAGE ========== */
    struct timeval tf1, tf2;
    if (rank == 0) {
        gettimeofday(&tf1, NULL);
    }

    /* Loop over each frame and apply domain decomposition. */
    for (int img = 0; img < n_images; img++) {
        int w = 0, h = 0;
        if (rank == 0) {
            w = image->width[img];
            h = image->height[img];
        }
        MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* Domain decomposition by rows */
        int rows_per_proc = h / size;
        int remainder     = h % size;

        /* Compute global_start row for each rank. */
        int global_start;
        if (rank < remainder) {
            global_start = rank * (rows_per_proc + 1);
        } else {
            global_start = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
        }
        int local_real_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;

        /* 1 ghost row on top, 1 on bottom, for all ranks. */
        int ghost_top    = 1;
        int ghost_bottom = 1;
        int local_alloc_rows = local_real_rows + ghost_top + ghost_bottom;

        pixel *local_block = (pixel*) calloc(local_alloc_rows * w, sizeof(pixel));
        if (!local_block) {
            fprintf(stderr, "Rank %d: cannot allocate local_block\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        /* Scatter the frame rows from rank=0 */
        if (rank == 0) {
            memcpy(local_block + ghost_top * w, image->p[img], local_real_rows * w * sizeof(pixel));
            int offset = local_real_rows;
            for (int r = 1; r < size; r++) {
                int r_rows = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
                MPI_Send(image->p[img] + offset * w,
                         r_rows * w * sizeof(pixel),
                         MPI_BYTE,
                         r, 100,
                         MPI_COMM_WORLD);
                offset += r_rows;
            }
        } else {
            MPI_Recv(local_block + ghost_top * w,
                     local_real_rows * w * sizeof(pixel),
                     MPI_BYTE,
                     0, 100,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        /* --- Exchange ghost rows and apply filters in sequence --- */

        /* Initial ghost exchange */
        exchange_ghost_rows(local_block, ghost_top, local_real_rows, w, rank, size);

        /* Apply grayscale filter (point-wise) */
        apply_gray_filter_local(local_block, ghost_top, local_real_rows, w);
        /* Update ghost rows with new grayscale values */
        exchange_ghost_rows(local_block, ghost_top, local_real_rows, w, rank, size);

        /* Apply blur filter (3×3 kernel) */
        apply_blur_filter_local_once(local_block, ghost_top, local_real_rows, w, 1);
        /* Update ghost rows with blurred values */
        exchange_ghost_rows(local_block, ghost_top, local_real_rows, w, rank, size);

        /* Apply Sobel filter (3×3 kernel) */
        apply_sobel_filter_local_once(local_block, ghost_top, local_real_rows, w);

        /* Gather results back to rank=0 */
        if (rank == 0) {
            memcpy(image->p[img], local_block + ghost_top * w, local_real_rows * w * sizeof(pixel));
            int offset = local_real_rows;
            for (int r = 1; r < size; r++) {
                int r_rows = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
                MPI_Recv(image->p[img] + offset * w,
                         r_rows * w * sizeof(pixel),
                         MPI_BYTE,
                         r, 999,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                offset += r_rows;
            }
        } else {
            MPI_Send(local_block + ghost_top * w,
                     local_real_rows * w * sizeof(pixel),
                     MPI_BYTE,
                     0, 999,
                     MPI_COMM_WORLD);
        }

        free(local_block);
    } // end for each frame

    /* Summarize filter time at rank=0 */
    if (rank == 0) {
        gettimeofday(&tf2, NULL);
        filter_time = (tf2.tv_sec - tf1.tv_sec) + (tf2.tv_usec - tf1.tv_usec) / 1e6;
        /* Print “SOBEL done in X s” so your test script can parse. */
        printf("SOBEL done in %.3f s\n", filter_time);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ========== EXPORT STAGE ========== */
    if (rank == 0) {
        struct timeval te1, te2;
        gettimeofday(&te1, NULL);

        /* Store the final GIF */
        store_pixels(argv[2], image);

        gettimeofday(&te2, NULL);
        export_time = (te2.tv_sec - te1.tv_sec) + (te2.tv_usec - te1.tv_usec) / 1e6;

        /* Print “Export done in X s in file <output>” to match scripts. */
        printf("Export done in %.3f s in file %s\n", export_time, argv[2]);
        fflush(stdout);
    }

    MPI_Finalize();
    return 0;
}
