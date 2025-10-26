/*
 * INF560
 *
 * Image Filtering Project - Hybrid MPI + OpenMP + CUDA Implementation
 * 
 * This implementation uses three levels of parallelism:
 * 1. MPI - Distribute frames across computing nodes
 * 2. OpenMP - CPU thread-level parallelism within each node
 * 3. CUDA - GPU acceleration for compute-intensive filters
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdexcept>  // For std::runtime_error

#include "gif_lib.h"
#include "gif_utils.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Define MPI tags for messaging */
#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102

/* Default CUDA parameters */
#define TILE_WIDTH 16
#define TILE_HEIGHT 16
#define MAX_STREAMS_PER_GPU 4

/* Threshold to decide CPU vs GPU processing */
#define MIN_SIZE_FOR_GPU 128*128  /* Process images larger than this on GPU */

/* CUDA error checking macro */
#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/* Struct to send image data between processes */
typedef struct image_data {
    int width;
    int height;
    int image_idx;
    int total_size;
} image_data;

/* Struct to handle workload distribution using MPI */
typedef struct scatter_info {
    int *sendcounts;  /* Number of elements to send to each process */
    int *displs;      /* Displacement for each process */
    int *image_counts; /* Number of images per process */
    int *image_displs; /* Displacement for image indices */
    int *scatter_byte_counts; /* Number of bytes for each process - precomputed */
    int *scatter_byte_displs; /* Byte displacements - precomputed */
} scatter_info;

/* CUDA resources for each GPU */
typedef struct cuda_resources {
    int device_id;           /* CUDA device ID */
    cudaStream_t streams[MAX_STREAMS_PER_GPU]; /* CUDA streams for async execution */
    pixel *d_pixels;         /* Device buffer for pixels */
    pixel *d_temp;           /* Device temporary buffer */
    int *d_continue;         /* Device flag for convergence */
    size_t allocated_size;   /* Size of allocated device memory */
} cuda_resources;

/* Min/Max function helpers - removed since we'll use CUDA built-ins */
#ifndef MIN_MAX_FUNC
#define MIN_MAX_FUNC
// Only use these in host code - for device code, use CUDA's built-in min and max
static inline int min_int(int a, int b) {
    return (a < b) ? a : b;
}

static inline int max_int(int a, int b) {
    return (a > b) ? a : b;
}
#endif

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

/*
 * Forward declarations for GIF-related functions (implemented elsewhere)
 */
// These forward declarations are no longer needed as we include gif_utils.h
// animated_gif *load_pixels(char *filename);
// int output_modified_read_gif(char *filename, GifFileType *g);
// int store_pixels(char *filename, animated_gif *image);

// Added forward declaration for optimized store_pixels with color quantization
int store_pixels_optimized(char *filename, animated_gif *image);

/* Modified store_pixels function with color quantization for handling large color palettes */
int
store_pixels_optimized( char * filename, animated_gif * image )
{
    int n_colors = 0 ;
    pixel ** p ;
    int i, j, k ;
    GifColorType * colormap ;

    /* Initialize the new set of colors */
    colormap = (GifColorType *)malloc( 256 * sizeof( GifColorType ) ) ;
    if ( colormap == NULL ) 
    {
        fprintf( stderr,
                "Unable to allocate 256 colors\n" ) ;
        return 0 ;
    }

    /* Everything is white by default */
    for ( i = 0 ; i < 256 ; i++ ) 
    {
        colormap[i].Red = 255 ;
        colormap[i].Green = 255 ;
        colormap[i].Blue = 255 ;
    }

    /* Change the background color and store it */
    int moy ;
    moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green
            +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
            )/3 ;
    if ( moy < 0 ) moy = 0 ;
    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
    printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
            moy, moy, moy ) ;
#endif

    colormap[0].Red = moy ;
    colormap[0].Green = moy ;
    colormap[0].Blue = moy ;

    image->g->SBackGroundColor = 0 ;

    n_colors++ ;

    /* Process extension blocks in main structure */
    for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ )
    {
        int f ;

        f = image->g->ExtensionBlocks[j].Function ;
        if ( f == GRAPHICS_EXT_FUNC_CODE )
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3] ;

            if ( tr_color >= 0 &&
                    tr_color < 255 )
            {

                int found = -1 ;

                moy = 
                    (
                     image->g->SColorMap->Colors[ tr_color ].Red
                     +
                     image->g->SColorMap->Colors[ tr_color ].Green
                     +
                     image->g->SColorMap->Colors[ tr_color ].Blue
                    ) / 3 ;
                if ( moy < 0 ) moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                        i,
                        image->g->SColorMap->Colors[ tr_color ].Red,
                        image->g->SColorMap->Colors[ tr_color ].Green,
                        image->g->SColorMap->Colors[ tr_color ].Blue,
                        moy, moy, moy ) ;
#endif

                for ( k = 0 ; k < n_colors ; k++ )
                {
                    if ( 
                            moy == colormap[k].Red
                            &&
                            moy == colormap[k].Green
                            &&
                            moy == colormap[k].Blue
                       )
                    {
                        found = k ;
                    }
                }
                if ( found == -1  ) 
                {
                    if ( n_colors >= 256 ) 
                    {
                        fprintf( stderr, 
                                "Error: Found too many colors inside the image\n"
                               ) ;
                        return 0 ;
                    }

#if SOBELF_DEBUG
                    printf( "[DEBUG]\tNew color %d\n",
                            n_colors ) ;
#endif

                    colormap[n_colors].Red = moy ;
                    colormap[n_colors].Green = moy ;
                    colormap[n_colors].Blue = moy ;


                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors ;

                    n_colors++ ;
                } else
                {
#if SOBELF_DEBUG
                    printf( "[DEBUG]\tFound existing color %d\n",
                            found ) ;
#endif
                    image->g->ExtensionBlocks[j].Bytes[3] = found ;
                }
            }
        }
    }

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ )
        {
            int f ;

            f = image->g->SavedImages[i].ExtensionBlocks[j].Function ;
            if ( f == GRAPHICS_EXT_FUNC_CODE )
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] ;

                if ( tr_color >= 0 &&
                        tr_color < 255 )
                {

                    int found = -1 ;

                    moy = 
                        (
                         image->g->SColorMap->Colors[ tr_color ].Red
                         +
                         image->g->SColorMap->Colors[ tr_color ].Green
                         +
                         image->g->SColorMap->Colors[ tr_color ].Blue
                        ) / 3 ;
                    if ( moy < 0 ) moy = 0 ;
                    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                    printf( "[DEBUG] Transparency color image %d (%d,%d,%d) -> (%d,%d,%d)\n",
                            i,
                            image->g->SColorMap->Colors[ tr_color ].Red,
                            image->g->SColorMap->Colors[ tr_color ].Green,
                            image->g->SColorMap->Colors[ tr_color ].Blue,
                            moy, moy, moy ) ;
#endif

                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if ( 
                                moy == colormap[k].Red
                                &&
                                moy == colormap[k].Green
                                &&
                                moy == colormap[k].Blue
                           )
                        {
                            found = k ;
                        }
                    }
                    if ( found == -1  ) 
                    {
                        if ( n_colors >= 256 ) 
                        {
                            fprintf( stderr, 
                                    "Error: Found too many colors inside the image\n"
                                   ) ;
                            return 0 ;
                        }

#if SOBELF_DEBUG
                        printf( "[DEBUG]\tNew color %d\n",
                                n_colors ) ;
#endif

                        colormap[n_colors].Red = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue = moy ;


                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors ;

                        n_colors++ ;
                    } else
                    {
#if SOBELF_DEBUG
                        printf( "[DEBUG]\tFound existing color %d\n",
                                found ) ;
#endif
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background and transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p ;

    /* 
     * Color quantization step 
     * Convert colors to a reduced palette before processing 
     * This ensures we don't exceed the 256 color limit for GIF files
     */
    
    // We'll use a simple color quantization by masking least significant bits
    // This reduces color space significantly
    const int COLOR_MASK = 0xE0; // Keeps only 3 most significant bits (8 values per channel)
                                 // 11100000 in binary
    
    // Apply quantization to all pixels
    for (i = 0; i < image->n_images; i++) {
        for (j = 0; j < image->width[i] * image->height[i]; j++) {
            // Apply color quantization by masking out lower bits
            p[i][j].r = p[i][j].r & COLOR_MASK;
            p[i][j].g = p[i][j].g & COLOR_MASK;
            p[i][j].b = p[i][j].b & COLOR_MASK;
        }
    }

    /* Find the number of colors inside the image */
    for ( i = 0 ; i < image->n_images ; i++ )
    {

#if SOBELF_DEBUG
        printf( "OUTPUT: Processing image %d (total of %d images) -> %d x %d\n",
                i, image->n_images, image->width[i], image->height[i] ) ;
#endif

        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) 
        {
            int found = 0 ;
            for ( k = 0 ; k < n_colors ; k++ )
            {
                if ( p[i][j].r == colormap[k].Red &&
                        p[i][j].g == colormap[k].Green &&
                        p[i][j].b == colormap[k].Blue )
                {
                    found = 1 ;
                }
            }

            if ( found == 0 ) 
            {
                if ( n_colors >= 256 ) 
                {
                    fprintf( stderr, 
                            "Error: Found too many colors inside the image\n"
                           ) ;
                    return 0 ;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new %d color (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red = p[i][j].r ;
                colormap[n_colors].Green = p[i][j].g ;
                colormap[n_colors].Blue = p[i][j].b ;
                n_colors++ ;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif


    /* Round up to a power of 2 */
    if ( n_colors != (1 << GifBitSize(n_colors) ) )
    {
        n_colors = (1 << GifBitSize(n_colors) ) ;
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Change the color map inside the animated gif */
    ColorMapObject * cmo ;

    cmo = GifMakeMapObject( n_colors, colormap ) ;
    if ( cmo == NULL )
    {
        fprintf( stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                n_colors ) ;
        return 0 ;
    }

    image->g->SColorMap = cmo ;

    /* Update the raster bits according to color map */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ ) 
        {
            int found_index = -1 ;
            for ( k = 0 ; k < n_colors ; k++ ) 
            {
                if ( p[i][j].r == image->g->SColorMap->Colors[k].Red &&
                        p[i][j].g == image->g->SColorMap->Colors[k].Green &&
                        p[i][j].b == image->g->SColorMap->Colors[k].Blue )
                {
                    found_index = k ;
                }
            }

            if ( found_index == -1 ) 
            {
                fprintf( stderr,
                        "Error: Unable to find a pixel in the color map\n" ) ;
                return 0 ;
            }

            image->g->SavedImages[i].RasterBits[j] = found_index ;
        }
    }

    /* Write the final image */
    if ( !output_modified_read_gif( filename, image->g ) ) { return 0 ; }

    return 1 ;
}

/*
 * CPU implementations of the filters
 */
/* Apply grayscale filter to an image on CPU */
void apply_gray_filter_cpu(pixel *p, int width, int height)
{
    #pragma omp parallel for default(none) shared(p, width, height) schedule(static)
    for (int j = 0; j < height; j++)
    {
        #pragma omp simd
        for (int k = 0; k < width; k++)
        {
            int index = CONV(j, k, width);
            int moy = (p[index].r + p[index].g + p[index].b)/3;
            if (moy < 0) moy = 0;
            if (moy > 255) moy = 255;

            p[index].r = moy;
            p[index].g = moy;
            p[index].b = moy;
        }
    }
}

/* Apply blur filter to an image on CPU */
void apply_blur_filter_cpu(pixel *p, int width, int height, int size, int threshold, pixel *buffer)
{
    int end = 0;
    int n_iter = 0;
    
    // Initialize buffer with original pixel values
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(static)
    for (int j = 0; j < height; j++) {
        #pragma omp simd
        for (int k = 0; k < width; k++) {
            int index = CONV(j, k, width);
            buffer[index].r = p[index].r;
            buffer[index].g = p[index].g;
            buffer[index].b = p[index].b;
        }
    }

    /* Perform blur iterations until convergence */
    do
    {
        end = 1;
        n_iter++;

        // Process all pixels except the border
        #pragma omp parallel for default(none) shared(p, buffer, width, height, size) schedule(static)
        for (int j = size; j < height - size; j++)
        {
            for (int k = size; k < width - size; k++)
            {
                int t_r = 0, t_g = 0, t_b = 0;
                
                // Calculate average in the neighborhood
                for (int stencil_j = -size; stencil_j <= size; stencil_j++)
                {
                    for (int stencil_k = -size; stencil_k <= size; stencil_k++)
                    {
                        int idx = CONV(j+stencil_j, k+stencil_k, width);
                        t_r += p[idx].r;
                        t_g += p[idx].g;
                        t_b += p[idx].b;
                    }
                }

                // Calculate the average value for the pixel
                const int denom = (2*size+1)*(2*size+1);
                buffer[CONV(j, k, width)].r = t_r / denom;
                buffer[CONV(j, k, width)].g = t_g / denom;
                buffer[CONV(j, k, width)].b = t_b / denom;
            }
        }

        // Check convergence and update pixels
        int continue_flag = 0;
        
        #pragma omp parallel for default(none) shared(p, buffer, width, height, size, threshold) reduction(|:continue_flag) schedule(static)
        for (int j = size; j < height - size; j++)
        {
            for (int k = size; k < width - size; k++)
            {
                int index = CONV(j, k, width);
                int diff_r = buffer[index].r - p[index].r;
                int diff_g = buffer[index].g - p[index].g;
                int diff_b = buffer[index].b - p[index].b;

                if (abs(diff_r) > threshold || abs(diff_g) > threshold || abs(diff_b) > threshold)
                {
                    continue_flag = 1;
                }

                p[index].r = buffer[index].r;
                p[index].g = buffer[index].g;
                p[index].b = buffer[index].b;
            }
        }
        
        end = !continue_flag;
    }
    while (threshold > 0 && !end);

#if SOBELF_DEBUG
    printf("BLUR CPU: number of iterations: %d\n", n_iter);
#endif
}

/* Apply sobel filter to an image on CPU */
void apply_sobel_filter_cpu(pixel *p, int width, int height, pixel *buffer)
{
    // First, copy the original image to buffer to preserve it
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(static)
    for (int j = 0; j < height; j++)
    {
        #pragma omp simd
        for (int k = 0; k < width; k++)
        {
            int index = CONV(j, k, width);
            buffer[index].r = p[index].r;
            buffer[index].g = p[index].g;
            buffer[index].b = p[index].b;
        }
    }
    
    // Apply Sobel filter with improved memory access pattern
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(static)
    for (int j = 1; j < height - 1; j++)
    {
        // Calculate row offsets for better cache locality
        const int row_prev = CONV(j-1, 0, width);
        const int row_curr = CONV(j, 0, width);
        const int row_next = CONV(j+1, 0, width);
        
        #pragma omp simd
        for (int k = 1; k < width - 1; k++)
        {
            // Get pixel values from the 3x3 neighborhood
            // For all three channels - we'll use the maximum gradient
            
            // Red channel gradients
            float deltaX_red = -p[row_prev + k-1].r + p[row_prev + k+1].r 
                             - 2*p[row_curr + k-1].r + 2*p[row_curr + k+1].r 
                             - p[row_next + k-1].r + p[row_next + k+1].r;
                             
            float deltaY_red = p[row_prev + k-1].r + 2*p[row_prev + k].r + p[row_prev + k+1].r
                             - p[row_next + k-1].r - 2*p[row_next + k].r - p[row_next + k+1].r;
                             
            float val_red = sqrt(deltaX_red * deltaX_red + deltaY_red * deltaY_red)/4;
            
            // Green channel gradients  
            float deltaX_green = -p[row_prev + k-1].g + p[row_prev + k+1].g 
                               - 2*p[row_curr + k-1].g + 2*p[row_curr + k+1].g 
                               - p[row_next + k-1].g + p[row_next + k+1].g;
                               
            float deltaY_green = p[row_prev + k-1].g + 2*p[row_prev + k].g + p[row_prev + k+1].g
                               - p[row_next + k-1].g - 2*p[row_next + k].g - p[row_next + k+1].g;
                               
            float val_green = sqrt(deltaX_green * deltaX_green + deltaY_green * deltaY_green)/4;
            
            // Blue channel gradients
            float deltaX_blue = -p[row_prev + k-1].b + p[row_prev + k+1].b 
                              - 2*p[row_curr + k-1].b + 2*p[row_curr + k+1].b 
                              - p[row_next + k-1].b + p[row_next + k+1].b;
                              
            float deltaY_blue = p[row_prev + k-1].b + 2*p[row_prev + k].b + p[row_prev + k+1].b
                              - p[row_next + k-1].b - 2*p[row_next + k].b - p[row_next + k+1].b;
                              
            float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;

            // Find maximum gradient value across all channels
            float max_val = val_red;
            if (val_green > max_val) max_val = val_green;
            if (val_blue > max_val) max_val = val_blue;

            // Apply binary thresholding
            int idx = row_curr + k;
            if (max_val > 15) {
                buffer[idx].r = 255;
                buffer[idx].g = 255;
                buffer[idx].b = 255;
            } else {
                buffer[idx].r = 0;
                buffer[idx].g = 0;
                buffer[idx].b = 0;
            }
        }
    }
    
    // Copy result back to original image
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(static)
    for (int j = 0; j < height; j++)
    {
        #pragma omp simd
        for (int k = 0; k < width; k++)
        {
            int index = CONV(j, k, width);
            p[index].r = buffer[index].r;
            p[index].g = buffer[index].g;
            p[index].b = buffer[index].b;
        }
    }
}

/* Process an image on CPU using all filters */
void process_image_cpu(pixel *pixels, int width, int height)
{
    // Allocate buffer for intermediate operations
    pixel *buffer = (pixel*)malloc(width * height * sizeof(pixel));
    if (!buffer) {
        fprintf(stderr, "ERROR: Failed to allocate CPU buffer\n");
        exit(EXIT_FAILURE);
    }
    
    // Apply filters in sequence
    apply_gray_filter_cpu(pixels, width, height);
    apply_blur_filter_cpu(pixels, width, height, 3, 0, buffer);
    apply_sobel_filter_cpu(pixels, width, height, buffer);
    
    // Free buffer
    free(buffer);
}

/*
 * CUDA implementations of the filters
 */
/* Grayscale conversion kernel */
__global__ void grayscale_kernel(pixel *d_pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        int moy = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;
        
        d_pixels[idx].r = moy;
        d_pixels[idx].g = moy;
        d_pixels[idx].b = moy;
    }
}

/* Blur filter kernel with shared memory tiling */
__global__ void blur_kernel(pixel *d_in, pixel *d_out, int width, int height, int size)
{
    extern __shared__ pixel sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    
    // Shared memory dimensions including halo
    int sharedWidth = blockDim.x + 2 * size;
    int sharedHeight = blockDim.y + 2 * size;
    
    // Load center and halo regions into shared memory
    for (int j = ty; j < sharedHeight; j += blockDim.y) {
        int y_global = by + j - size;
        
        for (int i = tx; i < sharedWidth; i += blockDim.x) {
            int x_global = bx + i - size;
            
            // Clamp coordinates to valid range
            int valid_x = max(0, min(width - 1, x_global));
            int valid_y = max(0, min(height - 1, y_global));
            
            sharedMem[j * sharedWidth + i] = d_in[valid_y * width + valid_x];
        }
    }
    
    __syncthreads();
    
    // Only process interior pixels
    if (x < width && y < height) {
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;
        int count = 0;
        
        // Compute average using stencil
        for (int j = -size; j <= size; j++) {
            for (int i = -size; i <= size; i++) {
                int sx = tx + i + size;
                int sy = ty + j + size;
                
                if (sx >= 0 && sx < sharedWidth && sy >= 0 && sy < sharedHeight) {
                    t_r += sharedMem[sy * sharedWidth + sx].r;
                    t_g += sharedMem[sy * sharedWidth + sx].g;
                    t_b += sharedMem[sy * sharedWidth + sx].b;
                    count++;
                }
            }
        }
        
        // Write output
        int idx = y * width + x;
        d_out[idx].r = t_r / count;
        d_out[idx].g = t_g / count;
        d_out[idx].b = t_b / count;
    }
}

/* Check for convergence of blur iteration */
__global__ void check_convergence_kernel(pixel *d_prev, pixel *d_curr, int width, int height, 
                                         int threshold, int *d_continue)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        
        int diff_r = abs(d_curr[idx].r - d_prev[idx].r);
        int diff_g = abs(d_curr[idx].g - d_prev[idx].g);
        int diff_b = abs(d_curr[idx].b - d_prev[idx].b);
        
        if (diff_r > threshold || diff_g > threshold || diff_b > threshold) {
            *d_continue = 1;
        }
    }
}

/* Sobel filter kernel using shared memory for efficient stencil operation */
__global__ void sobel_kernel(pixel *d_in, pixel *d_out, int width, int height)
{
    extern __shared__ pixel sharedMem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int x = bx + tx;
    int y = by + ty;
    
    // Shared memory dimensions including 1-pixel halo
    int sharedWidth = blockDim.x + 2;
    int sharedHeight = blockDim.y + 2;
    
    // Load center and halo regions into shared memory
    for (int j = ty; j < sharedHeight; j += blockDim.y) {
        int y_global = by + j - 1;
        
        for (int i = tx; i < sharedWidth; i += blockDim.x) {
            int x_global = bx + i - 1;
            
            // Clamp coordinates to valid range
            int valid_x = max(0, min(width - 1, x_global));
            int valid_y = max(0, min(height - 1, y_global));
            
            sharedMem[j * sharedWidth + i] = d_in[valid_y * width + valid_x];
        }
    }
    
    __syncthreads();
    
    // Only process interior pixels
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        
        // Extract red channel values from the 3x3 neighborhood
        int p00_r = sharedMem[(ty) * sharedWidth + (tx)].r;
        int p01_r = sharedMem[(ty) * sharedWidth + (tx+1)].r;
        int p02_r = sharedMem[(ty) * sharedWidth + (tx+2)].r;
        int p10_r = sharedMem[(ty+1) * sharedWidth + (tx)].r;
        int p11_r = sharedMem[(ty+1) * sharedWidth + (tx+1)].r;
        int p12_r = sharedMem[(ty+1) * sharedWidth + (tx+2)].r;
        int p20_r = sharedMem[(ty+2) * sharedWidth + (tx)].r;
        int p21_r = sharedMem[(ty+2) * sharedWidth + (tx+1)].r;
        int p22_r = sharedMem[(ty+2) * sharedWidth + (tx+2)].r;
        
        // Process red channel
        float Gx_r = -p00_r + p02_r
                   - 2.0f * p10_r + 2.0f * p12_r
                   - p20_r + p22_r;
                   
        float Gy_r = -p00_r - 2.0f * p01_r - p02_r
                   + p20_r + 2.0f * p21_r + p22_r;
        
        float val_r = sqrtf(Gx_r * Gx_r + Gy_r * Gy_r) / 4.0f;
        
        // Extract green channel values
        int p00_g = sharedMem[(ty) * sharedWidth + (tx)].g;
        int p01_g = sharedMem[(ty) * sharedWidth + (tx+1)].g;
        int p02_g = sharedMem[(ty) * sharedWidth + (tx+2)].g;
        int p10_g = sharedMem[(ty+1) * sharedWidth + (tx)].g;
        int p11_g = sharedMem[(ty+1) * sharedWidth + (tx+1)].g;
        int p12_g = sharedMem[(ty+1) * sharedWidth + (tx+2)].g;
        int p20_g = sharedMem[(ty+2) * sharedWidth + (tx)].g;
        int p21_g = sharedMem[(ty+2) * sharedWidth + (tx+1)].g;
        int p22_g = sharedMem[(ty+2) * sharedWidth + (tx+2)].g;
        
        // Process green channel
        float Gx_g = -p00_g + p02_g
                   - 2.0f * p10_g + 2.0f * p12_g
                   - p20_g + p22_g;
                   
        float Gy_g = -p00_g - 2.0f * p01_g - p02_g
                   + p20_g + 2.0f * p21_g + p22_g;
        
        float val_g = sqrtf(Gx_g * Gx_g + Gy_g * Gy_g) / 4.0f;
        
        // Extract blue channel values
        int p00_b = sharedMem[(ty) * sharedWidth + (tx)].b;
        int p01_b = sharedMem[(ty) * sharedWidth + (tx+1)].b;
        int p02_b = sharedMem[(ty) * sharedWidth + (tx+2)].b;
        int p10_b = sharedMem[(ty+1) * sharedWidth + (tx)].b;
        int p11_b = sharedMem[(ty+1) * sharedWidth + (tx+1)].b;
        int p12_b = sharedMem[(ty+1) * sharedWidth + (tx+2)].b;
        int p20_b = sharedMem[(ty+2) * sharedWidth + (tx)].b;
        int p21_b = sharedMem[(ty+2) * sharedWidth + (tx+1)].b;
        int p22_b = sharedMem[(ty+2) * sharedWidth + (tx+2)].b;
        
        // Process blue channel
        float Gx_b = -p00_b + p02_b
                   - 2.0f * p10_b + 2.0f * p12_b
                   - p20_b + p22_b;
                   
        float Gy_b = -p00_b - 2.0f * p01_b - p02_b
                   + p20_b + 2.0f * p21_b + p22_b;
        
        float val_b = sqrtf(Gx_b * Gx_b + Gy_b * Gy_b) / 4.0f;
        
        // Find maximum gradient value across all channels (matching CPU implementation)
        float max_val = val_r;
        if (val_g > max_val) max_val = val_g;
        if (val_b > max_val) max_val = val_b;
        
        // Binary threshold using same threshold as CPU version (15)
        if (max_val > 15) {
            d_out[idx].r = 255;
            d_out[idx].g = 255;
            d_out[idx].b = 255;
        } else {
            d_out[idx].r = 0;
            d_out[idx].g = 0;
            d_out[idx].b = 0;
        }
    }
    else if (x < width && y < height) {
        // Copy edge pixels unchanged
        d_out[y * width + x] = d_in[y * width + x];
    }
}

/* Initialize CUDA resources for a GPU */
cuda_resources* init_cuda_resources(int device_id)
{
    cuda_resources *res = (cuda_resources*)malloc(sizeof(cuda_resources));
    if (!res) return NULL;
    
    // Initialize fields
    res->device_id = device_id;
    res->d_pixels = NULL;
    res->d_temp = NULL;
    res->d_continue = NULL;
    res->allocated_size = 0;
    
    // Set device
    CUDA_CHECK(cudaSetDevice(device_id));
    
    // Create streams
    for (int i = 0; i < MAX_STREAMS_PER_GPU; i++) {
        CUDA_CHECK(cudaStreamCreate(&res->streams[i]));
    }
    
    // Allocate device memory for continue flag
    CUDA_CHECK(cudaMalloc((void**)&res->d_continue, sizeof(int)));
    
    return res;
}

/* Ensure device buffers are large enough */
void ensure_device_memory(cuda_resources *res, size_t required_size)
{
    if (required_size > res->allocated_size) {
        // Free previous allocations if any
        if (res->d_pixels) {
            cudaError_t err = cudaFree(res->d_pixels);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error freeing d_pixels: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("CUDA free error");
            }
            res->d_pixels = NULL;
        }
        
        if (res->d_temp) {
            cudaError_t err = cudaFree(res->d_temp);
            if (err != cudaSuccess) {
                fprintf(stderr, "CUDA error freeing d_temp: %s\n", cudaGetErrorString(err));
                throw std::runtime_error("CUDA free error");
            }
            res->d_temp = NULL;
        }
        
        // Allocate new memory with some extra to avoid frequent reallocations
        size_t new_size = required_size * 1.2; // 20% extra
        
        cudaError_t err = cudaMalloc((void**)&res->d_pixels, new_size);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error allocating d_pixels: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA malloc error");
        }
        
        err = cudaMalloc((void**)&res->d_temp, new_size);
        if (err != cudaSuccess) {
            // Free the first allocation to avoid memory leaks
            cudaFree(res->d_pixels);
            res->d_pixels = NULL;
            fprintf(stderr, "CUDA error allocating d_temp: %s\n", cudaGetErrorString(err));
            throw std::runtime_error("CUDA malloc error");
        }
        
        res->allocated_size = new_size;
    }
}

/* Free CUDA resources */
void free_cuda_resources(cuda_resources *res)
{
    if (res) {
        CUDA_CHECK(cudaSetDevice(res->device_id));
        
        // Destroy streams
        for (int i = 0; i < MAX_STREAMS_PER_GPU; i++) {
            CUDA_CHECK(cudaStreamDestroy(res->streams[i]));
        }
        
        // Free memory
        if (res->d_pixels) CUDA_CHECK(cudaFree(res->d_pixels));
        if (res->d_temp) CUDA_CHECK(cudaFree(res->d_temp));
        if (res->d_continue) CUDA_CHECK(cudaFree(res->d_continue));
        
        free(res);
    }
}

/* Process an image using CUDA with a specific stream */
void process_image_cuda(pixel *h_pixels, int width, int height, cuda_resources *res, int stream_idx)
{
    // Validate inputs
    if (!h_pixels || width <= 0 || height <= 0 || !res) {
        fprintf(stderr, "Error: Invalid parameters to process_image_cuda\n");
        throw std::runtime_error("Invalid parameters to process_image_cuda");
    }
    
    // Calculate buffer size
    size_t pixelsSize = width * height * sizeof(pixel);
    
    // Set device
    cudaError_t err = cudaSetDevice(res->device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error setting device: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA device error");
    }
    
    // Ensure device memory is large enough
    try {
        ensure_device_memory(res, pixelsSize);
    } catch (...) {
        fprintf(stderr, "Failed to allocate device memory\n");
        throw;
    }
    
    // Get the stream to use
    cudaStream_t stream = res->streams[stream_idx % MAX_STREAMS_PER_GPU];
    
    // Copy input data to device
    err = cudaMemcpyAsync(res->d_pixels, h_pixels, pixelsSize, 
                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA memcpy error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA memory copy error");
    }
    
    // Setup execution parameters
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Apply grayscale filter
    grayscale_kernel<<<gridSize, blockSize, 0, stream>>>(res->d_pixels, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA grayscale kernel error: %s\n", cudaGetErrorString(err));
        throw std::runtime_error("CUDA kernel error");
    }
    
    // Apply blur filter with convergence
    const int blurSize = 3;  // Same as in the CPU version
    const int threshold = 0; // Same as in the CPU version
    
    // Calculate shared memory size for blur kernel
    size_t sharedMemSizeBlur = (blockSize.x + 2 * blurSize) * 
                              (blockSize.y + 2 * blurSize) * 
                              sizeof(pixel);
    
    if (threshold > 0) {
        int h_continue;
        int max_iterations = 50;  // Limit iterations to avoid infinite loop
        int iter = 0;
        
        do {
            // Initialize continue flag
            h_continue = 0;
            CUDA_CHECK(cudaMemcpyAsync(res->d_continue, &h_continue, sizeof(int), 
                                    cudaMemcpyHostToDevice, stream));
            
            // Apply blur
            blur_kernel<<<gridSize, blockSize, sharedMemSizeBlur, stream>>>(
                res->d_pixels, res->d_temp, width, height, blurSize);
            
            // Check for convergence
            check_convergence_kernel<<<gridSize, blockSize, 0, stream>>>(
                res->d_pixels, res->d_temp, width, height, threshold, res->d_continue);
            
            // Swap buffers
            pixel *temp = res->d_pixels;
            res->d_pixels = res->d_temp;
            res->d_temp = temp;
            
            // Get continue flag
            CUDA_CHECK(cudaMemcpyAsync(&h_continue, res->d_continue, sizeof(int), 
                                    cudaMemcpyDeviceToHost, stream));
            
            // Need to synchronize to check the flag
            CUDA_CHECK(cudaStreamSynchronize(stream));
            
            iter++;
        } while (h_continue && iter < max_iterations);
        
#if SOBELF_DEBUG
        printf("BLUR GPU: number of iterations: %d\n", iter);
#endif
    } else {
        // Just apply blur once
        blur_kernel<<<gridSize, blockSize, sharedMemSizeBlur, stream>>>(
            res->d_pixels, res->d_temp, width, height, blurSize);
        
        // Swap buffers
        pixel *temp = res->d_pixels;
        res->d_pixels = res->d_temp;
        res->d_temp = temp;
    }
    
    // Apply Sobel filter
    size_t sharedMemSizeSobel = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(pixel);
    sobel_kernel<<<gridSize, blockSize, sharedMemSizeSobel, stream>>>(
        res->d_pixels, res->d_temp, width, height);
    
    // Copy results back to host
    CUDA_CHECK(cudaMemcpyAsync(h_pixels, res->d_temp, pixelsSize, 
                            cudaMemcpyDeviceToHost, stream));
    
    // Synchronize to ensure the transfer is complete
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/* Free scatter/gather information */
void free_scatter_info(scatter_info *info)
{
    if (info) {
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
    }
}

/* Create scatter/gather information for collective operations */
scatter_info* create_scatter_info(int n_images, int size) {
    printf("DEBUG: create_scatter_info called with n_images=%d, size=%d\n", n_images, size);
    
    scatter_info* info = (scatter_info*)malloc(sizeof(scatter_info));
    if (!info) {
        fprintf(stderr, "ERROR: Failed to allocate scatter_info struct\n");
        return NULL;
    }
    
    // Initialize all pointers to NULL to make cleanup easier
    info->sendcounts = NULL;
    info->displs = NULL;
    info->image_counts = NULL;
    info->image_displs = NULL;
    info->scatter_byte_counts = NULL;
    info->scatter_byte_displs = NULL;
    
    info->sendcounts = (int*)calloc(size, sizeof(int));
    if (!info->sendcounts) {
        fprintf(stderr, "ERROR: Failed to allocate sendcounts array\n");
        free_scatter_info(info);
        return NULL;
    }
    
    info->displs = (int*)calloc(size, sizeof(int));
    if (!info->displs) {
        fprintf(stderr, "ERROR: Failed to allocate displs array\n");
        free_scatter_info(info);
        return NULL;
    }
    
    info->image_counts = (int*)calloc(size, sizeof(int));
    if (!info->image_counts) {
        fprintf(stderr, "ERROR: Failed to allocate image_counts array\n");
        free_scatter_info(info);
        return NULL;
    }
    
    info->image_displs = (int*)calloc(size, sizeof(int));
    if (!info->image_displs) {
        fprintf(stderr, "ERROR: Failed to allocate image_displs array\n");
        free_scatter_info(info);
        return NULL;
    }
    
    info->scatter_byte_counts = (int*)calloc(size, sizeof(int));
    if (!info->scatter_byte_counts) {
        fprintf(stderr, "ERROR: Failed to allocate scatter_byte_counts array\n");
        free_scatter_info(info);
        return NULL;
    }
    
    info->scatter_byte_displs = (int*)calloc(size, sizeof(int));
    if (!info->scatter_byte_displs) {
        fprintf(stderr, "ERROR: Failed to allocate scatter_byte_displs array\n");
        free_scatter_info(info);
        return NULL;
    }
    
    // Distribute images evenly among processes
    int base_count = n_images / size;
    int remainder = n_images % size;
    
    printf("DEBUG: Distributing %d images among %d processes (base=%d, remainder=%d)\n", 
           n_images, size, base_count, remainder);
    
    for (int i = 0; i < size; i++) {
        info->image_counts[i] = base_count + (i < remainder ? 1 : 0);
        if (i > 0) {
            info->image_displs[i] = info->image_displs[i-1] + info->image_counts[i-1];
        }
        printf("DEBUG:   Process %d gets %d images starting at index %d\n", 
               i, info->image_counts[i], info->image_displs[i]);
    }
    
    return info;
}

/* Calculate scatter/gather counts for pixel data */
void calculate_pixel_counts(scatter_info* scatter_data, int* widths, int* heights, int size) {
    if (!scatter_data || !widths || !heights) return;
    
    // Calculate displacements and pixel counts
    for (int i = 0; i < size; i++) {
        int start_img = scatter_data->image_displs[i];
        int end_img = start_img + scatter_data->image_counts[i];
        
        scatter_data->sendcounts[i] = 0;
        for (int j = start_img; j < end_img; j++) {
            scatter_data->sendcounts[i] += widths[j] * heights[j];
        }
        
        if (i > 0) {
            scatter_data->displs[i] = scatter_data->displs[i-1] + scatter_data->sendcounts[i-1];
        }
        
        // Pre-calculate byte counts and displacements to avoid redundant calculations
        scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
        scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
    }
}

// Helper function to free allocated resources
void free_resources(pixel **p, int n_images)
{
    if (p) {
        for (int i = 0; i < n_images; i++) {
            if (p[i]) free(p[i]);
        }
        free(p);
    }
}

/*
 * Main entry point
 */
int main(int argc, char **argv)
{
    char *input_filename;
    char *output_filename;
    animated_gif *image = NULL;
    struct timeval t1, t2;
    double duration;
    int rank, size;
    MPI_Status status;
    scatter_info *scatter_data = NULL;
    int gpu_count = 0;
    int num_threads = 4; // Default number of OpenMP threads

    /* Initialize MPI with thread support for OpenMP and CUDA */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("DEBUG: MPI initialized, starting hybrid implementation\n");
    }

    /* Check command-line arguments */
    if (argc < 3)
    {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input.gif output.gif [num_threads]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];
    
    if (rank == 0) {
        printf("DEBUG: Processing file %s -> %s\n", input_filename, output_filename);
    }
    
    /* Set number of OpenMP threads if provided */
    if (argc > 3)
    {
        num_threads = atoi(argv[3]);
        if (num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        }
    }
    
    /* Initialize CUDA and get GPU count */
    cudaError_t err = cudaGetDeviceCount(&gpu_count);
    if (err != cudaSuccess) {
        // Reset to 0 if there was an error checking for GPU availability
        fprintf(stderr, "CUDA error: %s. Continuing with CPU only.\n", cudaGetErrorString(err));
        gpu_count = 0;
    }
    
    if (rank == 0) {
        printf("DEBUG: CUDA initialization complete\n");
        printf("Using %d MPI processes with %d OpenMP threads per process\n", 
               size, omp_get_max_threads());
        printf("Detected %d CUDA-capable GPU(s)\n", gpu_count);
    }
    
    /* Initialize CUDA resources */
    cuda_resources **gpu_resources = NULL;
    int my_gpu_count = 0;
    
    if (gpu_count > 0) {
        printf("DEBUG[%d]: Initializing GPU resources\n", rank);
        
        /* Determine number of GPUs for this process */
        int gpus_per_process = (gpu_count + size - 1) / size; // Ceiling division
        int start_gpu = min(rank * gpus_per_process, gpu_count);
        int end_gpu = min(start_gpu + gpus_per_process, gpu_count);
        my_gpu_count = end_gpu - start_gpu;
        
        printf("DEBUG[%d]: Assigned GPUs %d to %d (total: %d)\n", rank, start_gpu, end_gpu-1, my_gpu_count);
        
        if (my_gpu_count > 0) {
            /* Allocate and initialize GPU resources */
            gpu_resources = (cuda_resources**)malloc(my_gpu_count * sizeof(cuda_resources*));
            if (!gpu_resources) {
                fprintf(stderr, "ERROR: Failed to allocate GPU resources array\n");
                /* Continue without CUDA rather than aborting */
                my_gpu_count = 0;
            } else {
                /* Initialize each GPU */
                for (int i = 0; i < my_gpu_count; i++) {
                    int device_id = start_gpu + i;
                    
                    printf("DEBUG[%d]: Initializing GPU %d\n", rank, device_id);
                    
                    // Check if this GPU is usable
                    cudaError_t err = cudaSetDevice(device_id);
                    if (err != cudaSuccess) {
                        fprintf(stderr, "WARNING: Failed to set device %d: %s\n", 
                                device_id, cudaGetErrorString(err));
                        gpu_resources[i] = NULL;
                        continue;
                    }
                    
                    gpu_resources[i] = init_cuda_resources(device_id);
                    if (!gpu_resources[i]) {
                        fprintf(stderr, "WARNING: Failed to initialize CUDA resources for GPU %d\n", 
                                device_id);
                    } else {
                        printf("Process %d initialized GPU %d\n", rank, device_id);
                    }
                }
            }
        }
    }
    
    printf("DEBUG[%d]: GPU initialization complete, my_gpu_count=%d\n", rank, my_gpu_count);

    int n_images = 0;
    int *image_widths = NULL;
    int *image_heights = NULL;
    
    /* Process 0 (master) loads the image and distributes work */
    if (rank == 0)
    {
        /* IMPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Load file and store the pixels in array */
        image = load_pixels(input_filename);
        if (image == NULL) { 
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1; 
        }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
                input_filename, image->n_images, duration);
                
        n_images = image->n_images;
        image_widths = image->width;
        image_heights = image->height;
        
        printf("DEBUG[0]: Creating scatter info for %d images\n", n_images);
        
        // Create scatter info for workload distribution
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Failed to allocate scatter info\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        printf("DEBUG[0]: Calculating pixel counts\n");
        
        // Calculate pixel counts for each process
        calculate_pixel_counts(scatter_data, image_widths, image_heights, size);
        
        printf("DEBUG[0]: Scatter info ready\n");
    }

    /* Broadcast the number of images to all processes */
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    printf("DEBUG[%d]: Received n_images=%d\n", rank, n_images);

    /* Allocate memory for image dimensions on all processes */
    if (rank != 0) {
        image_widths = (int*)malloc(n_images * sizeof(int));
        image_heights = (int*)malloc(n_images * sizeof(int));
        if (!image_widths || !image_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Also allocate scatter_info structure for non-root processes
        scatter_data = (scatter_info*)malloc(sizeof(scatter_info));
        if (!scatter_data) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Initialize the structure with NULL pointers to avoid double frees
        scatter_data->sendcounts = NULL;
        scatter_data->displs = NULL;
        scatter_data->image_counts = NULL;
        scatter_data->image_displs = NULL;
        scatter_data->scatter_byte_counts = NULL;
        scatter_data->scatter_byte_displs = NULL;
        
        // Allocate memory for the arrays
        scatter_data->sendcounts = (int*)malloc(size * sizeof(int));
        scatter_data->displs = (int*)malloc(size * sizeof(int));
        scatter_data->image_counts = (int*)malloc(size * sizeof(int));
        scatter_data->image_displs = (int*)malloc(size * sizeof(int));
        scatter_data->scatter_byte_counts = (int*)malloc(size * sizeof(int));
        scatter_data->scatter_byte_displs = (int*)malloc(size * sizeof(int));
        
        if (!scatter_data->sendcounts || !scatter_data->displs || 
            !scatter_data->image_counts || !scatter_data->image_displs ||
            !scatter_data->scatter_byte_counts || !scatter_data->scatter_byte_displs) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info arrays\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Broadcast the image dimensions to all processes */
    MPI_Bcast(image_widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image_heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* Broadcast scatter info arrays to all processes */
    printf("DEBUG[%d]: Broadcasting scatter info\n", rank);

    // Broadcast all arrays in separate MPI operations
    MPI_Bcast(scatter_data->image_counts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(scatter_data->image_displs, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(scatter_data->sendcounts, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(scatter_data->displs, size, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate derived values for non-root processes
    if (rank != 0) {
        for (int i = 0; i < size; i++) {
            scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
            scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
        }
    }

    printf("DEBUG[%d]: My work: %d images, starting at image %d\n", 
           rank, scatter_data->image_counts[rank], scatter_data->image_displs[rank]);

    /* FILTER Timer start */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t1, NULL);
    }

    /* Get information for this process */
    int my_n_images = scatter_data->image_counts[rank];
    int my_start_image = scatter_data->image_displs[rank];
    
    /* Allocate memory for local image processing */
    pixel **my_pixels = NULL;
    int *my_widths = NULL;
    int *my_heights = NULL;
    
    if (my_n_images > 0) {
        /* Allocate memory for local image data */
        my_pixels = (pixel**)malloc(my_n_images * sizeof(pixel*));
        my_widths = (int*)malloc(my_n_images * sizeof(int));
        my_heights = (int*)malloc(my_n_images * sizeof(int));
        
        if (!my_pixels || !my_widths || !my_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Initialize local image dimensions and allocate buffers */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = my_start_image + i;
            my_widths[i] = image_widths[img_idx];
            my_heights[i] = image_heights[img_idx];
            
            /* Allocate memory for local image data */
            my_pixels[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
            
            if (!my_pixels[i]) {
                fprintf(stderr, "Process %d: Memory allocation failed for image %d\n", rank, i);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }
    
    /* Create array for scatterv/gatherv */
    pixel *all_pixels = NULL;
    pixel *scattered_pixels = NULL;
    int sendcount = 0;
    
    if (rank == 0) {
        /* Calculate total pixels for all processes */
        int total_pixels = 0;
        for (int i = 0; i < size; i++) {
            total_pixels += scatter_data->sendcounts[i];
        }
        
        /* Allocate buffer for all pixels */
        all_pixels = (pixel*)malloc(total_pixels * sizeof(pixel));
        if (!all_pixels) {
            fprintf(stderr, "Process 0: Memory allocation failed for all_pixels\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Copy all pixels to contiguous buffer */
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(&all_pixels[pixel_idx], image->p[i], 
                   image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
    }
    
    /* Allocate receive buffer for scattered pixels */
    sendcount = scatter_data->sendcounts[rank];
    scattered_pixels = (pixel*)malloc(sendcount * sizeof(pixel));
    if (!scattered_pixels && sendcount > 0) {
        fprintf(stderr, "Process %d: Memory allocation failed for scattered_pixels\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /* Scatter pixels to all processes using optimized MPI_Scatterv */
    // Use pre-calculated byte counts and displacements for more efficient transfers
    MPI_Scatterv(all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                 scattered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                 0, MPI_COMM_WORLD);
    
    /* Copy scattered pixels to individual image buffers */
    if (my_n_images > 0) {
        int pixel_offset = 0;
        for (int i = 0; i < my_n_images; i++) {
            int pixel_count = my_widths[i] * my_heights[i];
            memcpy(my_pixels[i], &scattered_pixels[pixel_offset], 
                   pixel_count * sizeof(pixel));
            pixel_offset += pixel_count;
        }
    }
    
    /* Free temporary scattered buffer */
    if (scattered_pixels) free(scattered_pixels);
    
    /* Apply the filters to the local images using CPU or GPU */
    if (my_n_images > 0) {
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < my_n_images; i++) {
            int pixel_count = my_widths[i] * my_heights[i];
            
            // Decide whether to use CPU or GPU based on image size and GPU availability
            bool use_gpu = false;
            if (pixel_count >= MIN_SIZE_FOR_GPU && my_gpu_count > 0) {
                int gpu_idx = omp_get_thread_num() % my_gpu_count;
                // Make sure the GPU resource is valid
                if (gpu_resources[gpu_idx] != NULL) {
                    int stream_idx = omp_get_thread_num() / my_gpu_count;
                    
                    // Try GPU processing
                    try {
                        process_image_cuda(my_pixels[i], my_widths[i], my_heights[i],
                                      gpu_resources[gpu_idx], stream_idx);
                        use_gpu = true;
                    } catch (...) {
                        fprintf(stderr, "Warning: GPU processing failed, falling back to CPU\n");
                    }
                }
            }
            
            // Fall back to CPU if GPU wasn't used
            if (!use_gpu) {
                process_image_cpu(my_pixels[i], my_widths[i], my_heights[i]);
            }
        }
    }
    
    /* Prepare buffer for gathering results */
    pixel *gathered_pixels = NULL;
    if (sendcount > 0) {
        gathered_pixels = (pixel*)malloc(sendcount * sizeof(pixel));
        if (!gathered_pixels) {
            fprintf(stderr, "Process %d: Memory allocation failed for gathered_pixels\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Copy processed pixels to contiguous buffer */
        int pixel_offset = 0;
        for (int i = 0; i < my_n_images; i++) {
            int pixel_count = my_widths[i] * my_heights[i];
            memcpy(&gathered_pixels[pixel_offset], my_pixels[i], 
                   pixel_count * sizeof(pixel));
            pixel_offset += pixel_count;
        }
    }
    
    /* Make sure all processes are synchronized before gather */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Debug information to check for buffer size mismatches */
    if (rank == 0 && SOBELF_DEBUG) {
        printf("Gathering data from processes: \n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d pixels (%d bytes) at displacement %d (%d bytes)\n", 
                   i, scatter_data->sendcounts[i], scatter_data->scatter_byte_counts[i],
                   scatter_data->displs[i], scatter_data->scatter_byte_displs[i]);
        }
    }
    
    /* Gather results back to master using optimized MPI_Gatherv */
    MPI_Gatherv(gathered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
                
    /* Master copies results back to image structure */
    if (rank == 0) {
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(image->p[i], &all_pixels[pixel_idx], 
                   image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
    }
    
    /* FILTER Timer stop */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("SOBEL done in %lf s\n", duration);
        
        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if (!store_pixels_optimized(output_filename, image)) { 
            free_resources(image->p, image->n_images);
            free(image->width);
            free(image->height);
            free(image);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1; 
        }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }
    
    /* Free resources */
    if (my_n_images > 0) {
        free_resources(my_pixels, my_n_images);
        free(my_widths);
        free(my_heights);
    }
    
    if (gathered_pixels) free(gathered_pixels);
    if (all_pixels) free(all_pixels);
    
    free_scatter_info(scatter_data);
    
    if (rank == 0 && image) {
        free_resources(image->p, image->n_images);
        free(image->width);
        free(image->height);
        free(image);
    }
    
    if (rank != 0) {
        free(image_widths);
        free(image_heights);
    }
    
    /* Free CUDA resources */
    if (gpu_resources) {
        for (int i = 0; i < my_gpu_count; i++) {
            free_cuda_resources(gpu_resources[i]);
        }
        free(gpu_resources);
    }
    
    MPI_Finalize();
    return 0;
}