#include "cuda_omp_mpi_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>
#include <stdexcept>  
#include "cuda_common.h"
#include "gif_lib.h"
#include "gif_io.h"

/* --------------------------------------------------------------------
   Macros and MPI tags
--------------------------------------------------------------------- */
#ifndef SOBELF_DEBUG
#define SOBELF_DEBUG 1
#endif

#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102

/* Minimum image size for GPU processing */
#define MIN_SIZE_FOR_GPU (128*128)
#define MAX_STREAMS_PER_GPU 4



/* Struct to send image data between processes */
typedef struct image_data {
    int width;
    int height;
    int image_idx;
    int total_size;
} image_data;



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



int run_cuda_omp_mpi_filter(char *input_filename, char *output_filename, int omp_threads) {
   
    animated_gif *image = NULL;
    struct timeval t1, t2;
    double duration;
    int rank, size;
    MPI_Status status;
    scatter_info *scatter_data = NULL;
    int gpu_count = 0;
    

    /* Initialize MPI with thread support for OpenMP and CUDA */
    int provided;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("DEBUG: MPI initialized, starting hybrid implementation\n");
    }

    
    if (rank == 0) {
        printf("DEBUG: Processing file %s -> %s\n", input_filename, output_filename);
    }
    
    
    if (omp_threads > 0)
        {
            omp_set_num_threads(omp_threads);
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
    
    return 0;
}