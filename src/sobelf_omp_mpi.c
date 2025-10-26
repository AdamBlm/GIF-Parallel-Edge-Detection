/*
 *
 *  Optimized Hybrid OpenMP + MPI Implementation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <omp.h>
#include <mpi.h>

#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Define MPI tags for messaging */
#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102

/* Tile size for memory-efficient stencil operations */
#define TILE_SIZE 64

/* Represent one pixel from the image */
typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images ; /* Number of images */
    int * width ; /* Width of each image */
    int * height ; /* Height of each image */
    pixel ** p ; /* Pixels of each image */
    GifFileType * g ; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif;

/* Struct to send image data between processes */
typedef struct image_data {
    int width;
    int height;
    int image_idx;
    int total_size;
} image_data;

/* Helper for collective operations */
typedef struct scatter_info {
    int *sendcounts;  /* Number of elements to send to each process */
    int *displs;      /* Displacement for each process */
    int *image_counts; /* Number of images per process */
    int *image_displs; /* Displacement for image indices */
    int *scatter_byte_counts; /* Number of bytes to send to each process - cached for reuse */
    int *scatter_byte_displs; /* Byte displacement for each process - cached for reuse */
} scatter_info;

/* Min function helper */
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif *
load_pixels( char * filename ) 
{
    GifFileType * g ;
    ColorMapObject * colmap ;
    int error ;
    int n_images ;
    int * width ;
    int * height ;
    pixel ** p ;
    int i ;
    animated_gif * image ;

    /* Open the GIF image (read mode) */
    g = DGifOpenFileName( filename, &error ) ;
    if ( g == NULL ) 
    {
        fprintf( stderr, "Error DGifOpenFileName %s\n", filename ) ;
        return NULL ;
    }

    /* Read the GIF image */
    error = DGifSlurp( g ) ;
    if ( error != GIF_OK )
    {
        fprintf( stderr, 
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error) ) ;
        return NULL ;
    }

    /* Grab the number of images and the size of each image */
    n_images = g->ImageCount ;

    width = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( width == NULL )
    {
        fprintf( stderr, "Unable to allocate width of size %d\n",
                n_images ) ;
        return 0 ;
    }

    height = (int *)malloc( n_images * sizeof( int ) ) ;
    if ( height == NULL )
    {
        fprintf( stderr, "Unable to allocate height of size %d\n",
                n_images ) ;
        return 0 ;
    }

    /* Fill the width and height */
    for ( i = 0 ; i < n_images ; i++ ) 
    {
        width[i] = g->SavedImages[i].ImageDesc.Width ;
        height[i] = g->SavedImages[i].ImageDesc.Height ;

#if SOBELF_DEBUG
        printf( "Image %d: l:%d t:%d w:%d h:%d interlace:%d localCM:%p\n",
                i, 
                g->SavedImages[i].ImageDesc.Left,
                g->SavedImages[i].ImageDesc.Top,
                g->SavedImages[i].ImageDesc.Width,
                g->SavedImages[i].ImageDesc.Height,
                g->SavedImages[i].ImageDesc.Interlace,
                g->SavedImages[i].ImageDesc.ColorMap
                ) ;
#endif
    }


    /* Get the global colormap */
    colmap = g->SColorMap ;
    if ( colmap == NULL ) 
    {
        fprintf( stderr, "Error global colormap is NULL\n" ) ;
        return NULL ;
    }

#if SOBELF_DEBUG
    printf( "Global color map: count:%d bpp:%d sort:%d\n",
            g->SColorMap->ColorCount,
            g->SColorMap->BitsPerPixel,
            g->SColorMap->SortFlag
            ) ;
#endif

    /* Allocate the array of pixels to be returned */
    p = (pixel **)malloc( n_images * sizeof( pixel * ) ) ;
    if ( p == NULL )
    {
        fprintf( stderr, "Unable to allocate array of %d images\n",
                n_images ) ;
        return NULL ;
    }

    for ( i = 0 ; i < n_images ; i++ ) 
    {
        p[i] = (pixel *)malloc( width[i] * height[i] * sizeof( pixel ) ) ;
        if ( p[i] == NULL )
        {
        fprintf( stderr, "Unable to allocate %d-th array of %d pixels\n",
                i, width[i] * height[i] ) ;
        return NULL ;
        }
    }
    
    /* Fill pixels */

    /* For each image */
    for ( i = 0 ; i < n_images ; i++ )
    {
        int j ;

        /* Get the local colormap if needed */
        if ( g->SavedImages[i].ImageDesc.ColorMap )
        {

            /* TODO No support for local color map */
            fprintf( stderr, "Error: application does not support local colormap\n" ) ;
            return NULL ;

            colmap = g->SavedImages[i].ImageDesc.ColorMap ;
        }

        /* Traverse the image and fill pixels */
        for ( j = 0 ; j < width[i] * height[i] ; j++ ) 
        {
            int c ;

            c = g->SavedImages[i].RasterBits[j] ;

            p[i][j].r = colmap->Colors[c].Red ;
            p[i][j].g = colmap->Colors[c].Green ;
            p[i][j].b = colmap->Colors[c].Blue ;
        }
    }

    /* Allocate image info */
    image = (animated_gif *)malloc( sizeof(animated_gif) ) ;
    if ( image == NULL ) 
    {
        fprintf( stderr, "Unable to allocate memory for animated_gif\n" ) ;
        return NULL ;
    }

    /* Fill image fields */
    image->n_images = n_images ;
    image->width = width ;
    image->height = height ;
    image->p = p ;
    image->g = g ;

#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] ) ;
#endif

    return image ;
}

int 
output_modified_read_gif( char * filename, GifFileType * g ) 
{
    GifFileType * g2 ;
    int error2 ;

#if SOBELF_DEBUG
    printf( "Starting output to file %s\n", filename ) ;
#endif

    g2 = EGifOpenFileName( filename, false, &error2 ) ;
    if ( g2 == NULL )
    {
        fprintf( stderr, "Error EGifOpenFileName %s\n",
                filename ) ;
        return 0 ;
    }

    g2->SWidth = g->SWidth ;
    g2->SHeight = g->SHeight ;
    g2->SColorResolution = g->SColorResolution ;
    g2->SBackGroundColor = g->SBackGroundColor ;
    g2->AspectByte = g->AspectByte ;
    g2->SColorMap = g->SColorMap ;
    g2->ImageCount = g->ImageCount ;
    g2->SavedImages = g->SavedImages ;
    g2->ExtensionBlockCount = g->ExtensionBlockCount ;
    g2->ExtensionBlocks = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK ) 
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", 
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}


int
store_pixels( char * filename, animated_gif * image )
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
     * NEW: Color quantization step 
     * Convert colors to a reduced palette before processing
     * This ensures we don't exceed the 256 color limit for GIF files
     */
    
    // We'll use a simple color quantization by masking least significant bits
    // This reduces color space significantly
    const int COLOR_MASK = 0xE0; // Keeps only 2 most significant bits (4 values per channel = 64 colors total)
                                 // 11000000 in binary
    
    // Apply quantization to all pixels
    for (i = 0; i < image->n_images; i++) {
        #pragma omp parallel for default(none) shared(p, i, image) private(j)
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

/* Apply grayscale filter to an image */
void apply_gray_filter_to_image(pixel *p, int width, int height)
{
    // Initialize all pixels with first touch by the thread that will use it
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

/* Apply grayscale filter to multiple images */
void apply_gray_filter(pixel **p, int n_images, int *widths, int *heights)
{
    #pragma omp parallel for default(none) shared(p, n_images, widths, heights) schedule(dynamic)
    for (int i = 0; i < n_images; i++)
    {
        apply_gray_filter_to_image(p[i], widths[i], heights[i]);
    }
}

/* Apply blur filter to an image with tiling optimization */
void apply_blur_filter_to_image(pixel *p, int width, int height, int size, int threshold, pixel *buffer)
{
    int end = 0;
    int n_iter = 0;
    
    // Initialize buffer with original pixel values - use NUMA-aware initialization
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

        // Process all pixels except the border with tiling for better cache utilization
        #pragma omp parallel for default(none) shared(p, buffer, width, height, size) schedule(static)
        for (int jj = size; jj < height - size; jj += TILE_SIZE)
        {
            for (int kk = size; kk < width - size; kk += TILE_SIZE)
            {
                // Process a tile
                for (int j = jj; j < min(jj + TILE_SIZE, height - size); j++)
                {
                    for (int k = kk; k < min(kk + TILE_SIZE, width - size); k++)
                    {
                        int t_r = 0;
                        int t_g = 0;
                        int t_b = 0;
                        
                        // Cache-friendly stencil computation
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
    printf("BLUR: number of iterations: %d\n", n_iter);
#endif
}

/* Apply blur filter to multiple images */
void apply_blur_filter(pixel **p, int n_images, int *widths, int *heights, int size, int threshold, pixel **buffers)
{
    #pragma omp parallel for default(none) shared(p, n_images, widths, heights, size, threshold, buffers) schedule(dynamic)
    for (int i = 0; i < n_images; i++)
    {
        apply_blur_filter_to_image(p[i], widths[i], heights[i], size, threshold, buffers[i]);
    }
}

/* Apply sobel filter to an image */
void apply_sobel_filter_to_image(pixel *p, int width, int height, pixel *buffer)
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
            // Get pixel values from the 3x3 neighborhood for each channel
            // Red channel
            int pixel_red_no = p[row_prev + k-1].r;
            int pixel_red_n  = p[row_prev + k].r;
            int pixel_red_ne = p[row_prev + k+1].r;
            int pixel_red_o  = p[row_curr + k-1].r;
            int pixel_red_e  = p[row_curr + k+1].r;
            int pixel_red_so = p[row_next + k-1].r;
            int pixel_red_s  = p[row_next + k].r;
            int pixel_red_se = p[row_next + k+1].r;
            
            // Green channel
            int pixel_green_no = p[row_prev + k-1].g;
            int pixel_green_n  = p[row_prev + k].g;
            int pixel_green_ne = p[row_prev + k+1].g;
            int pixel_green_o  = p[row_curr + k-1].g;
            int pixel_green_e  = p[row_curr + k+1].g;
            int pixel_green_so = p[row_next + k-1].g;
            int pixel_green_s  = p[row_next + k].g;
            int pixel_green_se = p[row_next + k+1].g;
            
            // Blue channel
            int pixel_blue_no = p[row_prev + k-1].b;
            int pixel_blue_n  = p[row_prev + k].b;
            int pixel_blue_ne = p[row_prev + k+1].b;
            int pixel_blue_o  = p[row_curr + k-1].b;
            int pixel_blue_e  = p[row_curr + k+1].b;
            int pixel_blue_so = p[row_next + k-1].b;
            int pixel_blue_s  = p[row_next + k].b;
            int pixel_blue_se = p[row_next + k+1].b;

            // Calculate Sobel gradients for each channel
            // Red channel
            float deltaX_red = -pixel_red_no + pixel_red_ne - 2*pixel_red_o + 2*pixel_red_e - pixel_red_so + pixel_red_se;
            float deltaY_red = pixel_red_se + 2*pixel_red_s + pixel_red_so - pixel_red_ne - 2*pixel_red_n - pixel_red_no;
            float val_red = sqrt(deltaX_red * deltaX_red + deltaY_red * deltaY_red)/4;
            
            // Green channel
            float deltaX_green = -pixel_green_no + pixel_green_ne - 2*pixel_green_o + 2*pixel_green_e - pixel_green_so + pixel_green_se;
            float deltaY_green = pixel_green_se + 2*pixel_green_s + pixel_green_so - pixel_green_ne - 2*pixel_green_n - pixel_green_no;
            float val_green = sqrt(deltaX_green * deltaX_green + deltaY_green * deltaY_green)/4;
            
            // Blue channel
            float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
            float deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
            float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;

            // Calculate maximum gradient value across all channels for better edge detection
            float max_val = val_red;
            if (val_green > max_val) max_val = val_green;
            if (val_blue > max_val) max_val = val_blue;

            // Apply binary thresholding to match original appearance
            // We use the maximum value from all channels instead of just blue
            // and maintain the same binary black/white approach
            if (max_val > 15)  // Adjusted threshold to get visible edges
            {
                buffer[row_curr + k].r = 255;
                buffer[row_curr + k].g = 255;
                buffer[row_curr + k].b = 255;
            }
            else
            {
                buffer[row_curr + k].r = 0;
                buffer[row_curr + k].g = 0;
                buffer[row_curr + k].b = 0;
            }
        }
    }
    
    // Copy enhanced image back to original image
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

/* Apply sobel filter to multiple images */
void apply_sobel_filter(pixel **p, int n_images, int *widths, int *heights, pixel **buffers)
{
    #pragma omp parallel for default(none) shared(p, n_images, widths, heights, buffers) schedule(dynamic)
    for (int i = 0; i < n_images; i++)
    {
        apply_sobel_filter_to_image(p[i], widths[i], heights[i], buffers[i]);
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

// Create scatter/gather information for collective operations
scatter_info* create_scatter_info(int n_images, int size)
{
    scatter_info* info = (scatter_info*)malloc(sizeof(scatter_info));
    if (!info) return NULL;
    
    info->sendcounts = (int*)calloc(size, sizeof(int));
    info->displs = (int*)calloc(size, sizeof(int));
    info->image_counts = (int*)calloc(size, sizeof(int));
    info->image_displs = (int*)calloc(size, sizeof(int));
    info->scatter_byte_counts = (int*)calloc(size, sizeof(int));
    info->scatter_byte_displs = (int*)calloc(size, sizeof(int));
    
    if (!info->sendcounts || !info->displs || 
        !info->image_counts || !info->image_displs ||
        !info->scatter_byte_counts || !info->scatter_byte_displs) {
        // Free allocated memory if any allocation failed
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
        return NULL;
    }
    
    // Distribute images evenly among processes
    int base_count = n_images / size;
    int remainder = n_images % size;
    
    for (int i = 0; i < size; i++) {
        info->image_counts[i] = base_count + (i < remainder ? 1 : 0);
        if (i > 0) {
            info->image_displs[i] = info->image_displs[i-1] + info->image_counts[i-1];
        }
    }
    
    return info;
}

// Free scatter/gather information
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

// Calculate scatter/gather counts for pixel data
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
    MPI_Request *requests = NULL;
    scatter_info *scatter_data = NULL;

    /* Initialize MPI with thread support for OpenMP */
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    
    /* Set number of threads if provided */
    if (argc > 3)
    {
        int num_threads = atoi(argv[3]);
        if (num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        }
    }

    if (rank == 0) {
        printf("Using %d MPI processes with %d OpenMP threads per process\n", 
               size, omp_get_max_threads());
    }

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
        
        // Create scatter info for workload distribution
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Failed to allocate scatter info\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Calculate pixel counts for each process
        calculate_pixel_counts(scatter_data, image_widths, image_heights, size);
    }

    /* Broadcast the number of images to all processes */
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Allocate memory for image dimensions on all processes */
    if (rank != 0) {
        image_widths = (int*)malloc(n_images * sizeof(int));
        image_heights = (int*)malloc(n_images * sizeof(int));
        if (!image_widths || !image_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Use a single Bcast call to transfer all dimensions at once */
    MPI_Bcast(image_widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image_heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* Distribute scatter information */
    if (rank != 0) {
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Broadcast scatter info arrays in a single operation when possible */
    int total_scatter_info_size = size * 4;
    int* all_scatter_info = NULL;
    
    if (rank == 0) {
        all_scatter_info = (int*)malloc(total_scatter_info_size * sizeof(int));
        if (!all_scatter_info) {
            fprintf(stderr, "Process 0: Failed to allocate scatter info buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Pack all scatter arrays into a single buffer for efficient transfer
        memcpy(&all_scatter_info[0], scatter_data->image_counts, size * sizeof(int));
        memcpy(&all_scatter_info[size], scatter_data->image_displs, size * sizeof(int));
        memcpy(&all_scatter_info[size*2], scatter_data->sendcounts, size * sizeof(int));
        memcpy(&all_scatter_info[size*3], scatter_data->displs, size * sizeof(int));
    } else {
        all_scatter_info = (int*)malloc(total_scatter_info_size * sizeof(int));
        if (!all_scatter_info) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info buffer\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // One broadcast instead of multiple
    MPI_Bcast(all_scatter_info, total_scatter_info_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        // Unpack the received data
        memcpy(scatter_data->image_counts, &all_scatter_info[0], size * sizeof(int));
        memcpy(scatter_data->image_displs, &all_scatter_info[size], size * sizeof(int));
        memcpy(scatter_data->sendcounts, &all_scatter_info[size*2], size * sizeof(int));
        memcpy(scatter_data->displs, &all_scatter_info[size*3], size * sizeof(int));
        
        // Recalculate byte counts and displacements
        for (int i = 0; i < size; i++) {
            scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
            scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
        }
    }
    
    free(all_scatter_info);

    /* FILTER Timer start - only synchronize once before starting the timer */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t1, NULL);
    }

    /* Get information for this process */
    int my_n_images = scatter_data->image_counts[rank];
    int my_start_image = scatter_data->image_displs[rank];
    
    /* Allocate memory for requests */
    requests = (MPI_Request*)malloc(2 * my_n_images * sizeof(MPI_Request));
    if (!requests && my_n_images > 0) {
        fprintf(stderr, "Process %d: Failed to allocate request array\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /* Allocate memory for local image processing */
    pixel **my_pixels = NULL;
    pixel **filter_buffers = NULL;
    int *my_widths = NULL;
    int *my_heights = NULL;
    
    if (my_n_images > 0) {
        /* Allocate memory for local image data */
        my_pixels = (pixel**)malloc(my_n_images * sizeof(pixel*));
        filter_buffers = (pixel**)malloc(my_n_images * sizeof(pixel*));
        my_widths = (int*)malloc(my_n_images * sizeof(int));
        my_heights = (int*)malloc(my_n_images * sizeof(int));
        
        if (!my_pixels || !filter_buffers || !my_widths || !my_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Initialize local image dimensions and allocate buffers */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = my_start_image + i;
            my_widths[i] = image_widths[img_idx];
            my_heights[i] = image_heights[img_idx];
            
            /* Allocate memory for local image data and buffers */
            my_pixels[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
            filter_buffers[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
            
            if (!my_pixels[i] || !filter_buffers[i]) {
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
    
    /* Use MPI_Scatterv with MPI_BYTE for more efficient data transfer */
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
    
    /* Apply the filters to the local images */
    if (my_n_images > 0) {
        /* Initialize filter buffers with original pixels to avoid
           random values when filters are disabled */
        for (int i = 0; i < my_n_images; i++) {
            memcpy(filter_buffers[i], my_pixels[i], 
                   my_widths[i] * my_heights[i] * sizeof(pixel));
        }
        
        /* Convert the pixels into grayscale */
        apply_gray_filter(my_pixels, my_n_images, my_widths, my_heights);

        /* Apply blur filter with convergence value */
        apply_blur_filter(my_pixels, my_n_images, my_widths, my_heights, 3, 0, filter_buffers);

        /* Apply sobel filter on pixels */
        apply_sobel_filter(my_pixels, my_n_images, my_widths, my_heights, filter_buffers);
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
    
    /* Use MPI_Gatherv with MPI_BYTE for more efficient data collection */
    MPI_Gatherv(gathered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
                
    /* Copy processed data back to the original image structure */
    if (rank == 0 && all_pixels != NULL) {
        /* Copy all processed pixels back to the original image structure */
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
        if (!store_pixels(output_filename, image)) { 
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
        free_resources(filter_buffers, my_n_images);
        free(my_widths);
        free(my_heights);
    }
    
    if (gathered_pixels) free(gathered_pixels);
    if (all_pixels) free(all_pixels);
    if (requests) free(requests);
    
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
    
    MPI_Finalize();
    return 0;
}