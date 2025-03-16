/*
 * INF560
 *
 * Image Filtering Project - Optimized OpenMP Implementation
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;

/* Represent one GIF image (animated or not */
typedef struct animated_gif
{
    int n_images ; /* Number of images */
    int * width ; /* Width of each image */
    int * height ; /* Height of each image */
    pixel ** p ; /* Pixels of each image */
    GifFileType * g ; /* Internal representation.
                         DO NOT MODIFY */
} animated_gif ;

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

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

/* Apply grayscale filter to the image */
void apply_gray_filter(animated_gif * image)
{
    pixel ** p;
    p = image->p;

    // Process each image sequentially (will be parallelized with MPI later)
    for (int i = 0; i < image->n_images; i++)
    {
        int width = image->width[i];
        int height = image->height[i];
        
        // Parallelize at the row level for better load balancing
        #pragma omp parallel for default(none) shared(p, i, width, height) schedule(static)
        for (int j = 0; j < height; j++)
        {
            // Process each pixel in the row
            for (int k = 0; k < width; k++)
            {
                int index = CONV(j, k, width);
                int moy = (p[i][index].r + p[i][index].g + p[i][index].b)/3;
                if (moy < 0) moy = 0;
                if (moy > 255) moy = 255;

                p[i][index].r = moy;
                p[i][index].g = moy;
                p[i][index].b = moy;
            }
        }
    }
}

/* Apply blur filter with improved parallelization */
void apply_blur_filter(animated_gif * image, int size, int threshold)
{
    pixel ** p;
    pixel ** buffer;
    p = image->p;
    
    // Pre-allocate buffers outside of parallel region
    buffer = (pixel **)malloc(image->n_images * sizeof(pixel *));
    if (buffer == NULL) {
        fprintf(stderr, "Unable to allocate memory for blur filter buffers\n");
        return;
    }
    
    for (int i = 0; i < image->n_images; i++) {
        buffer[i] = (pixel *)malloc(image->width[i] * image->height[i] * sizeof(pixel));
        if (buffer[i] == NULL) {
            fprintf(stderr, "Unable to allocate memory for blur filter buffer %d\n", i);
            // Free previously allocated buffers
            for (int j = 0; j < i; j++) {
                free(buffer[j]);
            }
            free(buffer);
            return;
        }
    }

    // Process each image sequentially (will be parallelized with MPI later)
    for (int i = 0; i < image->n_images; i++)
    {
        int width = image->width[i];
        int height = image->height[i];
        int end = 0;
        int n_iter = 0;
        
        // Initialize buffer with original pixel values
        #pragma omp parallel for default(none) shared(p, buffer, i, width, height) schedule(static)
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {
                int index = CONV(j, k, width);
                buffer[i][index].r = p[i][index].r;
                buffer[i][index].g = p[i][index].g;
                buffer[i][index].b = p[i][index].b;
            }
        }

        /* Perform blur iterations until convergence */
        do
        {
            end = 1;
            n_iter++;

            // Process all pixels except the border
            #pragma omp parallel for default(none) shared(p, buffer, i, width, height, size) schedule(static)
            for (int j = size; j < height - size; j++)
            {
                // Process each pixel in the row
                for (int k = size; k < width - size; k++)
                {
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;
                    
                    // Use a more cache-friendly access pattern for stencil
                    for (int stencil_j = -size; stencil_j <= size; stencil_j++)
                    {
                        for (int stencil_k = -size; stencil_k <= size; stencil_k++)
                        {
                            // Calculer correctement l'indice avec la macro CONV
                            int idx = CONV(j+stencil_j, k+stencil_k, width);
                            t_r += p[i][idx].r;
                            t_g += p[i][idx].g;
                            t_b += p[i][idx].b;
                        }
                    }

                    // Calculate the average value for the pixel
                    const int denom = (2*size+1)*(2*size+1);
                    buffer[i][CONV(j, k, width)].r = t_r / denom;
                    buffer[i][CONV(j, k, width)].g = t_g / denom;
                    buffer[i][CONV(j, k, width)].b = t_b / denom;
                }
            }

            // Check convergence and update pixels
            int continue_flag = 0;
            
            // Update pixels and check convergence
            #pragma omp parallel for default(none) shared(p, buffer, i, width, height, size, threshold) reduction(|:continue_flag) schedule(static)
            for (int j = size; j < height - size; j++)
            {
                for (int k = size; k < width - size; k++)
                {
                    // Calculate difference between old and new pixels
                    int index = CONV(j, k, width);
                    int diff_r = buffer[i][index].r - p[i][index].r;
                    int diff_g = buffer[i][index].g - p[i][index].g;
                    int diff_b = buffer[i][index].b - p[i][index].b;

                    // Check if we need to continue iterations
                    if (abs(diff_r) > threshold || abs(diff_g) > threshold || abs(diff_b) > threshold)
                    {
                        continue_flag = 1;
                    }

                    // Update original pixels with new values
                    p[i][index].r = buffer[i][index].r;
                    p[i][index].g = buffer[i][index].g;
                    p[i][index].b = buffer[i][index].b;
                }
            }
            
            end = !continue_flag;
        }
        while (threshold > 0 && !end);

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d: %d\n", i, n_iter);
#endif
    }
    
    // Free the allocated buffers
    for (int i = 0; i < image->n_images; i++) {
        free(buffer[i]);
    }
    free(buffer);
}

/* Apply Sobel filter with optimized implementation */
void apply_sobel_filter(animated_gif * image)
{
    pixel ** p;
    pixel ** sobel_buffer;
    p = image->p;

    // Pre-allocate buffers outside of parallel region
    sobel_buffer = (pixel **)malloc(image->n_images * sizeof(pixel *));
    if (sobel_buffer == NULL) {
        fprintf(stderr, "Unable to allocate memory for sobel filter buffers\n");
        return;
    }
    
    for (int i = 0; i < image->n_images; i++) {
        sobel_buffer[i] = (pixel *)malloc(image->width[i] * image->height[i] * sizeof(pixel));
        if (sobel_buffer[i] == NULL) {
            fprintf(stderr, "Unable to allocate memory for sobel filter buffer %d\n", i);
            // Free previously allocated buffers
            for (int j = 0; j < i; j++) {
                free(sobel_buffer[j]);
            }
            free(sobel_buffer);
            return;
        }
    }

    // Process each image sequentially (will be parallelized with MPI later)
    for (int i = 0; i < image->n_images; i++)
    {
        int width = image->width[i];
        int height = image->height[i];
        
        // Apply Sobel filter with improved memory access pattern
        #pragma omp parallel for default(none) shared(p, sobel_buffer, i, width, height) schedule(static)
        for (int j = 1; j < height - 1; j++)
        {
            // Calculate row offsets for better cache locality
            const int row_prev = CONV(j-1, 0, width);
            const int row_curr = CONV(j, 0, width);
            const int row_next = CONV(j+1, 0, width);
            
            for (int k = 1; k < width - 1; k++)
            {
                // Get pixel values from the 3x3 neighborhood with better locality
                int pixel_blue_no = p[i][row_prev + k-1].b;
                int pixel_blue_n  = p[i][row_prev + k].b;
                int pixel_blue_ne = p[i][row_prev + k+1].b;
                int pixel_blue_o  = p[i][row_curr + k-1].b;
                int pixel_blue_e  = p[i][row_curr + k+1].b;
                int pixel_blue_so = p[i][row_next + k-1].b;
                int pixel_blue_s  = p[i][row_next + k].b;
                int pixel_blue_se = p[i][row_next + k+1].b;

                // Calculate Sobel gradients
                float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
                float deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
                float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;

                // Apply threshold to create binary edge image
                if (val_blue > 50)
                {
                    sobel_buffer[i][row_curr + k].r = 255;
                    sobel_buffer[i][row_curr + k].g = 255;
                    sobel_buffer[i][row_curr + k].b = 255;
                }
                else
                {
                    sobel_buffer[i][row_curr + k].r = 0;
                    sobel_buffer[i][row_curr + k].g = 0;
                    sobel_buffer[i][row_curr + k].b = 0;
                }
            }
        }
        
        // Copy sobel results back to original image
        #pragma omp parallel for default(none) shared(p, sobel_buffer, i, width, height) schedule(static)
        for (int j = 1; j < height - 1; j++)
        {
            for (int k = 1; k < width - 1; k++)
            {
                int index = CONV(j, k, width);
                p[i][index].r = sobel_buffer[i][index].r;
                p[i][index].g = sobel_buffer[i][index].g;
                p[i][index].b = sobel_buffer[i][index].b;
            }
        }
    }
    
    // Free the allocated buffers
    for (int i = 0; i < image->n_images; i++) {
        free(sobel_buffer[i]);
    }
    free(sobel_buffer);
}

/*
 * Main entry point
 */
int main(int argc, char **argv)
{
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;

    /* Check command-line arguments */
    if(argc < 3)
    {
        fprintf(stderr, "Usage: %s input.gif output.gif [num_threads]\n", argv[0]);
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];
    
    /* Set number of threads if provided */
    if(argc > 3)
    {
        int num_threads = atoi(argv[3]);
        if(num_threads > 0)
        {
            omp_set_num_threads(num_threads);
        }
    }

    printf("Using %d OpenMP threads\n", omp_get_max_threads());

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if(image == NULL) { return 1; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration);

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale */
    apply_gray_filter(image);

    /* Apply blur filter with convergence value */
    apply_blur_filter(image, 5, 20);

    /* Apply sobel filter on pixels */
    apply_sobel_filter(image);

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("SOBEL done in %lf s\n", duration);

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if(!store_pixels(output_filename, image)) { return 1; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}
