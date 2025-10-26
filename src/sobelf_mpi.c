
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <gif_lib.h>  

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Represent one pixel from the image */
typedef struct pixel
{
    int r ; /* Red */
    int g ; /* Green */
    int b ; /* Blue */
} pixel ;

/* Represent one GIF image (animated or not) */
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
                "Error DGifSlurp: %d <%s>\n", error, GifErrorString(error) ) ;
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
                error2, GifErrorString(error2) ) ;
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


void
apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ )
        {
            int moy ;

            moy = (p[i][j].r + p[i][j].g + p[i][j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[i][j].r = moy ;
            p[i][j].g = moy ;
            p[i][j].b = moy ;
        }
    }
}

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

void apply_gray_line( animated_gif * image ) 
{
    int i, j, k ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < 10 ; j++ )
        {
            for ( k = image->width[i]/2 ; k < image->width[i] ; k++ )
            {
            p[i][CONV(j,k,image->width[i])].r = 0 ;
            p[i][CONV(j,k,image->width[i])].g = 0 ;
            p[i][CONV(j,k,image->width[i])].b = 0 ;
            }
        }
    }
}

void
apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i, j, k;
    int width, height;
    int end;
    int n_iter;
    int stencil_j, stencil_k;

    pixel ** p;
    pixel ** buffer;

    /* Get the pixels of all images */
    p = image->p;

    /* Allocate buffers for all images */
    buffer = (pixel **)malloc(image->n_images * sizeof(pixel *));
    if (buffer == NULL) {
        fprintf(stderr, "Unable to allocate memory for blur filter buffers\n");
        return;
    }
    
    for (i = 0; i < image->n_images; i++) {
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

    /* Process all images */
    for (i = 0; i < image->n_images; i++)
    {
        n_iter = 0;
        width = image->width[i];
        height = image->height[i];

        // Initialize buffer with original pixel values
        for (j = 0; j < height; j++) {
            for (k = 0; k < width; k++) {
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
            for (j = size; j < height - size; j++)
            {
                for (k = size; k < width - size; k++)
                {
                    int t_r = 0;
                    int t_g = 0;
                    int t_b = 0;
                    
                    // Compute stencil
                    for (stencil_j = -size; stencil_j <= size; stencil_j++)
                    {
                        for (stencil_k = -size; stencil_k <= size; stencil_k++)
                        {
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
            
            for (j = size; j < height - size; j++)
            {
                for (k = size; k < width - size; k++)
                {
                    int index = CONV(j, k, width);
                    int diff_r = buffer[i][index].r - p[i][index].r;
                    int diff_g = buffer[i][index].g - p[i][index].g;
                    int diff_b = buffer[i][index].b - p[i][index].b;

                    if (abs(diff_r) > threshold || abs(diff_g) > threshold || abs(diff_b) > threshold)
                    {
                        continue_flag = 1;
                    }

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
    for (i = 0; i < image->n_images; i++) {
        free(buffer[i]);
    }
    free(buffer);
}

void
apply_sobel_filter(animated_gif * image)
{
    int i, j, k;
    int width, height;
    
    float deltaX_red, deltaY_red, val_red;
    float deltaX_green, deltaY_green, val_green;
    float deltaX_blue, deltaY_blue, val_blue;
    float max_val;

    pixel ** p;
    pixel ** sobel_buffer;

    p = image->p;

    // Allocate buffers for all images
    sobel_buffer = (pixel **)malloc(image->n_images * sizeof(pixel *));
    if (sobel_buffer == NULL) {
        fprintf(stderr, "Unable to allocate memory for sobel filter buffers\n");
        return;
    }
    
    for (i = 0; i < image->n_images; i++) {
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

    // Process all images
    for (i = 0; i < image->n_images; i++)
    {
        width = image->width[i];
        height = image->height[i];
        
        // First, copy the original image to buffer to preserve it
        for (j = 0; j < height; j++)
        {
            for (k = 0; k < width; k++)
            {
                int index = CONV(j, k, width);
                sobel_buffer[i][index].r = p[i][index].r;
                sobel_buffer[i][index].g = p[i][index].g;
                sobel_buffer[i][index].b = p[i][index].b;
            }
        }
        
        // Apply Sobel filter
        for (j = 1; j < height - 1; j++)
        {
            for (k = 1; k < width - 1; k++)
            {
                // Get pixel values from the 3x3 neighborhood for each channel
                // Red channel
                int pixel_red_no = p[i][CONV(j-1, k-1, width)].r;
                int pixel_red_n  = p[i][CONV(j-1, k  , width)].r;
                int pixel_red_ne = p[i][CONV(j-1, k+1, width)].r;
                int pixel_red_o  = p[i][CONV(j  , k-1, width)].r;
                int pixel_red_e  = p[i][CONV(j  , k+1, width)].r;
                int pixel_red_so = p[i][CONV(j+1, k-1, width)].r;
                int pixel_red_s  = p[i][CONV(j+1, k  , width)].r;
                int pixel_red_se = p[i][CONV(j+1, k+1, width)].r;
                
                // Green channel
                int pixel_green_no = p[i][CONV(j-1, k-1, width)].g;
                int pixel_green_n  = p[i][CONV(j-1, k  , width)].g;
                int pixel_green_ne = p[i][CONV(j-1, k+1, width)].g;
                int pixel_green_o  = p[i][CONV(j  , k-1, width)].g;
                int pixel_green_e  = p[i][CONV(j  , k+1, width)].g;
                int pixel_green_so = p[i][CONV(j+1, k-1, width)].g;
                int pixel_green_s  = p[i][CONV(j+1, k  , width)].g;
                int pixel_green_se = p[i][CONV(j+1, k+1, width)].g;
                
                // Blue channel
                int pixel_blue_no = p[i][CONV(j-1, k-1, width)].b;
                int pixel_blue_n  = p[i][CONV(j-1, k  , width)].b;
                int pixel_blue_ne = p[i][CONV(j-1, k+1, width)].b;
                int pixel_blue_o  = p[i][CONV(j  , k-1, width)].b;
                int pixel_blue_e  = p[i][CONV(j  , k+1, width)].b;
                int pixel_blue_so = p[i][CONV(j+1, k-1, width)].b;
                int pixel_blue_s  = p[i][CONV(j+1, k  , width)].b;
                int pixel_blue_se = p[i][CONV(j+1, k+1, width)].b;

                // Calculate Sobel gradients for each channel
                // Red channel
                deltaX_red = -pixel_red_no + pixel_red_ne - 2*pixel_red_o + 2*pixel_red_e - pixel_red_so + pixel_red_se;
                deltaY_red = pixel_red_se + 2*pixel_red_s + pixel_red_so - pixel_red_ne - 2*pixel_red_n - pixel_red_no;
                val_red = sqrt(deltaX_red * deltaX_red + deltaY_red * deltaY_red)/4;
                
                // Green channel
                deltaX_green = -pixel_green_no + pixel_green_ne - 2*pixel_green_o + 2*pixel_green_e - pixel_green_so + pixel_green_se;
                deltaY_green = pixel_green_se + 2*pixel_green_s + pixel_green_so - pixel_green_ne - 2*pixel_green_n - pixel_green_no;
                val_green = sqrt(deltaX_green * deltaX_green + deltaY_green * deltaY_green)/4;
                
                // Blue channel
                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
                deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;

                // Calculate maximum gradient value across all channels for better edge detection
                max_val = val_red;
                if (val_green > max_val) max_val = val_green;
                if (val_blue > max_val) max_val = val_blue;

                // Apply binary thresholding for edge detection
                if (max_val > 15)  // Use threshold 15 to match sequential.c
                {
                    sobel_buffer[i][CONV(j, k, width)].r = 255;
                    sobel_buffer[i][CONV(j, k, width)].g = 255;
                    sobel_buffer[i][CONV(j, k, width)].b = 255;
                }
                else
                {
                    sobel_buffer[i][CONV(j, k, width)].r = 0;
                    sobel_buffer[i][CONV(j, k, width)].g = 0;
                    sobel_buffer[i][CONV(j, k, width)].b = 0;
                }
            }
        }
        
        // Copy enhanced image back to original image
        for (j = 0; j < height; j++)
        {
            for (k = 0; k < width; k++)
            {
                int index = CONV(j, k, width);
                p[i][index].r = sobel_buffer[i][index].r;
                p[i][index].g = sobel_buffer[i][index].g;
                p[i][index].b = sobel_buffer[i][index].b;
            }
        }
    }
    
    // Free the allocated buffers
    for (i = 0; i < image->n_images; i++) {
        free(sobel_buffer[i]);
    }
    free(sobel_buffer);
}



int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char *input_filename, *output_filename;
    animated_gif *image = NULL;
    int n_images = 0;
    struct timeval t1, t2;
    double duration;
    
    if (rank == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        input_filename = argv[1];
        output_filename = argv[2];
        
        gettimeofday(&t1, NULL);
        image = load_pixels(input_filename);
        if (!image) {
            fprintf(stderr, "Error loading GIF\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n_images = image->n_images;
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
               input_filename, image->n_images, duration);
    }
    

    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    

    MPI_Barrier(MPI_COMM_WORLD);
    

    int frames_per_proc = n_images / size;
    int remainder = n_images % size;
    int start = rank * frames_per_proc + (rank < remainder ? rank : remainder);
    int count = frames_per_proc + (rank < remainder ? 1 : 0);
    int end = start + count;
    

    animated_gif local_img;
    local_img.n_images = count;
    local_img.width = (int *)malloc(count * sizeof(int));
    local_img.height = (int *)malloc(count * sizeof(int));
    local_img.p = (pixel **)malloc(count * sizeof(pixel *));
    local_img.g = NULL;  
    
    if (rank == 0) {
    
        for (int i = start, j = 0; i < end; i++, j++) {
            local_img.width[j] = image->width[i];
            local_img.height[j] = image->height[i];
            int npixels = local_img.width[j] * local_img.height[j];
            local_img.p[j] = (pixel *)malloc(npixels * sizeof(pixel));
            memcpy(local_img.p[j], image->p[i], npixels * sizeof(pixel));
        }

        for (int r = 1; r < size; r++) {
            int r_start = r * frames_per_proc + (r < remainder ? r : remainder);
            int r_count = frames_per_proc + (r < remainder ? 1 : 0);
            for (int i = r_start; i < r_start + r_count; i++) {
                MPI_Send(&image->width[i], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(&image->height[i], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                int npixels = image->width[i] * image->height[i];
                MPI_Send(image->p[i], npixels * sizeof(pixel), MPI_BYTE, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {

        for (int j = 0; j < count; j++) {
            MPI_Recv(&local_img.width[j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&local_img.height[j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int npixels = local_img.width[j] * local_img.height[j];
            local_img.p[j] = (pixel *)malloc(npixels * sizeof(pixel));
            MPI_Recv(local_img.p[j], npixels * sizeof(pixel), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    
  
    MPI_Barrier(MPI_COMM_WORLD);

    gettimeofday(&t1, NULL);
    apply_gray_filter(&local_img);
    apply_blur_filter(&local_img, 3, 0);
    apply_sobel_filter(&local_img);
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
    if (rank == 0)
        printf("SOBEL done in %lf s\n", duration);
 
    MPI_Barrier(MPI_COMM_WORLD);
    
  
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            int r_start = r * frames_per_proc + (r < remainder ? r : remainder);
            int r_count = frames_per_proc + (r < remainder ? 1 : 0);
            for (int i = r_start, j = 0; i < r_start + r_count; i++, j++) {
                int width_frame, height_frame;
                MPI_Recv(&width_frame, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&height_frame, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int npixels = width_frame * height_frame;
                MPI_Recv(image->p[i], npixels * sizeof(pixel), MPI_BYTE, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
      
        for (int i = start, j = 0; i < end; i++, j++) {
            int npixels = image->width[i] * image->height[i];
            memcpy(image->p[i], local_img.p[j], npixels * sizeof(pixel));
        }
    } else {
        for (int j = 0; j < count; j++) {
            MPI_Send(&local_img.width[j], 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&local_img.height[j], 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            int npixels = local_img.width[j] * local_img.height[j];
            MPI_Send(local_img.p[j], npixels * sizeof(pixel), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }
    

    for (int j = 0; j < count; j++) {
        free(local_img.p[j]);
    }
    free(local_img.p);
    free(local_img.width);
    free(local_img.height);
    
  
    MPI_Barrier(MPI_COMM_WORLD);
    gettimeofday(&t1, NULL);
    if (rank == 0) {
        store_pixels(output_filename, image);
      
        for (int i = 0; i < n_images; i++) {
            free(image->p[i]);
        }
        free(image->p);
        free(image->width);
        free(image->height);
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
    if (rank == 0) {
        printf("Export done in %lf s in file %s\n", duration, output_filename);
    }
    
    MPI_Finalize();
    return 0;
}