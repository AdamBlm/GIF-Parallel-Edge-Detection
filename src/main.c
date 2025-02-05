/*
 * INF560
 *
 * Image Filtering Project (OpenMP version)
 *
 * This code closely matches the provided sequential version but uses
 * OpenMP pragmas for parallel processing. Compile with -fopenmp.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>          /* <-- Added for OpenMP */
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

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images ;    /* Number of images           */
    int * width ;     /* Width of each image        */
    int * height ;    /* Height of each image       */
    pixel ** p ;      /* Pixels of each image       */
    GifFileType * g ; /* Internal representation.
                         DO NOT MODIFY              */
} animated_gif ;

/* Load a GIF image from a file and return a structure of type animated_gif */
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

    /* Fill the width and height arrays */
    for ( i = 0 ; i < n_images ; i++ ) 
    {
        width[i]  = g->SavedImages[i].ImageDesc.Width ;
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
    for ( i = 0 ; i < n_images ; i++ )
    {
        int j ;

        /* If there is a local colormap, not supported by this code */
        if ( g->SavedImages[i].ImageDesc.ColorMap )
        {
            /* For simplicity, we do not handle local color maps */
            fprintf( stderr, "Error: application does not support local colormap\n" ) ;
            return NULL ;
        }

        for ( j = 0 ; j < width[i] * height[i] ; j++ ) 
        {
            int c = g->SavedImages[i].RasterBits[j] ;

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
    image->width    = width ;
    image->height   = height ;
    image->p        = p ;
    image->g        = g ;

#if SOBELF_DEBUG
    printf( "-> GIF w/ %d image(s) with first image of size %d x %d\n",
            image->n_images, image->width[0], image->height[0] ) ;
#endif

    return image ;
}

/* Wrapper to re-output the modified GIF data using giflib */
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

    g2->SWidth           = g->SWidth ;
    g2->SHeight          = g->SHeight ;
    g2->SColorResolution = g->SColorResolution ;
    g2->SBackGroundColor = g->SBackGroundColor ;
    g2->AspectByte       = g->AspectByte ;
    g2->SColorMap        = g->SColorMap ;
    g2->ImageCount       = g->ImageCount ;
    g2->SavedImages      = g->SavedImages ;
    g2->ExtensionBlockCount = g->ExtensionBlockCount ;
    g2->ExtensionBlocks     = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK ) 
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", 
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}

/* Store modified pixels into a GIF file on disk */
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
        colormap[i].Red   = 255 ;
        colormap[i].Green = 255 ;
        colormap[i].Blue  = 255 ;
    }

    /* Convert background color to grayscale */
    {
        int moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
        ) / 3 ;
        if ( moy < 0 )   moy = 0 ;
        if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
        printf( "[DEBUG] Background color (%d,%d,%d) -> (%d,%d,%d)\n",
                image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red,
                image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green,
                image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue,
                moy, moy, moy ) ;
#endif

        colormap[0].Red   = moy ;
        colormap[0].Green = moy ;
        colormap[0].Blue  = moy ;

        image->g->SBackGroundColor = 0 ;
        n_colors++ ;
    }

    /* Process extension blocks in the main structure (transparency) */
    for ( j = 0 ; j < image->g->ExtensionBlockCount ; j++ )
    {
        int f = image->g->ExtensionBlocks[j].Function ;
        if ( f == GRAPHICS_EXT_FUNC_CODE )
        {
            int tr_color = image->g->ExtensionBlocks[j].Bytes[3] ;
            if ( tr_color >= 0 && tr_color < 255 )
            {
                /* Convert transparency color to grayscale and insert into color map */
                int moy = (
                    image->g->SColorMap->Colors[ tr_color ].Red +
                    image->g->SColorMap->Colors[ tr_color ].Green +
                    image->g->SColorMap->Colors[ tr_color ].Blue
                ) / 3 ;
                if ( moy < 0 )   moy = 0 ;
                if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                printf( "[DEBUG] Transparency color in main ext block -> %d\n", moy );
#endif

                /* Check if this gray is already in colormap */
                int found = -1 ;
                for ( k = 0 ; k < n_colors ; k++ )
                {
                    if ( moy == colormap[k].Red &&
                         moy == colormap[k].Green &&
                         moy == colormap[k].Blue )
                    {
                        found = k ;
                        break ;
                    }
                }
                if ( found == -1 )
                {
                    if ( n_colors >= 256 ) 
                    {
                        fprintf( stderr, 
                                 "Error: too many colors inside the image\n") ;
                        return 0 ;
                    }
                    colormap[n_colors].Red   = moy ;
                    colormap[n_colors].Green = moy ;
                    colormap[n_colors].Blue  = moy ;
                    image->g->ExtensionBlocks[j].Bytes[3] = n_colors ;
                    n_colors++ ;
                }
                else
                {
                    image->g->ExtensionBlocks[j].Bytes[3] = found ;
                }
            }
        }
    }

    /* Process extension blocks in each frame (similarly) */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->g->SavedImages[i].ExtensionBlockCount ; j++ )
        {
            int f = image->g->SavedImages[i].ExtensionBlocks[j].Function ;
            if ( f == GRAPHICS_EXT_FUNC_CODE )
            {
                int tr_color = image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] ;
                if ( tr_color >= 0 && tr_color < 255 )
                {
                    int moy = (
                        image->g->SColorMap->Colors[ tr_color ].Red +
                        image->g->SColorMap->Colors[ tr_color ].Green +
                        image->g->SColorMap->Colors[ tr_color ].Blue
                    ) / 3 ;
                    if ( moy < 0 )   moy = 0 ;
                    if ( moy > 255 ) moy = 255 ;

#if SOBELF_DEBUG
                    printf("[DEBUG] Transparency color in frame %d -> %d\n", i, moy );
#endif

                    int found = -1 ;
                    for ( k = 0 ; k < n_colors ; k++ )
                    {
                        if ( moy == colormap[k].Red &&
                             moy == colormap[k].Green &&
                             moy == colormap[k].Blue )
                        {
                            found = k ;
                            break ;
                        }
                    }
                    if ( found == -1 )
                    {
                        if ( n_colors >= 256 ) 
                        {
                            fprintf( stderr, 
                                     "Error: too many colors inside the image\n") ;
                            return 0 ;
                        }
                        colormap[n_colors].Red   = moy ;
                        colormap[n_colors].Green = moy ;
                        colormap[n_colors].Blue  = moy ;
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = n_colors ;
                        n_colors++ ;
                    }
                    else
                    {
                        image->g->SavedImages[i].ExtensionBlocks[j].Bytes[3] = found ;
                    }
                }
            }
        }
    }

#if SOBELF_DEBUG
    printf( "[DEBUG] Number of colors after background/transparency: %d\n",
            n_colors ) ;
#endif

    p = image->p ;

    /* Insert the actual image colors into colormap */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
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
                    break ;
                }
            }

            if ( !found ) 
            {
                if ( n_colors >= 256 ) 
                {
                    fprintf( stderr, 
                             "Error: too many colors inside the image\n") ;
                    return 0 ;
                }

#if SOBELF_DEBUG
                printf( "[DEBUG] Found new color %d -> (%d,%d,%d)\n",
                        n_colors, p[i][j].r, p[i][j].g, p[i][j].b ) ;
#endif

                colormap[n_colors].Red   = p[i][j].r ;
                colormap[n_colors].Green = p[i][j].g ;
                colormap[n_colors].Blue  = p[i][j].b ;
                n_colors++ ;
            }
        }
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: found %d color(s)\n", n_colors ) ;
#endif

    /* Round up to a power of 2 if needed */
    if ( n_colors != (1 << GifBitSize(n_colors)) )
    {
        n_colors = (1 << GifBitSize(n_colors)) ;
    }

#if SOBELF_DEBUG
    printf( "OUTPUT: Rounding up to %d color(s)\n", n_colors ) ;
#endif

    /* Update the global color map in the GIF */
    {
        ColorMapObject * cmo = GifMakeMapObject( n_colors, colormap ) ;
        if ( cmo == NULL )
        {
            fprintf(stderr, "Error while creating a ColorMapObject w/ %d color(s)\n",
                    n_colors ) ;
            return 0 ;
        }
        image->g->SColorMap = cmo ;
    }

    /* Now update the RasterBits in each frame */
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
                    break ;
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

    /* Finally, write the modified GIF out */
    if ( !output_modified_read_gif( filename, image->g ) ) 
        return 0 ;

    return 1 ;
}

/***************************************************************************
 * Gray filter
 ***************************************************************************/
void
apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p = image->p ;

    /* For each image */
    /* Parallelize across images or across pixels (or both). 
       Here we do per-image parallel, then inside a parallel-for for pixels. */
#pragma omp parallel for private(j)
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        int size = image->width[i] * image->height[i];
        /* Convert each pixel to grayscale */
#pragma omp parallel for
        for ( j = 0 ; j < size ; j++ )
        {
            int moy = (p[i][j].r + p[i][j].g + p[i][j].b)/3 ;
            if ( moy < 0 )   moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[i][j].r = moy ;
            p[i][j].g = moy ;
            p[i][j].b = moy ;
        }
    }
}

/***************************************************************************
 * Macros for indexing
 ***************************************************************************/
#define CONV(l,c,nb_c) ((l)*(nb_c)+(c))

/***************************************************************************
 * Blur filter with threshold for convergence
 ***************************************************************************/
void
apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i ;

    pixel ** p = image->p ;

    /* For each image, run the blur process (possibly multiple iterations) */
#pragma omp parallel for private(i)
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        int width  = image->width[i] ;
        int height = image->height[i] ;
        pixel * new = (pixel *)malloc(width * height * sizeof(pixel)) ;
        if (!new) {
            fprintf(stderr, "Error: can't allocate blur buffer\n");
            continue;
        }

        int end    = 0 ;
        int n_iter = 0 ;

        /* If threshold=0, we do at least 1 iteration. Otherwise, we keep iterating until stable. */
        do
        {
            end = 1 ; /* Assume we won't need another iteration */

            n_iter++ ;

            /* 1) Copy everything initially so that new[] matches p[].
               This includes borders or any area we won't specifically blur below. */
#pragma omp parallel for
            for(int j = 0; j < height; j++)
            {
                for(int k = 0; k < width; k++)
                {
                    new[CONV(j,k,width)] = p[i][CONV(j,k,width)];
                }
            }

            /* 2) Apply blur to top 10% of image (excluding a border of 'size') */
#pragma omp parallel for
            for(int j = size; j < height/10 - size; j++)
            {
                for(int k = size; k < width - size; k++)
                {
                    int t_r = 0, t_g = 0, t_b = 0 ;
                    for ( int sj = -size ; sj <= size ; sj++ )
                    {
                        for ( int sk = -size ; sk <= size ; sk++ )
                        {
                            t_r += p[i][CONV(j+sj,k+sk,width)].r ;
                            t_g += p[i][CONV(j+sj,k+sk,width)].g ;
                            t_b += p[i][CONV(j+sj,k+sk,width)].b ;
                        }
                    }
                    int area = (2*size+1)*(2*size+1);
                    new[CONV(j,k,width)].r = t_r / area ;
                    new[CONV(j,k,width)].g = t_g / area ;
                    new[CONV(j,k,width)].b = t_b / area ;
                }
            }

            /* 3) Middle part remains unblurred (copy from p[], which we did above). 
                  So no action needed except the copying we already did. */

            /* 4) Apply blur to bottom 10% of image (excluding border) */
#pragma omp parallel for
            for(int j = height*0.9 + size; j < height - size; j++)
            {
                for(int k = size; k < width - size; k++)
                {
                    int t_r = 0, t_g = 0, t_b = 0 ;
                    for ( int sj = -size ; sj <= size ; sj++ )
                    {
                        for ( int sk = -size ; sk <= size ; sk++ )
                        {
                            t_r += p[i][CONV(j+sj,k+sk,width)].r ;
                            t_g += p[i][CONV(j+sj,k+sk,width)].g ;
                            t_b += p[i][CONV(j+sj,k+sk,width)].b ;
                        }
                    }
                    int area = (2*size+1)*(2*size+1);
                    new[CONV(j,k,width)].r = t_r / area ;
                    new[CONV(j,k,width)].g = t_g / area ;
                    new[CONV(j,k,width)].b = t_b / area ;
                }
            }

            /* 5) Compare new[] and p[] to see if we need another iteration (if difference > threshold). 
                  Also copy new[] back to p[]. */
            if ( threshold > 0 ) 
            {
                /* If threshold=0, we do exactly 1 pass. Otherwise, we check convergence. */
#pragma omp parallel for
                for(int j=1; j<height-1; j++)
                {
                    for(int k=1; k<width-1; k++)
                    {
                        float diff_r = (new[CONV(j,k,width)].r - p[i][CONV(j,k,width)].r) ;
                        float diff_g = (new[CONV(j,k,width)].g - p[i][CONV(j,k,width)].g) ;
                        float diff_b = (new[CONV(j,k,width)].b - p[i][CONV(j,k,width)].b) ;

                        if ( fabsf(diff_r) > threshold ||
                             fabsf(diff_g) > threshold ||
                             fabsf(diff_b) > threshold )
                        {
                            /* If any pixel differs too much, we need another iteration */
#pragma omp critical
                            {
                                end = 0 ;
                            }
                        }
                        /* Write back */
                        p[i][CONV(j,k,width)] = new[CONV(j,k,width)];
                    }
                }
            }
            else
            {
                /* threshold == 0 => only one iteration, just copy new[] into p[] fully */
#pragma omp parallel for
                for(int j=0; j<height; j++)
                {
                    for(int k=0; k<width; k++)
                    {
                        p[i][CONV(j,k,width)] = new[CONV(j,k,width)];
                    }
                }
            }

        } while ( threshold > 0 && !end ); /* Repeat if not converged */

#if SOBELF_DEBUG
        printf( "BLUR: number of iterations for image %d = %d\n", i, n_iter ) ;
#endif

        free(new) ;
    }
}

/***************************************************************************
 * Sobel filter
 ***************************************************************************/
void
apply_sobel_filter( animated_gif * image )
{
    int i ;

    pixel ** p = image->p ;

    /* For each image, compute sobel edges */
#pragma omp parallel for private(i)
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        int width  = image->width[i] ;
        int height = image->height[i] ;

        pixel * sobel = (pixel*)malloc(width * height * sizeof(pixel)) ;
        if (!sobel) {
            fprintf(stderr, "Error: can't allocate sobel buffer\n");
            continue;
        }

        /* For each pixel (excluding border), compute the sobel magnitude */
#pragma omp parallel for
        for(int j=1; j<height-1; j++)
        {
            for(int k=1; k<width-1; k++)
            {
                /* We only use the blue channel here (as in the original code)
                   because we've converted to grayscale (r=g=b). */
                int pixel_no = p[i][CONV(j-1,k-1,width)].b ;
                int pixel_n  = p[i][CONV(j-1,k  ,width)].b ;
                int pixel_ne = p[i][CONV(j-1,k+1,width)].b ;
                int pixel_so = p[i][CONV(j+1,k-1,width)].b ;
                int pixel_s  = p[i][CONV(j+1,k  ,width)].b ;
                int pixel_se = p[i][CONV(j+1,k+1,width)].b ;
                int pixel_o  = p[i][CONV(j  ,k-1,width)].b ;
                int pixel    = p[i][CONV(j  ,k  ,width)].b ;
                int pixel_e  = p[i][CONV(j  ,k+1,width)].b ;

                float deltaX = (-pixel_no + pixel_ne
                                -2*pixel_o  + 2*pixel_e
                                -pixel_so   + pixel_se) ;

                float deltaY = ( pixel_se + 2*pixel_s + pixel_so
                                -pixel_ne  - 2*pixel_n - pixel_no);

                float val = sqrtf(deltaX*deltaX + deltaY*deltaY) / 4.0f ;

                /* Threshold of 50 for edge detection => 0 or 255 */
                if ( val > 50.0f )
                {
                    sobel[CONV(j,k,width)].r = 255 ;
                    sobel[CONV(j,k,width)].g = 255 ;
                    sobel[CONV(j,k,width)].b = 255 ;
                }
                else
                {
                    sobel[CONV(j,k,width)].r = 0 ;
                    sobel[CONV(j,k,width)].g = 0 ;
                    sobel[CONV(j,k,width)].b = 0 ;
                }
            }
        }

        /* Write sobel result back into p[] (excluding border) */
#pragma omp parallel for
        for(int j=1; j<height-1; j++)
        {
            for(int k=1; k<width-1; k++)
            {
                p[i][CONV(j,k,width)] = sobel[CONV(j,k,width)];
            }
        }

        free(sobel);
    }
}

/***************************************************************************
 * Main
 ***************************************************************************/
int 
main( int argc, char ** argv )
{
    char * input_filename ; 
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    double duration ;

    /* Check command-line arguments */
    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif\n", argv[0] ) ;
        return 1 ;
    }

    input_filename  = argv[1] ;
    output_filename = argv[2] ;

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array (SEQUENTIAL) */
    image = load_pixels( input_filename ) ;
    if ( image == NULL ) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec)+((t2.tv_usec - t1.tv_usec)/1e6);
    printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration ) ;

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale (PARALLEL) */
    apply_gray_filter( image ) ;

    /* Apply blur filter with size=5, threshold=20 (PARALLEL) */
    apply_blur_filter( image, 5, 20 ) ;

    /* Apply sobel filter on pixels (PARALLEL) */
    apply_sobel_filter( image ) ;

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec)+((t2.tv_usec - t1.tv_usec)/1e6);
    printf( "SOBEL done in %lf s\n", duration ) ;

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file (SEQUENTIAL) */
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec)+((t2.tv_usec - t1.tv_usec)/1e6);
    printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;

    return 0 ;
}
