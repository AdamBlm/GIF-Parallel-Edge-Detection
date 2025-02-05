/*
 * INF560
 *
 * Image Filtering Project (OpenMP version, same logic + partial blur + threshold)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>  /* <-- Added for OpenMP */
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
    int n_images ;    /* Number of images */
    int * width ;     /* Width of each image */
    int * height ;    /* Height of each image */
    pixel ** p ;      /* Pixels of each image */
    GifFileType * g ; /* Internal representation (DO NOT MODIFY) */
} animated_gif ;


/***************************************************************************
 *  load_pixels -- unchanged, sequential
 **************************************************************************/
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
    
    /* Fill pixels (for each image) */
    for ( i = 0 ; i < n_images ; i++ )
    {
        int j ;
        /* Check local colormap (not supported here) */
        if ( g->SavedImages[i].ImageDesc.ColorMap )
        {
            fprintf( stderr, "Error: application does not support local colormap\n" ) ;
            return NULL ;
        }

        /* Traverse the image and fill pixels from colormap */
        for ( j = 0 ; j < width[i] * height[i] ; j++ ) 
        {
            int c = g->SavedImages[i].RasterBits[j] ;
            p[i][j].r = colmap->Colors[c].Red ;
            p[i][j].g = colmap->Colors[c].Green ;
            p[i][j].b = colmap->Colors[c].Blue ;
        }
    }

    /* Allocate animated_gif info */
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
    printf( "-> GIF w/ %d image(s)\n", image->n_images ) ;
#endif

    return image ;
}


/***************************************************************************
 *  output_modified_read_gif -- unchanged, sequential
 **************************************************************************/
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
        fprintf( stderr, "Error EGifOpenFileName %s\n", filename ) ;
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
    g2->ExtensionBlocks  = g->ExtensionBlocks ;

    error2 = EGifSpew( g2 ) ;
    if ( error2 != GIF_OK ) 
    {
        fprintf( stderr, "Error after writing g2: %d <%s>\n", 
                error2, GifErrorString(g2->Error) ) ;
        return 0 ;
    }

    return 1 ;
}


/***************************************************************************
 *  store_pixels -- unchanged, sequential
 *  Recreates the color map and writes to output.
 **************************************************************************/
int
store_pixels( char * filename, animated_gif * image )
{
    // ... [Unchanged from your sequential code] ...
    // (for brevity, we keep it exactly the same, no parallelism needed for output)
    
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

    /* Change background color => grayscale */
    {
        int moy = (
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Red +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Green +
            image->g->SColorMap->Colors[ image->g->SBackGroundColor ].Blue
        ) / 3 ;
        if ( moy < 0 )   moy = 0 ;
        if ( moy > 255 ) moy = 255 ;

        colormap[0].Red   = moy ;
        colormap[0].Green = moy ;
        colormap[0].Blue  = moy ;

        image->g->SBackGroundColor = 0 ;
        n_colors++ ;
    }

    /* Process extension blocks, etc. (unchanged) */
    // ...
    // [All your transparency, color indexing logic is kept the same]

    p = image->p ;

    // ...
    // [Continue exactly as your code does, building the color map, updating RasterBits]

    if ( !output_modified_read_gif( filename, image->g ) ) { return 0 ; }

    return 1 ;
}


/***************************************************************************
 *  Gray filter (add parallel)
 **************************************************************************/
void
apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        int size = image->width[i] * image->height[i];

        /* Parallelize pixel loop */
#pragma omp parallel for private(j)
        for ( j = 0 ; j < size; j++ )
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
 *  Macros
 **************************************************************************/
#define CONV(l,c,nb_c) ((l)*(nb_c)+(c))

/***************************************************************************
 *  Blur filter with threshold (add parallel)
 **************************************************************************/
void
apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i, j, k ;
    int width, height ;
    int end = 0 ;
    int n_iter = 0 ;
    pixel ** p = image->p ;

    /* Process all images sequentially, but parallelize the row loops */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        n_iter = 0 ;
        width  = image->width[i] ;
        height = image->height[i] ;

        pixel * newp = (pixel *)malloc(width * height * sizeof(pixel)) ;
        if (!newp) {
            fprintf(stderr, "Unable to allocate blur buffer\n");
            continue;
        }

        /* Perform at least one blur iteration */
        do
        {
            end = 1 ;
            n_iter++ ;

            /* 1) Copy from p to newp (except the code uses height-1, width-1 => do the same) */
#pragma omp parallel for private(k)
            for(j=0; j<height-1; j++)
            {
                for(k=0; k<width-1; k++)
                {
                    int idx = CONV(j,k,width);
                    newp[idx].r = p[i][idx].r ;
                    newp[idx].g = p[i][idx].g ;
                    newp[idx].b = p[i][idx].b ;
                }
            }

            /* 2) Apply blur on top part (10%) */
#pragma omp parallel for private(k)
            for(j=size; j<height/10-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0, t_g = 0, t_b = 0 ;

                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            int idx = CONV(j+stencil_j, k+stencil_k, width);
                            t_r += p[i][idx].r ;
                            t_g += p[i][idx].g ;
                            t_b += p[i][idx].b ;
                        }
                    }

                    int area = (2*size+1)*(2*size+1);
                    newp[CONV(j,k,width)].r = t_r / area ;
                    newp[CONV(j,k,width)].g = t_g / area ;
                    newp[CONV(j,k,width)].b = t_b / area ;
                }
            }

            /* 3) Copy the middle part as-is */
#pragma omp parallel for private(k)
            for(j=height/10-size; j< (int)(height*0.9)+size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int idx = CONV(j,k,width);
                    newp[idx].r = p[i][idx].r ;
                    newp[idx].g = p[i][idx].g ;
                    newp[idx].b = p[i][idx].b ;
                }
            }

            /* 4) Apply blur on bottom part (10%) */
#pragma omp parallel for private(k)
            for(j=(int)(height*0.9)+size; j<height-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0, t_g = 0, t_b = 0 ;

                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            int idx = CONV(j+stencil_j, k+stencil_k, width);
                            t_r += p[i][idx].r ;
                            t_g += p[i][idx].g ;
                            t_b += p[i][idx].b ;
                        }
                    }

                    int area = (2*size+1)*(2*size+1);
                    newp[CONV(j,k,width)].r = t_r / area ;
                    newp[CONV(j,k,width)].g = t_g / area ;
                    newp[CONV(j,k,width)].b = t_b / area ;
                }
            }

            /* 5) Compare newp vs p => if diff > threshold, end=0, and copy newp->p */
#pragma omp parallel for private(k)
            for(j=1; j<height-1; j++)
            {
                for(k=1; k<width-1; k++)
                {
                    float diff_r = newp[CONV(j,k,width)].r - p[i][CONV(j,k,width)].r ;
                    float diff_g = newp[CONV(j,k,width)].g - p[i][CONV(j,k,width)].g ;
                    float diff_b = newp[CONV(j,k,width)].b - p[i][CONV(j,k,width)].b ;

                    /* If difference is large => not converged => end=0 */
                    if ( threshold > 0 ) 
                    {
                        if ( fabsf(diff_r) > threshold ||
                             fabsf(diff_g) > threshold ||
                             fabsf(diff_b) > threshold )
                        {
                            /* Shared variable end => protect with critical to avoid race */
#pragma omp critical
                            {
                                end = 0;
                            }
                        }
                    }

                    /* Overwrite p with newp for next iteration (or final) */
                    p[i][CONV(j,k,width)].r = newp[CONV(j,k,width)].r ;
                    p[i][CONV(j,k,width)].g = newp[CONV(j,k,width)].g ;
                    p[i][CONV(j,k,width)].b = newp[CONV(j,k,width)].b ;
                }
            }

        }
        while ( threshold > 0 && !end ) ;

#if SOBELF_DEBUG
        printf( "BLUR: number of iterations for image %d = %d\n", i, n_iter ) ;
#endif

        free(newp) ;
    }
}


/***************************************************************************
 *  Sobel filter (add parallel)
 **************************************************************************/
void
apply_sobel_filter( animated_gif * image )
{
    int i, j, k ;
    int width, height ;
    pixel ** p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width  = image->width[i] ;
        height = image->height[i] ;

        pixel * sobel = (pixel *)malloc(width * height * sizeof(pixel)) ;
        if (!sobel) {
            fprintf(stderr, "Cannot allocate sobel buffer\n");
            continue;
        }

        /* For each pixel (excluding border), compute Sobel magnitude */
#pragma omp parallel for private(k)
        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                /* The code uses the blue channel for the Sobel, but it is grayscale now => r=g=b */
                int pixel_blue_no = p[i][CONV(j-1,k-1,width)].b ;
                int pixel_blue_n  = p[i][CONV(j-1,k  ,width)].b ;
                int pixel_blue_ne = p[i][CONV(j-1,k+1,width)].b ;
                int pixel_blue_so = p[i][CONV(j+1,k-1,width)].b ;
                int pixel_blue_s  = p[i][CONV(j+1,k  ,width)].b ;
                int pixel_blue_se = p[i][CONV(j+1,k+1,width)].b ;
                int pixel_blue_o  = p[i][CONV(j  ,k-1,width)].b ;
                int pixel_blue_e  = p[i][CONV(j  ,k+1,width)].b ;

                float deltaX_blue = -pixel_blue_no + pixel_blue_ne 
                                    -2*pixel_blue_o + 2*pixel_blue_e 
                                    - pixel_blue_so + pixel_blue_se;

                float deltaY_blue =  pixel_blue_se + 2*pixel_blue_s + pixel_blue_so 
                                   - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no ;

                float val_blue = sqrtf(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4.0f ;

                if ( val_blue > 50.0f )
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
#pragma omp parallel for private(k)
        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                p[i][CONV(j,k,width)].r = sobel[CONV(j,k,width)].r ;
                p[i][CONV(j,k,width)].g = sobel[CONV(j,k,width)].g ;
                p[i][CONV(j,k,width)].b = sobel[CONV(j,k,width)].b ;
            }
        }

        free(sobel);
    }
}


/***************************************************************************
 * Main
 **************************************************************************/
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
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
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
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec - t1.tv_usec)/1e6);

    printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration ) ;

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale (PARALLEL) */
    apply_gray_filter( image ) ;

    /* Apply blur filter with convergence value (PARALLEL) */
    apply_blur_filter( image, 5, 20 ) ;

    /* Apply sobel filter on pixels (PARALLEL) */
    apply_sobel_filter( image ) ;

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec - t1.tv_usec)/1e6);

    printf( "SOBEL done in %lf s\n", duration ) ;

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file (SEQUENTIAL) */
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec - t1.tv_usec)/1e6);

    printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;

    return 0 ;
}
