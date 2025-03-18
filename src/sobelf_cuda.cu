/*
 * INF560 - Image Filtering Project (CUDA version)
 *
 * This application loads a GIF image using our in-house gif_lib,
 * transfers each frame to the GPU, applies three filters (grayscale, blur, and Sobel)
 * using CUDA kernels with shared memory tiling, and then transfers the result back
 * to the host to write out the modified GIF.
 *
 *
 * Compile with (example):
 *   nvcc -o sobelf_cuda sobelf_cuda.cu dgif_lib.c egif_lib.c gif_err.c gif_font.c gif_hash.c gifalloc.c openbsd-reallocarray.c quantize.c -lm -Xcompiler -fopenmp
 *
 * Run with:
 *   ./sobelf_cuda input.gif output.gif
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h> 
#include "gif_lib.h"
#include <cuda_runtime.h>

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

// Kernel for converting an image to grayscale
__global__ void grayscaleKernel(pixel *d_pixels, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height){
        int idx = y * width + x;
        int gray = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        d_pixels[idx].r = gray;
        d_pixels[idx].g = gray;
        d_pixels[idx].b = gray;
    }
}
__global__ void blurKernel(pixel *d_in, pixel *d_out, int width, int height)
{
    extern __shared__ pixel s_data[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int x = bx + tx, y = by + ty;
    int tileWidth = blockDim.x + 2;
    int sm_x = tx + 1, sm_y = ty + 1;

    // Load center pixel if in range.
    if(x < width && y < height)
        s_data[sm_y * tileWidth + sm_x] = d_in[y * width + x];

    // Load halo pixels
    if(tx == 0 && x > 0)
        s_data[sm_y * tileWidth + 0] = d_in[y * width + (x - 1)];
    if(tx == blockDim.x - 1 && x < width - 1)
        s_data[sm_y * tileWidth + sm_x + 1] = d_in[y * width + (x + 1)];
    if(ty == 0 && y > 0)
        s_data[0 * tileWidth + sm_x] = d_in[(y - 1) * width + x];
    if(ty == blockDim.y - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x] = d_in[(y + 1) * width + x];
    __syncthreads();

    // Only process interior pixels that have a full  cross-neighborhood.
    if(x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        // Compute a weighted sum using a cross pattern:
        // weight center*4, each direct neighbor weight 1
        int center = s_data[sm_y * tileWidth + sm_x].r; // assume grayscale so any channel works
        int left   = s_data[sm_y * tileWidth + (sm_x - 1)].r;
        int right  = s_data[sm_y * tileWidth + (sm_x + 1)].r;
        int top    = s_data[(sm_y - 1) * tileWidth + sm_x].r;
        int bottom = s_data[(sm_y + 1) * tileWidth + sm_x].r;
        int newVal = (4 * center + left + right + top + bottom) / 8;

        int idx = y * width + x;
        // Apply new value to all channels.
        d_out[idx].r = newVal;
        d_out[idx].g = newVal;
        d_out[idx].b = newVal;
    }
}

// Sobel kernel using shared memory tiling (operating on the blue channel)
__global__ void sobelKernel(pixel *d_in, pixel *d_out, int width, int height)
{
    extern __shared__ pixel s_data[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int x = bx + tx, y = by + ty;
    int tileWidth = blockDim.x + 2;
    int sm_x = tx + 1, sm_y = ty + 1;

    if(x < width && y < height)
        s_data[sm_y * tileWidth + sm_x] = d_in[y * width + x];
    if(tx == 0 && x > 0)
        s_data[sm_y * tileWidth + 0] = d_in[y * width + (x - 1)];
    if(tx == blockDim.x - 1 && x < width - 1)
        s_data[sm_y * tileWidth + sm_x + 1] = d_in[y * width + (x + 1)];
    if(ty == 0 && y > 0)
        s_data[0 * tileWidth + sm_x] = d_in[(y - 1) * width + x];
    if(ty == blockDim.y - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x] = d_in[(y + 1) * width + x];
    if(tx == 0 && ty == 0 && x > 0 && y > 0)
        s_data[0] = d_in[(y - 1) * width + (x - 1)];
    if(tx == blockDim.x - 1 && ty == 0 && x < width - 1 && y > 0)
        s_data[0 * tileWidth + sm_x + 1] = d_in[(y - 1) * width + (x + 1)];
    if(tx == 0 && ty == blockDim.y - 1 && x > 0 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + 0] = d_in[(y + 1) * width + (x - 1)];
    if(tx == blockDim.x - 1 && ty == blockDim.y - 1 && x < width - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x + 1] = d_in[(y + 1) * width + (x + 1)];
    __syncthreads();

    if(x > 0 && x < width - 1 && y > 0 && y < height - 1)
    {
        int Gx = - s_data[(sm_y - 1) * tileWidth + (sm_x - 1)].b
                 - 2 * s_data[sm_y * tileWidth + (sm_x - 1)].b
                 - s_data[(sm_y + 1) * tileWidth + (sm_x - 1)].b
                 + s_data[(sm_y - 1) * tileWidth + (sm_x + 1)].b
                 + 2 * s_data[sm_y * tileWidth + (sm_x + 1)].b
                 + s_data[(sm_y + 1) * tileWidth + (sm_x + 1)].b;
        int Gy = s_data[(sm_y - 1) * tileWidth + (sm_x + 1)].b
                 + 2 * s_data[(sm_y - 1) * tileWidth + sm_x].b
                 + s_data[(sm_y - 1) * tileWidth + (sm_x - 1)].b
                 - s_data[(sm_y + 1) * tileWidth + (sm_x - 1)].b
                 - 2 * s_data[(sm_y + 1) * tileWidth + sm_x].b
                 - s_data[(sm_y + 1) * tileWidth + (sm_x + 1)].b;
        int magnitude = (int)(sqrtf((float)(Gx * Gx + Gy * Gy)) / 4.0f);
        int idx = y * width + x;
        if(magnitude > 35) {
            d_out[idx].r = 255;
            d_out[idx].g = 255;
            d_out[idx].b = 255;
        } else {
            d_out[idx].r = 0;
            d_out[idx].g = 0;
            d_out[idx].b = 0;
        }
    }
}


/* --------------------------------------------------------------------
   Tiling parameters.
   -------------------------------------------------------------------- */
#define TILE_WIDTH 1024
#define TILE_HEIGHT 1024

/* --------------------------------------------------------------------
   process_tile(): Process one tile of an image on the GPU.
   -------------------------------------------------------------------- */
void process_tile(pixel *tile_in, pixel *tile_out, int tile_w, int tile_h) {
    size_t tileSizeBytes = tile_w * tile_h * sizeof(pixel);
    pixel *d_in, *d_out;
    cudaError_t err;

    err = cudaMalloc((void**)&d_in, tileSizeBytes);
    if(err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_in for tile: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMalloc((void**)&d_out, tileSizeBytes);
    if(err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc d_out for tile: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    err = cudaMemcpy(d_in, tile_in, tileSizeBytes, cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        fprintf(stderr, "ERROR: cudaMemcpy to d_in for tile: %s\n", cudaGetErrorString(err));

    dim3 blockDim(16,16);
    dim3 gridDim((tile_w + blockDim.x - 1) / blockDim.x, (tile_h + blockDim.y - 1) / blockDim.y);
    size_t sharedMemSize = (blockDim.x + 2) * (blockDim.y + 2) * sizeof(pixel);

    // Grayscale kernel
    grayscaleKernel<<<gridDim, blockDim>>>(d_in, tile_w, tile_h);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)
        fprintf(stderr, "ERROR: After grayscaleKernel: %s\n", cudaGetErrorString(err));

    // Blur kernel
    blurKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, tile_w, tile_h);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if(err != cudaSuccess)
        fprintf(stderr, "ERROR: After blurKernel: %s\n", cudaGetErrorString(err));

    // Copy blurred result back into d_in.
    cudaMemcpy(d_in, d_out, tileSizeBytes, cudaMemcpyDeviceToDevice);

    // Sobel kernel with timing debug.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    sobelKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, tile_w, tile_h);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
   
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(tile_out, d_out, tileSizeBytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);

#if SOBELF_DEBUG
    // Print a few pixel values from the tile_out (top-left corner)
    printf("DEBUG (tile): First pixel after processing: (%d, %d, %d)\n",
           tile_out[0].r, tile_out[0].g, tile_out[0].b);
#endif
}
int main( int argc, char ** argv )
{
    char * input_filename;
    char * output_filename;
    animated_gif * image;
    struct timeval t1, t2;
    double duration;
    
    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif\n", argv[0] );
        return 1;
    }
    
    input_filename = argv[1];
    output_filename = argv[2];
    
    /* ---------------------- Load Time ---------------------- */
    gettimeofday(&t1, NULL);
    image = load_pixels( input_filename );
    if ( image == NULL ) { return 1; }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
    // Print load time: field 10 will be the duration (in seconds)
    printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
           input_filename, image->n_images, duration);
    
    /* ---------------------- Filter Time ---------------------- */
    gettimeofday(&t1, NULL);
    
    /* --- Apply filters on GPU --- */
    // Grayscale, blur and Sobel are applied by process_tile() in your tiling loop.
    // (The code below processes each image frame in tiles.)
    for (int img = 0; img < image->n_images; img++) {
        int w = image->width[img];
        int h = image->height[img];
        printf("DEBUG: Processing image %d, dimensions: %d x %d\n", img, w, h);
        
        // Allocate a temporary buffer for the processed image.
        pixel *temp = (pixel *)malloc(w * h * sizeof(pixel));
        if(temp == NULL) {
            fprintf(stderr, "ERROR: Failed to allocate host memory for image %d\n", img);
            exit(1);
        }
        
        // Process image in tiles.
        for (int y = 0; y < h; y += TILE_HEIGHT) {
            for (int x = 0; x < w; x += TILE_WIDTH) {
                int currentTileWidth = (x + TILE_WIDTH > w) ? (w - x) : TILE_WIDTH;
                int currentTileHeight = (y + TILE_HEIGHT > h) ? (h - y) : TILE_HEIGHT;
                size_t tileSize = currentTileWidth * currentTileHeight * sizeof(pixel);
                
                pixel *tile_in = (pixel *)malloc(tileSize);
                pixel *tile_out = (pixel *)malloc(tileSize);
                if(!tile_in || !tile_out) {
                    fprintf(stderr, "ERROR: Unable to allocate tile buffers.\n");
                    exit(1);
                }
                
                // Copy tile from the full image.
                for (int j = 0; j < currentTileHeight; j++) {
                    memcpy(&tile_in[j * currentTileWidth],
                           &image->p[img][(y + j) * w + x],
                           currentTileWidth * sizeof(pixel));
                }
                
                process_tile(tile_in, tile_out, currentTileWidth, currentTileHeight);
                
                // Copy processed tile back into the temporary full-image buffer.
                for (int j = 0; j < currentTileHeight; j++) {
                    memcpy(&temp[(y + j) * w + x],
                           &tile_out[j * currentTileWidth],
                           currentTileWidth * sizeof(pixel));
                }
                
                free(tile_in);
                free(tile_out);
            }
        }
        
        // Replace original frame with processed image.
        memcpy(image->p[img], temp, w * h * sizeof(pixel));
        free(temp);
    }
    
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
    // Print filter time: field 5 of this message will be the duration.
    printf("GPU filters done in %lf s\n", duration);
    
    /* ---------------------- Export Time ---------------------- */
    gettimeofday(&t1, NULL);
    if(!store_pixels(output_filename, image)) {
        fprintf(stderr, "Error storing GIF\n");
        return 1;
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec-t1.tv_usec)/1e6);
    // Print export time: field 4 of this message will be the duration.
    printf("Export done in %lf s in file %s\n", duration, output_filename);
    
    return 0;
}