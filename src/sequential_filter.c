#include "sequential_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "gif_lib.h"
#include "gif_io.h"


#define SOBELF_DEBUG 0


#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)


static void apply_gray_filter(animated_gif * image) {
    int i, j, k;
    pixel ** p = image->p;
    for (i = 0; i < image->n_images; i++) {
        int width = image->width[i], height = image->height[i];
        for (j = 0; j < height; j++) {
            for (k = 0; k < width; k++) {
                int idx = j * width + k;
                int moy = (p[i][idx].r + p[i][idx].g + p[i][idx].b) / 3;
                if (moy < 0) moy = 0;
                if (moy > 255) moy = 255;
                p[i][idx].r = moy;
                p[i][idx].g = moy;
                p[i][idx].b = moy;
            }
        }
    }
}


static void apply_blur_filter(animated_gif * image, int size, int threshold)
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

static void apply_sobel_filter(animated_gif * image)
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
                if (max_val > 15)  // Adjusted threshold to get visible edges
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

void apply_gray_line(animated_gif * image) 
{
    int i, j, k;
    pixel ** p;

    p = image->p;

    for (i = 0; i < image->n_images; i++)
    {
        for (j = 0; j < 10; j++)
        {
            for (k = image->width[i]/2; k < image->width[i]; k++)
            {
                p[i][CONV(j,k,image->width[i])].r = 0;
                p[i][CONV(j,k,image->width[i])].g = 0;
                p[i][CONV(j,k,image->width[i])].b = 0;
            }
        }
    }
}


int run_sequential_filter(char *input_filename, char *output_filename) {
    animated_gif * image;
    struct timeval t1, t2;
    double duration;

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    image = load_pixels( input_filename ) ;
    if ( image == NULL ) { return 1 ; }

    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
            input_filename, image->n_images, duration ) ;

    /* FILTER Timer start */
    gettimeofday(&t1, NULL);

    /* Convert the pixels into grayscale */
    apply_gray_filter( image ) ;

    /* Apply blur filter with convergence value */
    apply_blur_filter( image, 3, 0 ) ;
    /* Apply sobel filter on pixels */
    apply_sobel_filter( image ) ;

    /* FILTER Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "SOBEL done in %lf s\n", duration ) ;

    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Store file from array of pixels to GIF file */
    if ( !store_pixels( output_filename, image ) ) { return 1 ; }

    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);

    duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

    printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;

    return 0 ;
}
