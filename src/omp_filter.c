#include "omp_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include "gif_lib.h"
#include "gif_io.h"


#ifndef SOBELF_DEBUG
#define SOBELF_DEBUG 0
#endif

/* ------------------------------------------------------------------
 * Helper Macros
 * ------------------------------------------------------------------ */
#define CONV(l,c,nb_c) ((l)*(nb_c)+(c))
#define TILE_SIZE 64

static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

/* ------------------------------------------------------------------
 * Filtering Functions (OpenMP Optimized)
 * ------------------------------------------------------------------ */

/* Apply grayscale filter to the image */
static void apply_gray_filter(animated_gif * image)
{
    pixel ** p = image->p;
    #pragma omp parallel for default(none) shared(p, image) schedule(dynamic)
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        #pragma omp parallel for default(none) shared(p, i, width, height) schedule(static)
        for (int j = 0; j < height; j++) {
            #pragma omp simd
            for (int k = 0; k < width; k++) {
                int index = CONV(j, k, width);
                int moy = (p[i][index].r + p[i][index].g + p[i][index].b) / 3;
                if (moy < 0) moy = 0;
                if (moy > 255) moy = 255;
                p[i][index].r = moy;
                p[i][index].g = moy;
                p[i][index].b = moy;
            }
        }
    }
}

/* Apply blur filter with tiling optimization */
static void apply_blur_filter(animated_gif * image, int size, int threshold)
{
    pixel ** p = image->p;
    pixel ** buffer = (pixel **)malloc(image->n_images * sizeof(pixel *));
    if (buffer == NULL) {
        fprintf(stderr, "Unable to allocate memory for blur filter buffers\n");
        return;
    }
    
    for (int i = 0; i < image->n_images; i++) {
        buffer[i] = (pixel *)malloc(image->width[i] * image->height[i] * sizeof(pixel));
        if (buffer[i] == NULL) {
            fprintf(stderr, "Unable to allocate memory for blur filter buffer %d\n", i);
            for (int j = 0; j < i; j++) {
                free(buffer[j]);
            }
            free(buffer);
            return;
        }
    }

    #pragma omp parallel for default(none) shared(p, buffer, image, size, threshold) schedule(dynamic)
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        int end = 0, n_iter = 0;
        
        #pragma omp parallel for default(none) shared(p, buffer, i, width, height) schedule(static)
        for (int j = 0; j < height; j++) {
            #pragma omp simd
            for (int k = 0; k < width; k++) {
                int index = CONV(j, k, width);
                buffer[i][index].r = p[i][index].r;
                buffer[i][index].g = p[i][index].g;
                buffer[i][index].b = p[i][index].b;
            }
        }

        do {
            end = 1;
            n_iter++;
            #pragma omp parallel for default(none) shared(p, buffer, i, width, height, size) schedule(static)
            for (int jj = size; jj < height - size; jj += TILE_SIZE) {
                for (int kk = size; kk < width - size; kk += TILE_SIZE) {
                    for (int j = jj; j < min(jj + TILE_SIZE, height - size); j++) {
                        for (int k = kk; k < min(kk + TILE_SIZE, width - size); k++) {
                            int t_r = 0, t_g = 0, t_b = 0;
                            for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
                                for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
                                    int idx = CONV(j+stencil_j, k+stencil_k, width);
                                    t_r += p[i][idx].r;
                                    t_g += p[i][idx].g;
                                    t_b += p[i][idx].b;
                                }
                            }
                            const int denom = (2*size+1)*(2*size+1);
                            buffer[i][CONV(j, k, width)].r = t_r / denom;
                            buffer[i][CONV(j, k, width)].g = t_g / denom;
                            buffer[i][CONV(j, k, width)].b = t_b / denom;
                        }
                    }
                }
            }

            int continue_flag = 0;
            #pragma omp parallel for default(none) shared(p, buffer, i, width, height, size, threshold) reduction(|:continue_flag) schedule(static)
            for (int j = size; j < height - size; j++) {
                for (int k = size; k < width - size; k++) {
                    int index = CONV(j, k, width);
                    int diff_r = buffer[i][index].r - p[i][index].r;
                    int diff_g = buffer[i][index].g - p[i][index].g;
                    int diff_b = buffer[i][index].b - p[i][index].b;
                    if (abs(diff_r) > threshold || abs(diff_g) > threshold || abs(diff_b) > threshold) {
                        continue_flag = 1;
                    }
                    p[i][index].r = buffer[i][index].r;
                    p[i][index].g = buffer[i][index].g;
                    p[i][index].b = buffer[i][index].b;
                }
            }
            end = !continue_flag;
        } while (threshold > 0 && !end);

#if SOBELF_DEBUG
        printf("BLUR: number of iterations for image %d: %d\n", i, n_iter);
#endif
    }
    
    for (int i = 0; i < image->n_images; i++) {
        free(buffer[i]);
    }
    free(buffer);
}

/* Apply Sobel filter with optimized multi-channel implementation */
static void apply_sobel_filter(animated_gif * image)
{
    pixel ** p = image->p;
    pixel ** sobel_buffer = (pixel **)malloc(image->n_images * sizeof(pixel *));
    if (sobel_buffer == NULL) {
        fprintf(stderr, "Unable to allocate memory for sobel filter buffers\n");
        return;
    }
    
    for (int i = 0; i < image->n_images; i++) {
        sobel_buffer[i] = (pixel *)malloc(image->width[i] * image->height[i] * sizeof(pixel));
        if (sobel_buffer[i] == NULL) {
            fprintf(stderr, "Unable to allocate memory for sobel filter buffer %d\n", i);
            for (int j = 0; j < i; j++) {
                free(sobel_buffer[j]);
            }
            free(sobel_buffer);
            return;
        }
    }

    #pragma omp parallel for default(none) shared(p, sobel_buffer, image) schedule(dynamic)
    for (int i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        #pragma omp parallel for default(none) shared(p, sobel_buffer, i, width, height) schedule(static)
        for (int j = 0; j < height; j++) {
            #pragma omp simd
            for (int k = 0; k < width; k++) {
                int index = CONV(j, k, width);
                sobel_buffer[i][index].r = p[i][index].r;
                sobel_buffer[i][index].g = p[i][index].g;
                sobel_buffer[i][index].b = p[i][index].b;
            }
        }
        
        #pragma omp parallel for default(none) shared(p, sobel_buffer, i, width, height) schedule(static)
        for (int j = 1; j < height - 1; j++) {
            const int row_prev = CONV(j-1, 0, width);
            const int row_curr = CONV(j, 0, width);
            const int row_next = CONV(j+1, 0, width);
            #pragma omp simd
            for (int k = 1; k < width - 1; k++) {
                int pixel_red_no = p[i][row_prev + k-1].r;
                int pixel_red_n  = p[i][row_prev + k].r;
                int pixel_red_ne = p[i][row_prev + k+1].r;
                int pixel_red_o  = p[i][row_curr + k-1].r;
                int pixel_red_e  = p[i][row_curr + k+1].r;
                int pixel_red_so = p[i][row_next + k-1].r;
                int pixel_red_s  = p[i][row_next + k].r;
                int pixel_red_se = p[i][row_next + k+1].r;
                
                int pixel_green_no = p[i][row_prev + k-1].g;
                int pixel_green_n  = p[i][row_prev + k].g;
                int pixel_green_ne = p[i][row_prev + k+1].g;
                int pixel_green_o  = p[i][row_curr + k-1].g;
                int pixel_green_e  = p[i][row_curr + k+1].g;
                int pixel_green_so = p[i][row_next + k-1].g;
                int pixel_green_s  = p[i][row_next + k].g;
                int pixel_green_se = p[i][row_next + k+1].g;
                
                int pixel_blue_no = p[i][row_prev + k-1].b;
                int pixel_blue_n  = p[i][row_prev + k].b;
                int pixel_blue_ne = p[i][row_prev + k+1].b;
                int pixel_blue_o  = p[i][row_curr + k-1].b;
                int pixel_blue_e  = p[i][row_curr + k+1].b;
                int pixel_blue_so = p[i][row_next + k-1].b;
                int pixel_blue_s  = p[i][row_next + k].b;
                int pixel_blue_se = p[i][row_next + k+1].b;
                
                float deltaX_red = -pixel_red_no + pixel_red_ne - 2*pixel_red_o + 2*pixel_red_e - pixel_red_so + pixel_red_se;
                float deltaY_red = pixel_red_se + 2*pixel_red_s + pixel_red_so - pixel_red_ne - 2*pixel_red_n - pixel_red_no;
                float val_red = sqrt(deltaX_red * deltaX_red + deltaY_red * deltaY_red) / 4.0f;
                
                float deltaX_green = -pixel_green_no + pixel_green_ne - 2*pixel_green_o + 2*pixel_green_e - pixel_green_so + pixel_green_se;
                float deltaY_green = pixel_green_se + 2*pixel_green_s + pixel_green_so - pixel_green_ne - 2*pixel_green_n - pixel_green_no;
                float val_green = sqrt(deltaX_green * deltaX_green + deltaY_green * deltaY_green) / 4.0f;
                
                float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
                float deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
                float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4.0f;
                
                float max_val = val_red;
                if (val_green > max_val) max_val = val_green;
                if (val_blue > max_val) max_val = val_blue;
                
                if (max_val > 15)
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
        
        #pragma omp parallel for default(none) shared(p, sobel_buffer, i, width, height) schedule(static)
        for (int j = 0; j < height; j++) {
            #pragma omp simd
            for (int k = 0; k < width; k++) {
                int index = CONV(j, k, width);
                p[i][index].r = sobel_buffer[i][index].r;
                p[i][index].g = sobel_buffer[i][index].g;
                p[i][index].b = sobel_buffer[i][index].b;
            }
        }
    }
    
    for (int i = 0; i < image->n_images; i++) {
        free(sobel_buffer[i]);
    }
    free(sobel_buffer);
}

/* ------------------------------------------------------------------
 * run_omp_filter: The Adaptive OpenMP Filtering Pipeline
 * ------------------------------------------------------------------
 *
 * This function loads the input GIF, applies grayscale, blur, and Sobel filters
 * using optimized OpenMP routines, and then writes the output GIF.
 *
 * Returns 0 on success, nonzero on failure.
 */
int run_omp_filter(char *input_filename, char *output_filename, int num_threads)
{
    if(num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
    printf("Using %d OpenMP threads\n", omp_get_max_threads());

    struct timeval t1, t2;
    double duration;

    /* Load the GIF */
    gettimeofday(&t1, NULL);
    animated_gif *image = load_pixels(input_filename);
    if (image == NULL) {
        return 1;
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
    printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
           input_filename, image->n_images, duration);

    /* Filtering Stage */
    gettimeofday(&t1, NULL);
    apply_gray_filter(image);
    apply_blur_filter(image, 3, 0);
    apply_sobel_filter(image);
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
    printf("SOBEL done in %lf s\n", duration);

    /* Export Stage */
    gettimeofday(&t1, NULL);
    if (!store_pixels(output_filename, image)) {
        return 1;
    }
    gettimeofday(&t2, NULL);
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
    printf("Export done in %lf s in file %s\n", duration, output_filename);

    return 0;
}
