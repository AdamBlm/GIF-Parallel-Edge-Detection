/*
 * MPI Implementation for distributed image processing
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

/* MPI tags for communication */
#define MPI_TAG_FRAME_COUNT 100
#define MPI_TAG_FRAME_SIZE  101
#define MPI_TAG_FRAME_DATA  102
#define MPI_TAG_RESULT_DATA 103

/* MPI implementation for main */
int main(int argc, char **argv) {
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;
    
    int rank, size, i, j;
    int frames_per_process;
    int *sendcounts;
    int *displs;
    MPI_Status status;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Check command-line arguments */
    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    input_filename = argv[1];
    output_filename = argv[2];

    /* MASTER PROCESS: Load file and distribute work */
    if (rank == 0) {
        /* IMPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Load file and store the pixels in array */
        image = load_pixels(input_filename);
        if (image == NULL) {
            MPI_Finalize();
            return 1;
        }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n",
                input_filename, image->n_images, duration);

        /* FILTER Timer start */
        gettimeofday(&t1, NULL);

        /* Broadcast the number of images */
        MPI_Bcast(&(image->n_images), 1, MPI_INT, 0, MPI_COMM_WORLD);

        /* Calculate frames per process and setup distribution arrays */
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        frames_per_process = image->n_images / size;
        int remainder = image->n_images % size;
        
        displs[0] = 0;
        for (i = 0; i < size; i++) {
            sendcounts[i] = frames_per_process + (i < remainder ? 1 : 0);
            if (i > 0) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
        }
        
        /* Distribute image dimensions to all processes */
        MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
        
        /* Send each process its assigned frames */
        for (i = 1; i < size; i++) {
            for (j = 0; j < sendcounts[i]; j++) {
                int frame_idx = displs[i] + j;
                int frame_size = image->width[frame_idx] * image->height[frame_idx];
                
                /* Send frame size and pixel data */
                MPI_Send(&frame_size, 1, MPI_INT, i, MPI_TAG_FRAME_SIZE, MPI_COMM_WORLD);
                MPI_Send(image->p[frame_idx], frame_size * sizeof(pixel), MPI_BYTE, 
                         i, MPI_TAG_FRAME_DATA, MPI_COMM_WORLD);
            }
        }
        
        /* Process frames assigned to rank 0 */
        for (j = 0; j < sendcounts[0]; j++) {
            int frame_idx = displs[0] + j;
            
            /* Apply filters to this frame */
            apply_gray_filter_frame(image, frame_idx);
            apply_blur_filter_frame(image, frame_idx, 5, 20);
            apply_sobel_filter_frame(image, frame_idx);
        }
        
        /* Receive processed frames from workers */
        for (i = 1; i < size; i++) {
            for (j = 0; j < sendcounts[i]; j++) {
                int frame_idx = displs[i] + j;
                int frame_size = image->width[frame_idx] * image->height[frame_idx];
                
                /* Receive processed frame */
                MPI_Recv(image->p[frame_idx], frame_size * sizeof(pixel), MPI_BYTE,
                         i, MPI_TAG_RESULT_DATA, MPI_COMM_WORLD, &status);
            }
        }
        
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("SOBEL done in %lf s\n", duration);
        
        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);
        
        /* Store file from array of pixels to GIF file */
        if (!store_pixels(output_filename, image)) {
            free(sendcounts);
            free(displs);
            MPI_Finalize();
            return 1;
        }
        
        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("Export done in %lf s in file %s\n", duration, output_filename);
        
        /* Free allocated memory */
        free(sendcounts);
        free(displs);
    } 
    /* WORKER PROCESSES: Receive frames, apply filters, send back results */
    else {
        int n_images;
        /* Receive broadcast of number of images */
        MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        /* Allocate and receive image dimensions */
        int *width = (int *)malloc(n_images * sizeof(int));
        int *height = (int *)malloc(n_images * sizeof(int));
        
        MPI_Bcast(width, n_images, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(height, n_images, MPI_INT, 0, MPI_COMM_WORLD);
        
        /* Calculate frames to process based on rank */
        frames_per_process = n_images / size;
        int remainder = n_images % size;
        int frames_to_process = frames_per_process + (rank < remainder ? 1 : 0);
        
        /* Process assigned frames */
        for (i = 0; i < frames_to_process; i++) {
            int frame_size;
            pixel *frame_pixels;
            
            /* Receive frame size and allocate memory */
            MPI_Recv(&frame_size, 1, MPI_INT, 0, MPI_TAG_FRAME_SIZE, MPI_COMM_WORLD, &status);
            frame_pixels = (pixel *)malloc(frame_size * sizeof(pixel));
            
            /* Receive frame data */
            MPI_Recv(frame_pixels, frame_size * sizeof(pixel), MPI_BYTE, 
                     0, MPI_TAG_FRAME_DATA, MPI_COMM_WORLD, &status);
            
            /* Create a temporary animated_gif for this frame */
            animated_gif temp_image;
            temp_image.n_images = 1;
            temp_image.width = (int *)malloc(sizeof(int));
            temp_image.height = (int *)malloc(sizeof(int));
            temp_image.p = (pixel **)malloc(sizeof(pixel *));
            
            /* Set the dimensions and pixel data */
            int frame_idx = rank + i * size; /* Compute actual frame index */
            if (frame_idx >= n_images) {
                frame_idx = rank;  /* Fallback for edge cases */
            }
            
            temp_image.width[0] = width[frame_idx];
            temp_image.height[0] = height[frame_idx];
            temp_image.p[0] = frame_pixels;
            
            /* Apply filters */
            apply_gray_filter(&temp_image);
            apply_blur_filter(&temp_image, 5, 20);
            apply_sobel_filter(&temp_image);
            
            /* Send processed frame back to master */
            MPI_Send(frame_pixels, frame_size * sizeof(pixel), MPI_BYTE,
                     0, MPI_TAG_RESULT_DATA, MPI_COMM_WORLD);
            
            /* Free temporary memory */
            free(temp_image.width);
            free(temp_image.height);
            free(temp_image.p);
            free(frame_pixels);
        }
        
        /* Free allocated memory */
        free(width);
        free(height);
    }
    
    MPI_Finalize();
    return 0;
}

/* Frame-specific filter functions for the master process */
void apply_gray_filter_frame(animated_gif *image, int frame_idx) {
    int j;
    pixel *p = image->p[frame_idx];
    int frame_size = image->width[frame_idx] * image->height[frame_idx];
    
    #pragma omp parallel for
    for (j = 0; j < frame_size; j++) {
        int moy = (p[j].r + p[j].g + p[j].b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;
        
        p[j].r = moy;
        p[j].g = moy;
        p[j].b = moy;
    }
}

void apply_blur_filter_frame(animated_gif *image, int frame_idx, int size, int threshold) {
    int j, k;
    int width = image->width[frame_idx];
    int height = image->height[frame_idx];
    int end = 0;
    int n_iter = 0;
    
    pixel *p = image->p[frame_idx];
    pixel *new = (pixel *)malloc(width * height * sizeof(pixel));
    
    /* Perform at least one blur iteration */
    do {
        end = 1;
        n_iter++;
        
        /* Initialize new pixels */
        #pragma omp parallel for private(k)
        for (j = 0; j < height; j++) {
            for (k = 0; k < width; k++) {
                new[CONV(j, k, width)].r = p[CONV(j, k, width)].r;
                new[CONV(j, k, width)].g = p[CONV(j, k, width)].g;
                new[CONV(j, k, width)].b = p[CONV(j, k, width)].b;
            }
        }
        
        /* Apply blur on entire image, not just top/bottom 10% */
        #pragma omp parallel for private(k) schedule(dynamic)
        for (j = size; j < height - size; j++) {
            for (k = size; k < width - size; k++) {
                int stencil_j, stencil_k;
                int t_r = 0;
                int t_g = 0;
                int t_b = 0;
                
                for (stencil_j = -size; stencil_j <= size; stencil_j++) {
                    for (stencil_k = -size; stencil_k <= size; stencil_k++) {
                        t_r += p[CONV(j + stencil_j, k + stencil_k, width)].r;
                        t_g += p[CONV(j + stencil_j, k + stencil_k, width)].g;
                        t_b += p[CONV(j + stencil_j, k + stencil_k, width)].b;
                    }
                }
                
                new[CONV(j, k, width)].r = t_r / ((2 * size + 1) * (2 * size + 1));
                new[CONV(j, k, width)].g = t_g / ((2 * size + 1) * (2 * size + 1));
                new[CONV(j, k, width)].b = t_b / ((2 * size + 1) * (2 * size + 1));
            }
        }
        
        /* Check convergence and update pixels */
        int local_end = 1;
        #pragma omp parallel for private(k) reduction(&:local_end)
        for (j = 1; j < height - 1; j++) {
            for (k = 1; k < width - 1; k++) {
                float diff_r = (new[CONV(j, k, width)].r - p[CONV(j, k, width)].r);
                float diff_g = (new[CONV(j, k, width)].g - p[CONV(j, k, width)].g);
                float diff_b = (new[CONV(j, k, width)].b - p[CONV(j, k, width)].b);
                
                if (diff_r > threshold || -diff_r > threshold ||
                    diff_g > threshold || -diff_g > threshold ||
                    diff_b > threshold || -diff_b > threshold) {
                    local_end = 0;
                }
                
                p[CONV(j, k, width)].r = new[CONV(j, k, width)].r;
                p[CONV(j, k, width)].g = new[CONV(j, k, width)].g;
                p[CONV(j, k, width)].b = new[CONV(j, k, width)].b;
            }
        }
        end = local_end;
    } while (threshold > 0 && !end);
    
    free(new);
}

void apply_sobel_filter_frame(animated_gif *image, int frame_idx) {
    int j, k;
    int width = image->width[frame_idx];
    int height = image->height[frame_idx];
    
    pixel *p = image->p[frame_idx];
    pixel *sobel = (pixel *)malloc(width * height * sizeof(pixel));
    
    #pragma omp parallel for private(k)
    for (j = 1; j < height - 1; j++) {
        for (k = 1; k < width - 1; k++) {
            int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
            int pixel_blue_so, pixel_blue_s, pixel_blue_se;
            int pixel_blue_o, pixel_blue, pixel_blue_e;
            
            float deltaX_blue;
            float deltaY_blue;
            float val_blue;
            
            /* Get neighboring pixel values */
            pixel_blue_no = p[CONV(j - 1, k - 1, width)].b;
            pixel_blue_n = p[CONV(j - 1, k, width)].b;
            pixel_blue_ne = p[CONV(j - 1, k + 1, width)].b;
            pixel_blue_so = p[CONV(j + 1, k - 1, width)].b;
            pixel_blue_s = p[CONV(j + 1, k, width)].b;
            pixel_blue_se = p[CONV(j + 1, k + 1, width)].b;
            pixel_blue_o = p[CONV(j, k - 1, width)].b;
            pixel_blue = p[CONV(j, k, width)].b;
            pixel_blue_e = p[CONV(j, k + 1, width)].b;
            
            /* Calculate gradient */
            deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 
                          2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;
            
            deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - 
                          pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;
            
            val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;
            
            /* Apply threshold */
            if (val_blue > 50) {
                sobel[CONV(j, k, width)].r = 255;
                sobel[CONV(j, k, width)].g = 255;
                sobel[CONV(j, k, width)].b = 255;
            } else {
                sobel[CONV(j, k, width)].r = 0;
                sobel[CONV(j, k, width)].g = 0;
                sobel[CONV(j, k, width)].b = 0;
            }
        }
    }
    
    /* Copy sobel result back to original image */
    #pragma omp parallel for private(k)
    for (j = 1; j < height - 1; j++) {
        for (k = 1; k < width - 1; k++) {
            p[CONV(j, k, width)].r = sobel[CONV(j, k, width)].r;
            p[CONV(j, k, width)].g = sobel[CONV(j, k, width)].g;
            p[CONV(j, k, width)].b = sobel[CONV(j, k, width)].b;
        }
    }
    
    free(sobel);
}
