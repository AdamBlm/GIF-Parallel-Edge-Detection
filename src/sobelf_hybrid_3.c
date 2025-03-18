/*
 * INF560
 *
 * Image Filtering Project - Hybrid Implementation (MPI + OpenMP + CUDA)
 *
 * This version combines:
 * - MPI for distributed processing of images across nodes
 * - OpenMP for thread-level parallelism within each node
 * - CUDA for GPU acceleration of pixel-processing operations
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <mpi.h>
#include <string.h>

#include "gif_lib.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Define MPI tags for messaging */
#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102
#define MPI_TAG_GPU_INFO 103

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images;     /* Number of images */
    int *width;       /* Width of each image */
    int *height;      /* Height of each image */
    pixel **p;        /* Pixels of each image */
    GifFileType *g;   /* Internal representation. DO NOT MODIFY */
} animated_gif;

/* GPU capabilities info */
typedef struct gpu_info {
    int has_gpu;           /* Whether this process has access to a GPU */
    int device_id;         /* Device ID for the GPU to use */
    int total_memory;      /* Total memory in MB */
    int compute_capability; /* Compute capability */
} gpu_info;

/* Function prototypes */
animated_gif* load_pixels(char* filename);
int store_pixels(char* filename, animated_gif* image);
void free_resources(pixel** p, int n_images);

/* CUDA wrapper functions (defined in sobelf_hybrid_3.cu) */
extern int init_cuda();
extern int get_gpu_count();
extern int get_gpu_info(int device_id, int* total_memory, int* compute_capability);
extern void process_image_gpu(pixel* pixels, int width, int height, pixel* output);

/* Apply grayscale filter to an image on CPU */
void apply_gray_filter_to_image_cpu(pixel* p, int width, int height)
{
    #pragma omp parallel for default(none) shared(p, width, height) schedule(guided)
    for (int j = 0; j < height; j++)
    {
        for (int k = 0; k < width; k++)
        {
            int index = j * width + k;
            int moy = (p[index].r + p[index].g + p[index].b) / 3;
            if (moy < 0) moy = 0;
            if (moy > 255) moy = 255;

            p[index].r = moy;
            p[index].g = moy;
            p[index].b = moy;
        }
    }
}

/* Apply blur filter to an image on CPU */
void apply_blur_filter_to_image_cpu(pixel* p, int width, int height, int size, int threshold, pixel* buffer)
{
    int end = 0;
    int n_iter = 0;
    
    // Initialize buffer with original pixel values
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(guided)
    for (int j = 0; j < height; j++) {
        for (int k = 0; k < width; k++) {
            int index = j * width + k;
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
        #pragma omp parallel for default(none) shared(p, buffer, width, height, size) schedule(guided)
        for (int j = size; j < height - size; j++)
        {
            for (int k = size; k < width - size; k++)
            {
                int t_r = 0;
                int t_g = 0;
                int t_b = 0;
                
                // Cache-friendly stencil computation
                for (int stencil_j = -size; stencil_j <= size; stencil_j++)
                {
                    const int row_offset = (j + stencil_j) * width;
                    
                    for (int stencil_k = -size; stencil_k <= size; stencil_k++)
                    {
                        int idx = row_offset + (k + stencil_k);
                        t_r += p[idx].r;
                        t_g += p[idx].g;
                        t_b += p[idx].b;
                    }
                }

                // Calculate the average value for the pixel
                const int denom = (2 * size + 1) * (2 * size + 1);
                buffer[j * width + k].r = t_r / denom;
                buffer[j * width + k].g = t_g / denom;
                buffer[j * width + k].b = t_b / denom;
            }
        }

        // Check convergence and update pixels
        int continue_flag = 0;
        
        #pragma omp parallel for default(none) shared(p, buffer, width, height, size, threshold) reduction(|:continue_flag) schedule(guided)
        for (int j = size; j < height - size; j++)
        {
            for (int k = size; k < width - size; k++)
            {
                int index = j * width + k;
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
    while (threshold > 0 && !end && n_iter < 50); // Cap iterations at 50 to avoid infinite loops

#if SOBELF_DEBUG
    printf("BLUR: number of iterations: %d\n", n_iter);
#endif
}

/* Apply sobel filter to an image on CPU */
void apply_sobel_filter_to_image_cpu(pixel* p, int width, int height, pixel* buffer)
{
    // Apply Sobel filter with improved memory access pattern
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(guided)
    for (int j = 1; j < height - 1; j++)
    {
        // Calculate row offsets for better cache locality
        const int row_prev = (j - 1) * width;
        const int row_curr = j * width;
        const int row_next = (j + 1) * width;
        
        for (int k = 1; k < width - 1; k++)
        {
            // Get pixel values from the 3x3 neighborhood with better locality
            int pixel_blue_no = p[row_prev + k - 1].b;
            int pixel_blue_n  = p[row_prev + k].b;
            int pixel_blue_ne = p[row_prev + k + 1].b;
            int pixel_blue_o  = p[row_curr + k - 1].b;
            int pixel_blue_e  = p[row_curr + k + 1].b;
            int pixel_blue_so = p[row_next + k - 1].b;
            int pixel_blue_s  = p[row_next + k].b;
            int pixel_blue_se = p[row_next + k + 1].b;

            // Calculate Sobel gradients
            float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;
            float deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;
            float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4;

            // Apply threshold to create binary edge image
            if (val_blue > 50)
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
    
    // Copy sobel results back to original image
    #pragma omp parallel for default(none) shared(p, buffer, width, height) schedule(guided)
    for (int j = 1; j < height - 1; j++)
    {
        for (int k = 1; k < width - 1; k++)
        {
            int index = j * width + k;
            p[index].r = buffer[index].r;
            p[index].g = buffer[index].g;
            p[index].b = buffer[index].b;
        }
    }
}

/* Process a single image either on GPU or CPU depending on availability */
void process_image(pixel* img_pixels, int width, int height, gpu_info* gpu, pixel* buffer)
{
    // Allocate a buffer for the temporary output on GPU if we need to create it
    pixel* output_buffer = NULL;
    
    if (gpu->has_gpu) {
        // Process the whole image on GPU in one pass
        output_buffer = (pixel*)malloc(width * height * sizeof(pixel));
        if (output_buffer == NULL) {
            fprintf(stderr, "Failed to allocate output buffer for GPU processing, falling back to CPU\n");
            gpu->has_gpu = 0; // Fallback to CPU
        }
    }
    
    if (gpu->has_gpu) {
        // GPU processing path
        #if SOBELF_DEBUG
        printf("Processing image of size %dx%d on GPU (device %d)\n", width, height, gpu->device_id);
        #endif
        
        // Process the image using CUDA
        process_image_gpu(img_pixels, width, height, output_buffer);
        
        // Copy the result back to the input buffer
        memcpy(img_pixels, output_buffer, width * height * sizeof(pixel));
        
        // Free the output buffer
        free(output_buffer);
    } else {
        // CPU processing path
        #if SOBELF_DEBUG
        printf("Processing image of size %dx%d on CPU\n", width, height);
        #endif
        
        // Apply the filters in sequence on CPU
        apply_gray_filter_to_image_cpu(img_pixels, width, height);
        apply_blur_filter_to_image_cpu(img_pixels, width, height, 5, 20, buffer);
        apply_sobel_filter_to_image_cpu(img_pixels, width, height, buffer);
    }
}

/* Main entry point */
int main(int argc, char** argv)
{
    char* input_filename;
    char* output_filename;
    animated_gif* image = NULL;
    struct timeval t1, t2;
    double duration;
    int rank, size;
    MPI_Status status;
    gpu_info gpu_data;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);
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

    /* Initialize CUDA and check GPU availability */
    gpu_data.has_gpu = 0;
    gpu_data.device_id = -1;
    gpu_data.total_memory = 0;
    gpu_data.compute_capability = 0;
    
    // Initialize CUDA and find available GPUs
    if (init_cuda() == 0) {
        int gpu_count = get_gpu_count();
        if (gpu_count > 0) {
            // Assign GPUs to processes in a round-robin fashion if multiple GPUs are available
            int device_to_use = rank % gpu_count;
            if (get_gpu_info(device_to_use, &gpu_data.total_memory, &gpu_data.compute_capability) == 0) {
                gpu_data.has_gpu = 1;
                gpu_data.device_id = device_to_use;
            }
        }
    }

    if (rank == 0) {
        printf("Using %d MPI processes with %d OpenMP threads per process\n", 
               size, omp_get_max_threads());
               
        // Print GPU information for rank 0
        if (gpu_data.has_gpu) {
            printf("Rank 0 using GPU device %d with %d MB memory, compute capability %d.%d\n", 
                   gpu_data.device_id, 
                   gpu_data.total_memory, 
                   gpu_data.compute_capability / 10, 
                   gpu_data.compute_capability % 10);
        } else {
            printf("Rank 0 using CPU only (no GPU available)\n");
        }
    }

    // Share GPU information across all ranks
    int* all_gpus = (int*)malloc(size * sizeof(int));
    MPI_Allgather(&gpu_data.has_gpu, 1, MPI_INT, all_gpus, 1, MPI_INT, MPI_COMM_WORLD);
    
    // Count the number of GPUs available
    int total_gpus = 0;
    for (int i = 0; i < size; i++) {
        if (all_gpus[i]) total_gpus++;
    }
    
    if (rank == 0) {
        printf("Total GPUs available: %d out of %d MPI processes\n", total_gpus, size);
    }
    free(all_gpus);

    int n_images = 0;
    int* image_widths = NULL;
    int* image_heights = NULL;
    
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
    }

    /* Broadcast the number of images to all processes */
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Allocate memory for image dimensions on worker processes */
    if (rank != 0) {
        image_widths = (int*)malloc(n_images * sizeof(int));
        image_heights = (int*)malloc(n_images * sizeof(int));
    }
    
    /* Broadcast the image dimensions to all processes */
    MPI_Bcast(image_widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image_heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* FILTER Timer start */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t1, NULL);
    }

    /* Calculate how many images each process will handle */
    int images_per_process = n_images / size;
    int remaining_images = n_images % size;
    
    int start_image = rank * images_per_process + (rank < remaining_images ? rank : remaining_images);
    int end_image = start_image + images_per_process + (rank < remaining_images ? 1 : 0);
    int my_n_images = end_image - start_image;
    
    /* Allocate memory for local image processing */
    pixel** my_pixels = NULL;
    pixel** filter_buffers = NULL;
    int* my_widths = NULL;
    int* my_heights = NULL;
    
    if (my_n_images > 0) {
        /* Allocate memory for local image data */
        my_pixels = (pixel**)malloc(my_n_images * sizeof(pixel*));
        filter_buffers = (pixel**)malloc(my_n_images * sizeof(pixel*));
        my_widths = (int*)malloc(my_n_images * sizeof(int));
        my_heights = (int*)malloc(my_n_images * sizeof(int));
        
        /* Initialize local image dimensions */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = start_image + i;
            my_widths[i] = image_widths[img_idx];
            my_heights[i] = image_heights[img_idx];
            
            /* Allocate memory for local image data and buffers */
            my_pixels[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
            filter_buffers[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
        }
    }
    
    /* Process 0 sends image data to worker processes */
    if (rank == 0) {
        /* Process 0 copies its own portion of the data */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = start_image + i;
            memcpy(my_pixels[i], image->p[img_idx], 
                   my_widths[i] * my_heights[i] * sizeof(pixel));
        }
        
        /* Send data to worker processes */
        for (int i = 1; i < size; i++) {
            int worker_start = i * images_per_process + (i < remaining_images ? i : remaining_images);
            int worker_count = images_per_process + (i < remaining_images ? 1 : 0);
            
            for (int j = 0; j < worker_count; j++) {
                int img_idx = worker_start + j;
                MPI_Send(image->p[img_idx], image_widths[img_idx] * image_heights[img_idx] * sizeof(pixel),
                         MPI_BYTE, i, MPI_TAG_IMAGE_DATA + j, MPI_COMM_WORLD);
            }
        }
    } else {
        /* Worker processes receive their assigned images */
        for (int i = 0; i < my_n_images; i++) {
            MPI_Recv(my_pixels[i], my_widths[i] * my_heights[i] * sizeof(pixel),
                     MPI_BYTE, 0, MPI_TAG_IMAGE_DATA + i, MPI_COMM_WORLD, &status);
        }
    }
    
    /* Apply the filters to the local images */
    if (my_n_images > 0) {
        for (int i = 0; i < my_n_images; i++) {
            // Process each image either on GPU or CPU based on availability
            process_image(my_pixels[i], my_widths[i], my_heights[i], &gpu_data, filter_buffers[i]);
        }
    }
    
    /* Send processed images back to the master process */
    if (rank == 0) {
        /* Process 0 copies its own processed data back */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = start_image + i;
            memcpy(image->p[img_idx], my_pixels[i], 
                   my_widths[i] * my_heights[i] * sizeof(pixel));
        }
        
        /* Receive data from worker processes */
        for (int i = 1; i < size; i++) {
            int worker_start = i * images_per_process + (i < remaining_images ? i : remaining_images);
            int worker_count = images_per_process + (i < remaining_images ? 1 : 0);
            
            for (int j = 0; j < worker_count; j++) {
                int img_idx = worker_start + j;
                MPI_Recv(image->p[img_idx], image_widths[img_idx] * image_heights[img_idx] * sizeof(pixel),
                         MPI_BYTE, i, MPI_TAG_RESULT + j, MPI_COMM_WORLD, &status);
            }
        }
    } else {
        /* Worker processes send their processed images back */
        for (int i = 0; i < my_n_images; i++) {
            MPI_Send(my_pixels[i], my_widths[i] * my_heights[i] * sizeof(pixel),
                     MPI_BYTE, 0, MPI_TAG_RESULT + i, MPI_COMM_WORLD);
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
        for (int i = 0; i < my_n_images; i++) {
            free(my_pixels[i]);
            free(filter_buffers[i]);
        }
        free(my_pixels);
        free(filter_buffers);
        free(my_widths);
        free(my_heights);
    }
    
    if (rank == 0 && image) {
        free_resources(image->p, image->n_images);
        free(image->width);
        free(image->height);
        free(image);
    } else if (rank != 0) {
        free(image_widths);
        free(image_heights);
    }
    
    MPI_Finalize();
    return 0;
} 
