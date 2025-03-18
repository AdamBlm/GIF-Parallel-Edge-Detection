/*
 * INF560
 *
 * Image Filtering Project - Hybrid MPI + CUDA Implementation
 */
#include "cuda_mpi_filter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include <mpi.h>


#include "cuda_common.h"
#include "gif_lib.h"
#include "gif_io.h"

/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Define MPI tags for messaging */
#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102



/* Struct to send image data between processes */
typedef struct image_data {
    int width;
    int height;
    int image_idx;
    int total_size;
} image_data;



/* Main CUDA processing function for a single image */
static void process_image_cuda(pixel *h_pixels, int width, int height)
{
    if (h_pixels == NULL || width <= 0 || height <= 0) {
        fprintf(stderr, "Error: Invalid input parameters to process_image_cuda\n");
        return;
    }

    // Calculate buffer sizes
    size_t pixelsSize = width * height * sizeof(pixel);
    
    // Allocate device memory with error checking
    pixel *d_pixels = NULL, *d_temp = NULL;
    int *d_continue = NULL;
    cudaError_t cuda_err;
    
    // Allocate device memory with proper error checking
    cuda_err = cudaMalloc((void**)&d_pixels, pixelsSize);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMalloc d_pixels: %s\n", 
                cudaGetErrorString(cuda_err));
        return;
    }
    
    cuda_err = cudaMalloc((void**)&d_temp, pixelsSize);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMalloc d_temp: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        return;
    }
    
    cuda_err = cudaMalloc((void**)&d_continue, sizeof(int));
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMalloc d_continue: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        return;
    }
    
    // Copy input data to device with error checking
    cuda_err = cudaMemcpy(d_pixels, h_pixels, pixelsSize, cudaMemcpyHostToDevice);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemcpy HostToDevice: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        cudaFree(d_continue);
        return;
    }
    
    // Setup execution parameters
    dim3 blockSize(TILE_WIDTH, TILE_HEIGHT);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Verify block size is valid
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (blockSize.x * blockSize.y > deviceProp.maxThreadsPerBlock) {
        fprintf(stderr, "Block size exceeds device limits. Adjusting...\n");
        blockSize.x = 16;
        blockSize.y = 16;
        gridSize.x = (width + blockSize.x - 1) / blockSize.x;
        gridSize.y = (height + blockSize.y - 1) / blockSize.y;
    }
    
    // Apply grayscale filter
    grayscale_kernel<<<gridSize, blockSize>>>(d_pixels, width, height);
    cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in grayscale_kernel: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        cudaFree(d_continue);
        return;
    }
    
    // Apply blur filter with convergence
    const int blurSize = 3;  // Same as in the OpenMP version
    const int threshold = 0; // Same as in the OpenMP version
    
    // Calculate shared memory size for blur kernel
    size_t sharedMemSizeBlur = (blockSize.x + 2 * blurSize) * 
                              (blockSize.y + 2 * blurSize) * 
                              sizeof(pixel);
    
    // Check if we have enough shared memory
    if (sharedMemSizeBlur > deviceProp.sharedMemPerBlock) {
        fprintf(stderr, "Warning: Shared memory requirement exceeds device limits. Performance may suffer.\n");
        // Fall back to a smaller tile size if needed
        if (sharedMemSizeBlur > deviceProp.sharedMemPerBlock * 2) {
            blockSize.x /= 2;
            blockSize.y /= 2;
            gridSize.x = (width + blockSize.x - 1) / blockSize.x;
            gridSize.y = (height + blockSize.y - 1) / blockSize.y;
            sharedMemSizeBlur = (blockSize.x + 2 * blurSize) * 
                              (blockSize.y + 2 * blurSize) * 
                              sizeof(pixel);
        }
    }
    
    if (threshold > 0) {
        int h_continue;
        int max_iterations = 50;  // Limit iterations to avoid infinite loop
        int iter = 0;
        
        do {
            // Initialize continue flag
            h_continue = 0;
            cuda_err = cudaMemcpy(d_continue, &h_continue, sizeof(int), cudaMemcpyHostToDevice);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in cudaMemcpy HostToDevice for d_continue: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            // Apply blur
            blur_kernel<<<gridSize, blockSize, sharedMemSizeBlur>>>(
                d_pixels, d_temp, width, height, blurSize);
            cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in blur_kernel: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            // Check for convergence
            check_convergence_kernel<<<gridSize, blockSize>>>(
                d_pixels, d_temp, width, height, threshold, d_continue);
            cuda_err = cudaGetLastError();
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in check_convergence_kernel: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            // Swap buffers
            pixel *temp = d_pixels;
            d_pixels = d_temp;
            d_temp = temp;
            
            // Get continue flag
            cuda_err = cudaMemcpy(&h_continue, d_continue, sizeof(int), cudaMemcpyDeviceToHost);
            if (cuda_err != cudaSuccess) {
                fprintf(stderr, "CUDA error in cudaMemcpy DeviceToHost for d_continue: %s\n", 
                        cudaGetErrorString(cuda_err));
                break;
            }
            
            iter++;
        } while (h_continue && iter < max_iterations);
        
#if SOBELF_DEBUG
        printf("BLUR: number of iterations: %d\n", iter);
#endif
    } else {
        // Just apply blur once
        blur_kernel<<<gridSize, blockSize, sharedMemSizeBlur>>>(
            d_pixels, d_temp, width, height, blurSize);
        cuda_err = cudaGetLastError();
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "CUDA error in blur_kernel: %s\n", 
                    cudaGetErrorString(cuda_err));
            cudaFree(d_pixels);
            cudaFree(d_temp);
            cudaFree(d_continue);
            return;
        }
        
        // Swap buffers
        pixel *temp = d_pixels;
        d_pixels = d_temp;
        d_temp = temp;
    }
    
    // Apply Sobel filter
    size_t sharedMemSizeSobel = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(pixel);
    sobel_kernel<<<gridSize, blockSize, sharedMemSizeSobel>>>(d_pixels, d_temp, width, height);
    cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in sobel_kernel: %s\n", 
                cudaGetErrorString(cuda_err));
        cudaFree(d_pixels);
        cudaFree(d_temp);
        cudaFree(d_continue);
        return;
    }
    
    // Copy results back to host with error checking
    cuda_err = cudaMemcpy(h_pixels, d_temp, pixelsSize, cudaMemcpyDeviceToHost);
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA error in cudaMemcpy DeviceToHost: %s\n", 
                cudaGetErrorString(cuda_err));
    }
    
    // Free device memory
    cudaFree(d_pixels);
    cudaFree(d_temp);
    cudaFree(d_continue);
    
    // Make sure all CUDA operations are complete
    cudaDeviceSynchronize();
}




int run_cuda_mpi_filter(char *input_filename, char *output_filename) {
 
    animated_gif *image = NULL;
    struct timeval t1, t2;
    double duration;
    int rank, size;
    MPI_Status status;
    scatter_info *scatter_data = NULL;
    int gpu_count = 0;

  
    int provided;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

  

    
    /* Initialize CUDA and get GPU count */
    cudaError_t cuda_err = cudaGetDeviceCount(&gpu_count);
    if (cuda_err != cudaSuccess) {
        if (rank == 0) {
            fprintf(stderr, "Error getting CUDA device count: %s\n", cudaGetErrorString(cuda_err));
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Using %d MPI processes with %d CUDA-capable GPU(s)\n", size, gpu_count);
    }
    
    /* Set GPU device for this MPI process */
    if (gpu_count > 0) {
        int gpu_id = rank % gpu_count;
        
        // Reset device before using it
        cudaDeviceReset();
        
        // Set device with error checking
        cuda_err = cudaSetDevice(gpu_id);
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Process %d: Error setting GPU device %d: %s\n", 
                    rank, gpu_id, cudaGetErrorString(cuda_err));
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Set up exclusive process mode for better multi-process support
        cudaSetDeviceFlags(cudaDeviceScheduleYield);
        
        printf("Process %d using GPU %d\n", rank, gpu_id);
        
        // Verify device is working by trying a simple operation
        int *d_test = NULL;
        cuda_err = cudaMalloc((void**)&d_test, sizeof(int));
        if (cuda_err != cudaSuccess) {
            fprintf(stderr, "Process %d: Error allocating test memory on GPU %d: %s\n", 
                   rank, gpu_id, cudaGetErrorString(cuda_err));
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        cudaFree(d_test);
        
    } else {
        if (rank == 0) {
            fprintf(stderr, "Warning: No CUDA-capable GPUs detected\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }

    // Make sure all processes have initialized their GPU before proceeding
    MPI_Barrier(MPI_COMM_WORLD);

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
        
        // Check if we have valid data before proceeding
        if (n_images <= 0) {
            fprintf(stderr, "Error: No images found in input file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
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
        
        // Create scatter info structure on non-root processes too
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Broadcast the image dimensions to all processes */
    MPI_Bcast(image_widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image_heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);

    /* Broadcast scatter info arrays in a single operation */
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
        
        // Calculate pixel counts and displacements 
        for (int i = 0; i < size; i++) {
            scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
            scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
        }
        
        // Recalculate dimensions if needed
        if (n_images > 0) {
            calculate_pixel_counts(scatter_data, image_widths, image_heights, size);
        }
    }
    
    free(all_scatter_info);
    
    /* Sync before starting timing */
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* FILTER Timer start */
    if (rank == 0) {
        gettimeofday(&t1, NULL);
    }

    /* Get information for this process */
    int my_n_images = scatter_data->image_counts[rank];
    int my_start_image = scatter_data->image_displs[rank];
    
    /* Allocate memory for local image processing */
    pixel **my_pixels = NULL;
    int *my_widths = NULL;
    int *my_heights = NULL;
    
    if (my_n_images > 0) {
        /* Allocate memory for local image data */
        my_pixels = (pixel**)malloc(my_n_images * sizeof(pixel*));
        my_widths = (int*)malloc(my_n_images * sizeof(int));
        my_heights = (int*)malloc(my_n_images * sizeof(int));
        
        if (!my_pixels || !my_widths || !my_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Initialize local image dimensions and allocate buffers */
        for (int i = 0; i < my_n_images; i++) {
            int img_idx = my_start_image + i;
            my_widths[i] = image_widths[img_idx];
            my_heights[i] = image_heights[img_idx];
            
            /* Allocate memory for local image data */
            my_pixels[i] = (pixel*)malloc(my_widths[i] * my_heights[i] * sizeof(pixel));
            
            if (!my_pixels[i]) {
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
    
    /* Use optimized MPI_Scatterv with pre-calculated byte counts */
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
    
    /* Apply the filters to the local images using CUDA */
    if (my_n_images > 0) {
        for (int i = 0; i < my_n_images; i++) {
            process_image_cuda(my_pixels[i], my_widths[i], my_heights[i]);
        }
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
    
    /* Make sure all processes are synchronized before gather */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Debug information to check for buffer size mismatches */
    if (rank == 0 && SOBELF_DEBUG) {
        printf("Gathering data from processes: \n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: %d pixels (%d bytes) at displacement %d (%d bytes)\n", 
                   i, scatter_data->sendcounts[i], scatter_data->scatter_byte_counts[i],
                   scatter_data->displs[i], scatter_data->scatter_byte_displs[i]);
        }
    }
    
    /* Use optimized MPI_Gatherv with pre-calculated byte counts */
    MPI_Gatherv(gathered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
                
    /* Master copies results back to image structure */
    if (rank == 0) {
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
        free(my_widths);
        free(my_heights);
    }
    
    if (gathered_pixels) free(gathered_pixels);
    if (all_pixels) free(all_pixels);
    
    free_scatter_info(scatter_data);
    
    if (rank == 0 && image) {
        free_resources(image->p, image->n_images);
        free(image->width);
        free(image->height);
        free(image);
    }
    

    return 0;
}