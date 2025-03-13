/*
 * Hybrid MPI+OpenMP+CUDA Implementation
 * This file integrates all three parallelization approaches
 */

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/* MPI tags for communication */
#define MPI_TAG_FRAME_COUNT 100
#define MPI_TAG_FRAME_SIZE  101
#define MPI_TAG_FRAME_DATA  102
#define MPI_TAG_RESULT_DATA 103

/* CUDA error checking macro */
#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return 1; \
        } \
    }

/* Forward declarations for CUDA functions */
int process_frame_cuda(pixel *frame_pixels, int width, int height);

/* Main function with hybrid parallelization */
int main(int argc, char **argv) {
    char *input_filename;
    char *output_filename;
    animated_gif *image = NULL;
    struct timeval t1, t2;
    double duration;
    
    int rank, size, i, j;
    int frames_per_process;
    int *sendcounts = NULL;
    int *displs = NULL;
    MPI_Status status;
    int has_cuda = 0;
    
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
    
    /* Check for CUDA availability */
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        has_cuda = 1;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        if (rank == 0) {
            printf("Process %d using CUDA device: %s\n", rank, deviceProp.name);
        }
    } else {
        if (rank == 0) {
            printf("No CUDA devices found, using CPU implementation\n");
        }
    }
    
    /* Set OpenMP threads */
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    if (rank == 0) {
        printf("Each process using %d OpenMP threads\n", num_threads);
    }
    
    /* MASTER PROCESS: Load file and distribute work */
    if (rank == 0) {
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
        
        /* FILTER Timer start */
        gettimeofday(&t1, NULL);
        
        /* Fix: Remove black line function which is defined but not used */
        /* apply_gray_line() is not used in the optimized version */
    }
    
    /* Broadcast the number of images to all processes */
    int n_images = 0;
    if (rank == 0) {
        n_images = image->n_images;
    }
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Calculate frames distribution */
    frames_per_process = n_images / size;
    int remainder = n_images % size;
    
    /* Each process calculates its own frame range */
    int start_frame = rank * frames_per_process + (rank < remainder ? rank : remainder);
    int end_frame = start_frame + frames_per_process + (rank < remainder ? 1 : 0);
    int num_frames = end_frame - start_frame;
    
    /* Master process allocates arrays for gathering results */
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        displs[0] = 0;
        for (i = 0; i < size; i++) {
            sendcounts[i] = (frames_per_process + (i < remainder ? 1 : 0));
            if (i > 0) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
        }
    }
    
    /* Broadcast image dimensions to all processes */
    int *width = NULL;
    int *height = NULL;
    
    if (rank == 0) {
        width = image->width;
        height = image->height;
    } else {
        width = (int *)malloc(n_images * sizeof(int));
        height = (int *)malloc(n_images * sizeof(int));
    }
    
    MPI_Bcast(width, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(height, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Allocate memory for local frames */
    pixel **local_frames = (pixel **)malloc(num_frames * sizeof(pixel *));
    for (i = 0; i < num_frames; i++) {
        int frame_idx = start_frame + i;
        int frame_size = width[frame_idx] * height[frame_idx];
        local_frames[i] = (pixel *)malloc(frame_size * sizeof(pixel));
    }
    
    /* Distribute frame data */
    for (i = 0; i < num_frames; i++) {
        int frame_idx = start_frame + i;
        int frame_size = width[frame_idx] * height[frame_idx];
        
        if (rank == 0) {
            /* Master process copies its own frames directly */
            memcpy(local_frames[i], image->p[frame_idx], frame_size * sizeof(pixel));
        } else {
            /* Worker processes receive frame data from master */
            MPI_Recv(local_frames[i], frame_size * sizeof(pixel), MPI_BYTE,
                     0, MPI_TAG_FRAME_DATA + frame_idx, MPI_COMM_WORLD, &status);
        }
    }
    
    /* Send frame data to worker processes */
    if (rank == 0) {
        for (i = 1; i < size; i++) {
            for (j = 0; j < sendcounts[i]; j++) {
                int frame_idx = displs[i] + j;
                int frame_size = width[frame_idx] * height[frame_idx];
                
                MPI_Send(image->p[frame_idx], frame_size * sizeof(pixel), MPI_BYTE,
                         i, MPI_TAG_FRAME_DATA + frame_idx, MPI_COMM_WORLD);
            }
        }
    }
    
    /* Process local frames */
    for (i = 0; i < num_frames; i++) {
        int frame_idx = start_frame + i;
        
        if (has_cuda) {
            /* Use CUDA to process the frame */
            process_frame_cuda(local_frames[i], width[frame_idx], height[frame_idx]);
        } else {
            /* Use OpenMP to process the frame */
            apply_gray_filter_frame(local_frames[i], width[frame_idx], height[frame_idx]);
            apply_blur_filter_frame(local_frames[i], width[frame_idx], height[frame_idx], 5, 20);
            apply_sobel_filter_frame(local_frames[i], width[frame_idx], height[frame_idx]);
        }
    }
    
    /* Gather results back to master process */
    if (rank == 0) {
        /* Master process copies its own processed frames directly */
        for (i = 0; i < sendcounts[0]; i++) {
            int frame_idx = displs[0] + i;
            int frame_size = width[frame_idx] * height[frame_idx];
            
            memcpy(image->p[frame_idx], local_frames[i], frame_size * sizeof(pixel));
        }
        
        /* Receive processed frames from workers */
        for (i = 1; i < size; i++) {
            for (j = 0; j < sendcounts[i]; j++) {
                int frame_idx = displs[i] + j;
                int frame_size = width[frame_idx] * height[frame_idx];
                
                MPI_Recv(image->p[frame_idx], frame_size * sizeof(pixel), MPI_BYTE,
                         i, MPI_TAG_RESULT_DATA + frame_idx, MPI_COMM_WORLD, &status);
            }
        }
    } else {
        /* Worker processes send their processed frames back to master */
        for (i = 0; i < num_frames; i++) {
            int frame_idx = start_frame + i;
            int frame_size = width[frame_idx] * height[frame_idx];
            
            MPI_Send(local_frames[i], frame_size * sizeof(pixel), MPI_BYTE,
                     0, MPI_TAG_RESULT_DATA + frame_idx, MPI_COMM_WORLD);
        }
    }
    
    /* Free local memory */
    for (i = 0; i < num_frames; i++) {
        free(local_frames[i]);
    }
    free(local_frames);
    
    if (rank != 0) {
        free(width);
        free(height);
    }
    
    /* Store the final result and clean up */
    if (rank == 0) {
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("SOBEL done in %lf s (Hybrid MPI+OpenMP+CUDA)\n", duration);
        
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
    
    MPI_Finalize();
    return 0;
}

/* OpenMP version of frame-specific filter functions */
void apply_gray_filter_frame(pixel *p, int width, int height) {
    int j;
    int frame_size = width * height;
    
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

void apply_blur_filter_frame(pixel *p, int width, int height, int size, int threshold) {
    int j, k;
    int end = 0;
    int n_iter = 0;
    
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
        
        /* Apply blur on entire image (fixing the partial blur issue) */
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

void apply_sobel_filter_frame(pixel *p, int width, int height) {
    int j, k;
    
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

/* CUDA implementation for processing a single frame */
int process_frame_cuda(pixel *frame_pixels, int width, int height) {
    int num_pixels = width * height;
    size_t size = num_pixels * sizeof(pixel);
    
    /* Step 1: Apply grayscale filter */
    pixel *d_pixels;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pixels, size));
    CHECK_CUDA_ERROR(cudaMemcpy(d_pixels, frame_pixels, size, cudaMemcpyHostToDevice));
    
    int blockSize = 256;
    int numBlocks = (num_pixels + blockSize - 1) / blockSize;
    
    grayscaleKernel<<<numBlocks, blockSize>>>(d_pixels, num_pixels);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(frame_pixels, d_pixels, size, cudaMemcpyDeviceToHost));
    
    /* Step 2: Apply blur filter */
    pixel *d_output;
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, size));
    
    int size_param = 5;
    int threshold = 20;
    int n_iter = 0;
    int end = 0;
    
    do {
        n_iter++;
        
        dim3 blurBlockSize(16, 16);
        dim3 blurGridSize((width - 2 * size_param + blurBlockSize.x - 1) / blurBlockSize.x, 
                         (height - 2 * size_param + blurBlockSize.y - 1) / blurBlockSize.y);
        
        blurKernel<<<blurGridSize, blurBlockSize>>>(d_pixels, d_output, width, height, size_param);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        /* Check convergence on CPU */
        CHECK_CUDA_ERROR(cudaMemcpy(frame_pixels, d_output, size, cudaMemcpyDeviceToHost));
        
        /* Swap buffers */
        pixel *temp = d_pixels;
        d_pixels = d_output;
        d_output = temp;
        
        /* Use the host pixels for convergence check */
        end = 1;
        #pragma omp parallel for reduction(&:end)
        for (int j = 1; j < height - 1; j++) {
            for (int k = 1; k < width - 1; k++) {
                float diff = abs(frame_pixels[CONV(j, k, width)].r - frame_pixels[CONV(j, k, width)].r);
                if (diff > threshold) {
                    end = 0;
                }
            }
        }
    } while (threshold > 0 && !end && n_iter < 10); /* Add iteration limit to prevent infinite loops */
    
    /* Step 3: Apply Sobel filter */
    dim3 sobelBlockSize(16, 16);
    dim3 sobelGridSize((width - 2 + sobelBlockSize.x - 1) / sobelBlockSize.x, 
                     (height - 2 + sobelBlockSize.y - 1) / sobelBlockSize.y);
    
    CHECK_CUDA_ERROR(cudaMemset(d_output, 0, size));
    
    sobelKernel<<<sobelGridSize, sobelBlockSize>>>(d_pixels, d_output, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(frame_pixels, d_output, size, cudaMemcpyDeviceToHost));
    
    /* Free device memory */
    CHECK_CUDA_ERROR(cudaFree(d_pixels));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    return 0;
}

/* CUDA kernel declarations */
__global__ void grayscaleKernel(pixel *d_pixels, int num_pixels);
__global__ void blurKernel(pixel *d_input, pixel *d_output, int width, int height, int size);
__global__ void sobelKernel(pixel *d_input, pixel *d_output, int width, int height);
