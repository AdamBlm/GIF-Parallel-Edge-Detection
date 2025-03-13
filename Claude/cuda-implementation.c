/*
 * CUDA Implementation for GPU-accelerated image processing
 */

#include <cuda_runtime.h>

/* CUDA error checking macro */
#define CHECK_CUDA_ERROR(call) \
    { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    }

/* CUDA kernel for grayscale filter */
__global__ void grayscaleKernel(pixel *d_pixels, int num_pixels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pixels) {
        int moy = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;
        
        d_pixels[idx].r = moy;
        d_pixels[idx].g = moy;
        d_pixels[idx].b = moy;
    }
}

/* CUDA kernel for blur filter */
__global__ void blurKernel(pixel *d_input, pixel *d_output, int width, int height, int size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + size;
    int row = blockIdx.y * blockDim.y + threadIdx.y + size;
    
    if (col < width - size && row < height - size) {
        int t_r = 0;
        int t_g = 0;
        int t_b = 0;
        
        /* Apply the stencil */
        for (int stencil_j = -size; stencil_j <= size; stencil_j++) {
            for (int stencil_k = -size; stencil_k <= size; stencil_k++) {
                int idx = (row + stencil_j) * width + (col + stencil_k);
                t_r += d_input[idx].r;
                t_g += d_input[idx].g;
                t_b += d_input[idx].b;
            }
        }
        
        int idx = row * width + col;
        d_output[idx].r = t_r / ((2 * size + 1) * (2 * size + 1));
        d_output[idx].g = t_g / ((2 * size + 1) * (2 * size + 1));
        d_output[idx].b = t_b / ((2 * size + 1) * (2 * size + 1));
    }
}

/* CUDA kernel for Sobel filter */
__global__ void sobelKernel(pixel *d_input, pixel *d_output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int row = blockIdx.y * blockDim.y + threadIdx.y + 1;
    
    if (col < width - 1 && row < height - 1) {
        int idx = row * width + col;
        
        /* Get neighboring pixel values for blue channel */
        int pixel_blue_no = d_input[(row - 1) * width + (col - 1)].b;
        int pixel_blue_n  = d_input[(row - 1) * width + col].b;
        int pixel_blue_ne = d_input[(row - 1) * width + (col + 1)].b;
        int pixel_blue_so = d_input[(row + 1) * width + (col - 1)].b;
        int pixel_blue_s  = d_input[(row + 1) * width + col].b;
        int pixel_blue_se = d_input[(row + 1) * width + (col + 1)].b;
        int pixel_blue_o  = d_input[row * width + (col - 1)].b;
        int pixel_blue_e  = d_input[row * width + (col + 1)].b;
        
        /* Calculate gradient */
        float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2 * pixel_blue_o + 
                          2 * pixel_blue_e - pixel_blue_so + pixel_blue_se;
                          
        float deltaY_blue = pixel_blue_se + 2 * pixel_blue_s + pixel_blue_so - 
                          pixel_blue_ne - 2 * pixel_blue_n - pixel_blue_no;
                          
        float val_blue = sqrtf(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue) / 4.0f;
        
        /* Apply threshold */
        if (val_blue > 50.0f) {
            d_output[idx].r = 255;
            d_output[idx].g = 255;
            d_output[idx].b = 255;
        } else {
            d_output[idx].r = 0;
            d_output[idx].g = 0;
            d_output[idx].b = 0;
        }
    }
}

/* CUDA implementation of grayscale filter */
void apply_gray_filter_cuda(animated_gif *image) {
    pixel *d_pixels;
    int i;
    
    for (i = 0; i < image->n_images; i++) {
        int num_pixels = image->width[i] * image->height[i];
        size_t size = num_pixels * sizeof(pixel);
        
        /* Allocate device memory */
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_pixels, size));
        
        /* Copy data to device */
        CHECK_CUDA_ERROR(cudaMemcpy(d_pixels, image->p[i], size, cudaMemcpyHostToDevice));
        
        /* Configure kernel launch parameters */
        int blockSize = 256;
        int numBlocks = (num_pixels + blockSize - 1) / blockSize;
        
        /* Launch kernel */
        grayscaleKernel<<<numBlocks, blockSize>>>(d_pixels, num_pixels);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        /* Copy results back to host */
        CHECK_CUDA_ERROR(cudaMemcpy(image->p[i], d_pixels, size, cudaMemcpyDeviceToHost));
        
        /* Free device memory */
        CHECK_CUDA_ERROR(cudaFree(d_pixels));
    }
}

/* CUDA implementation of blur filter */
void apply_blur_filter_cuda(animated_gif *image, int size, int threshold) {
    int i;
    
    for (i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        int num_pixels = width * height;
        size_t mem_size = num_pixels * sizeof(pixel);
        
        pixel *d_input, *d_output;
        
        /* Allocate device memory */
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, mem_size));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, mem_size));
        
        /* Copy data to device */
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, image->p[i], mem_size, cudaMemcpyHostToDevice));
        
        /* Perform iterations until convergence */
        int n_iter = 0;
        int end = 0;
        
        do {
            n_iter++;
            
            /* Configure kernel launch parameters */
            dim3 blockSize(16, 16);
            dim3 gridSize((width - 2 * size + blockSize.x - 1) / blockSize.x, 
                         (height - 2 * size + blockSize.y - 1) / blockSize.y);
            
            /* Launch kernel */
            blurKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height, size);
            CHECK_CUDA_ERROR(cudaGetLastError());
            
            /* Check convergence on CPU */
            pixel *h_output = (pixel *)malloc(mem_size);
            CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, mem_size, cudaMemcpyDeviceToHost));
            
            end = 1;
            for (int j = 1; j < height - 1 && end; j++) {
                for (int k = 1; k < width - 1 && end; k++) {
                    int idx = j * width + k;
                    int orig_idx = idx;
                    
                    float diff_r = (h_output[idx].r - image->p[i][orig_idx].r);
                    float diff_g = (h_output[idx].g - image->p[i][orig_idx].g);
                    float diff_b = (h_output[idx].b - image->p[i][orig_idx].b);
                    
                    if (diff_r > threshold || -diff_r > threshold || 
                        diff_g > threshold || -diff_g > threshold || 
                        diff_b > threshold || -diff_b > threshold) {
                        end = 0;
                    }
                }
            }
            
            /* Swap input and output */
            pixel *temp = d_input;
            d_input = d_output;
            d_output = temp;
            
            /* Copy current result to host for next iteration check */
            CHECK_CUDA_ERROR(cudaMemcpy(image->p[i], d_input, mem_size, cudaMemcpyDeviceToHost));
            
            free(h_output);
        } while (threshold > 0 && !end);
        
        /* Free device memory */
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }
}

/* CUDA implementation of Sobel filter */
void apply_sobel_filter_cuda(animated_gif *image) {
    int i;
    
    for (i = 0; i < image->n_images; i++) {
        int width = image->width[i];
        int height = image->height[i];
        int num_pixels = width * height;
        size_t mem_size = num_pixels * sizeof(pixel);
        
        pixel *d_input, *d_output;
        
        /* Allocate device memory */
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_input, mem_size));
        CHECK_CUDA_ERROR(cudaMalloc((void **)&d_output, mem_size));
        
        /* Copy data to device */
        CHECK_CUDA_ERROR(cudaMemcpy(d_input, image->p[i], mem_size, cudaMemcpyHostToDevice));
        
        /* Configure kernel launch parameters */
        dim3 blockSize(16, 16);
        dim3 gridSize((width - 2 + blockSize.x - 1) / blockSize.x, 
                     (height - 2 + blockSize.y - 1) / blockSize.y);
        
        /* Initialize output to zeros (for border pixels) */
        CHECK_CUDA_ERROR(cudaMemset(d_output, 0, mem_size));
        
        /* Launch kernel */
        sobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
        CHECK_CUDA_ERROR(cudaGetLastError());
        
        /* Copy results back to host */
        CHECK_CUDA_ERROR(cudaMemcpy(image->p[i], d_output, mem_size, cudaMemcpyDeviceToHost));
        
        /* Free device memory */
        CHECK_CUDA_ERROR(cudaFree(d_input));
        CHECK_CUDA_ERROR(cudaFree(d_output));
    }
}

/* Hybrid CPU-GPU implementation for main function */
int main(int argc, char **argv) {
    char *input_filename;
    char *output_filename;
    animated_gif *image;
    struct timeval t1, t2;
    double duration;
    
    /* Check command-line arguments */
    if (argc < 3) {
        fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        return 1;
    }
    
    input_filename = argv[1];
    output_filename = argv[2];
    
    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);
    
    /* Load file and store the pixels in array */
    image = load_pixels(input_filename);
    if (image == NULL) {
        return 1;
    }
    
    /* IMPORT Timer stop */
    gettimeofday(&t2, NULL);
    
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    
    printf("GIF loaded from file %s with %d image(s) in %lf s\n",
            input_filename, image->n_images, duration);
    
    /* Initialize CUDA */
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA devices found, falling back to CPU implementation\n");
        apply_gray_filter(image);
        apply_blur_filter(image, 5, 20);
        apply_sobel_filter(image);
    } else {
        /* FILTER Timer start */
        gettimeofday(&t1, NULL);
        
        /* Get CUDA device properties */
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        printf("Using CUDA device: %s\n", deviceProp.name);
        
        /* Apply filters on GPU */
        apply_gray_filter_cuda(image);
        apply_blur_filter_cuda(image, 5, 20);
        apply_sobel_filter_cuda(image);
        
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);
        
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        
        printf("SOBEL done in %lf s (CUDA)\n", duration);
    }
    
    /* EXPORT Timer start */
    gettimeofday(&t1, NULL);
    
    /* Store file from array of pixels to GIF file */
    if (!store_pixels(output_filename, image)) {
        return 1;
    }
    
    /* EXPORT Timer stop */
    gettimeofday(&t2, NULL);
    
    duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
    
    printf("Export done in %lf s in file %s\n", duration, output_filename);
    
    return 0;
}
