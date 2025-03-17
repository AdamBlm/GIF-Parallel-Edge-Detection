/*
 * INF560 - Image Filtering Project (Hybrid MPI+CUDA)
 *
 * This code loads a GIF (on rank 0), distributes frames among MPI processes,
 * and each process applies three filters (grayscale, blur, and Sobel) on its
 * assigned frames using CUDA kernels. Finally, rank 0 gathers the processed
 * frames and writes out the final GIF.
 *
 * Compile (example):
 *    mpicc -o sobelf_mpi_cuda sobelf_mpi_cuda.c -lm -lgif -Xcompiler -fopenmp -L/usr/local/cuda/lib64 -lcudart
 *    (nvcc can also be used to compile CUDA portions.)
 *
 * Run with, for example:
 *    mpirun -np 4 ./sobelf_mpi_cuda input.gif output.gif
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "gif_lib.h"
#include <cuda_runtime.h>

// Set to 1 to enable debug prints.
#ifndef SOBELF_DEBUG
#define SOBELF_DEBUG 1
#endif

// -------------------- Data Structures --------------------
typedef struct pixel {
    int r;
    int g;
    int b;
} pixel;

typedef struct animated_gif {
    int n_images;      // number of frames
    int *width;        // width per frame
    int *height;       // height per frame
    pixel **p;         // pointer to frame pixel arrays (row-major)
    GifFileType *g;    // internal representation (do not modify)
} animated_gif;

// -------------------- GIF I/O Functions --------------------
animated_gif * load_pixels(char * filename) {
    GifFileType * g;
    ColorMapObject * colmap;
    int error;
    int n_images;
    int *width, *height;
    pixel **p;
    int i, j;
    animated_gif *image;

    g = DGifOpenFileName(filename, &error);
    if (g == NULL) {
        fprintf(stderr, "Error DGifOpenFileName %s\n", filename);
        return NULL;
    }
    error = DGifSlurp(g);
    if (error != GIF_OK) {
        fprintf(stderr, "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
        return NULL;
    }
    n_images = g->ImageCount;
    width = (int *)malloc(n_images * sizeof(int));
    height = (int *)malloc(n_images * sizeof(int));
    p = (pixel **)malloc(n_images * sizeof(pixel *));
    for (i = 0; i < n_images; i++) {
        width[i] = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;
        p[i] = (pixel *)malloc(width[i] * height[i] * sizeof(pixel));
    }
    colmap = g->SColorMap;
    if (colmap == NULL) {
        fprintf(stderr, "Error: global colormap is NULL\n");
        return NULL;
    }
    for (i = 0; i < n_images; i++) {
        for (j = 0; j < width[i]*height[i]; j++) {
            int c = g->SavedImages[i].RasterBits[j];
            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }
    image = (animated_gif *)malloc(sizeof(animated_gif));
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;
#if SOBELF_DEBUG
    printf("DEBUG: load_pixels: Loaded %d image(s); first image: %d x %d\n", n_images, width[0], height[0]);
#endif
    return image;
}

int output_modified_read_gif(char * filename, GifFileType * g) {
    GifFileType * g2;
    int error2;
#if SOBELF_DEBUG
    printf("DEBUG: Writing output to %s\n", filename);
#endif
    g2 = EGifOpenFileName(filename, false, &error2);
    if (g2 == NULL) {
        fprintf(stderr, "Error EGifOpenFileName %s\n", filename);
        return 0;
    }
    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;
    error2 = EGifSpew(g2);
    if (error2 != GIF_OK) {
        fprintf(stderr, "Error after writing g2: %d <%s>\n", error2, GifErrorString(g2->Error));
        return 0;
    }
    return 1;
}
int store_pixels(char * filename, animated_gif * image) {
    return output_modified_read_gif(filename, image->g);
}

// -------------------- CUDA Kernels --------------------
__global__ void grayscaleKernel(pixel *d_pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height) {
        int idx = y * width + x;
        int gray = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        d_pixels[idx].r = gray;
        d_pixels[idx].g = gray;
        d_pixels[idx].b = gray;
    }
}

__global__ void blurKernel(pixel *d_in, pixel *d_out, int width, int height) {
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
    // Load corners:
    if(tx == 0 && ty == 0 && x > 0 && y > 0)
        s_data[0] = d_in[(y - 1) * width + (x - 1)];
    if(tx == blockDim.x - 1 && ty == 0 && x < width - 1 && y > 0)
        s_data[0 * tileWidth + sm_x + 1] = d_in[(y - 1) * width + (x + 1)];
    if(tx == 0 && ty == blockDim.y - 1 && x > 0 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + 0] = d_in[(y + 1) * width + (x - 1)];
    if(tx == blockDim.x - 1 && ty == blockDim.y - 1 && x < width - 1 && y < height - 1)
        s_data[(sm_y + 1) * tileWidth + sm_x + 1] = d_in[(y + 1) * width + (x + 1)];
    __syncthreads();
    if(x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int sum_r = 0, sum_g = 0, sum_b = 0;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int idx = (sm_y + i) * tileWidth + (sm_x + j);
                sum_r += s_data[idx].r;
                sum_g += s_data[idx].g;
                sum_b += s_data[idx].b;
            }
        }
        int idx = y * width + x;
        d_out[idx].r = sum_r / 9;
        d_out[idx].g = sum_g / 9;
        d_out[idx].b = sum_b / 9;
    }
}

__global__ void sobelKernel(pixel *d_in, pixel *d_out, int width, int height) {
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
    if(x > 0 && x < width - 1 && y > 0 && y < height - 1) {
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

// -------------------- Hybrid MPI+CUDA Main --------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *input_filename, *output_filename;
    animated_gif *image = NULL;
    int n_images = 0;
    struct timeval t1, t2;
    double duration;

    // Rank 0 loads the entire GIF.
    if (rank == 0) {
        if (argc < 3) {
            fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        input_filename = argv[1];
        output_filename = argv[2];
        gettimeofday(&t1, NULL);
        image = load_pixels(input_filename);
        if (!image) {
            fprintf(stderr, "Error loading GIF\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        n_images = image->n_images;
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
        printf("Rank 0: GIF loaded from file %s with %d image(s) in %lf s\n", 
               input_filename, image->n_images, duration);
    }

    // Broadcast number of images to all processes.
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each process gets a subset of frames (simple frame distribution) */
    int frames_per_proc = n_images / size;
    int remainder = n_images % size;
    int start = rank * frames_per_proc + (rank < remainder ? rank : remainder);
    int count = frames_per_proc + (rank < remainder ? 1 : 0);

#if SOBELF_DEBUG
    printf("Rank %d: Assigned %d frame(s) starting at frame %d\n", rank, count, start);
#endif

    // Allocate local image structure for each process.
    animated_gif local_img;
    local_img.n_images = count;
    local_img.width = (int *)malloc(count * sizeof(int));
    local_img.height = (int *)malloc(count * sizeof(int));
    local_img.p = (pixel **)malloc(count * sizeof(pixel *));
    local_img.g = NULL;

    // Distribute frames from rank 0.
    if (rank == 0) {
        // Rank 0 copies its own frames.
        for (int i = start, j = 0; i < start + count; i++, j++) {
            local_img.width[j] = image->width[i];
            local_img.height[j] = image->height[i];
            int npixels = local_img.width[j] * local_img.height[j];
            local_img.p[j] = (pixel *)malloc(npixels * sizeof(pixel));
            memcpy(local_img.p[j], image->p[i], npixels * sizeof(pixel));
        }
        // Send frames to other ranks.
        for (int r = 1; r < size; r++) {
            int r_start = r * frames_per_proc + (r < remainder ? r : remainder);
            int r_count = frames_per_proc + (r < remainder ? 1 : 0);
            for (int i = r_start; i < r_start + r_count; i++) {
                MPI_Send(&image->width[i], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                MPI_Send(&image->height[i], 1, MPI_INT, r, 0, MPI_COMM_WORLD);
                int npixels = image->width[i] * image->height[i];
                MPI_Send(image->p[i], npixels * sizeof(pixel), MPI_BYTE, r, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (int j = 0; j < count; j++) {
            MPI_Recv(&local_img.width[j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&local_img.height[j], 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int npixels = local_img.width[j] * local_img.height[j];
            local_img.p[j] = (pixel *)malloc(npixels * sizeof(pixel));
            MPI_Recv(local_img.p[j], npixels * sizeof(pixel), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Set each process to use its GPU (assume one GPU per MPI rank).
    cudaSetDevice(rank); // Adjust this if your mapping is different.

    /* ---- Process Each Frame Using CUDA Kernels ---- */
    for (int j = 0; j < count; j++) {
        int w = local_img.width[j];
        int h = local_img.height[j];
        int npixels = w * h;
        size_t sizeBytes = npixels * sizeof(pixel);
#if SOBELF_DEBUG
        printf("Rank %d: Processing frame %d (dimensions: %d x %d)\n", rank, j, w, h);
#endif
        pixel *d_in, *d_out;
        cudaError_t err;
        err = cudaMalloc((void**)&d_in, sizeBytes);
        if(err != cudaSuccess) {
            fprintf(stderr, "Rank %d: ERROR: cudaMalloc d_in: %s\n", rank, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        err = cudaMalloc((void**)&d_out, sizeBytes);
        if(err != cudaSuccess) {
            fprintf(stderr, "Rank %d: ERROR: cudaMalloc d_out: %s\n", rank, cudaGetErrorString(err));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        err = cudaMemcpy(d_in, local_img.p[j], sizeBytes, cudaMemcpyHostToDevice);
        if(err != cudaSuccess)
            fprintf(stderr, "Rank %d: ERROR: cudaMemcpy to d_in: %s\n", rank, cudaGetErrorString(err));

        // Define CUDA grid and block dimensions.
        dim3 blockDim(16, 16);
        dim3 gridDim((w + blockDim.x - 1) / blockDim.x, (h + blockDim.y - 1) / blockDim.y);
        size_t sharedMemSize = (blockDim.x + 2) * (blockDim.y + 2) * sizeof(pixel);

        // Launch grayscale kernel.
        grayscaleKernel<<<gridDim, blockDim>>>(d_in, w, h);
        cudaDeviceSynchronize();

        // Launch blur kernel.
        blurKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, w, h);
        cudaDeviceSynchronize();

        // Copy blur result back to d_in.
        cudaMemcpy(d_in, d_out, sizeBytes, cudaMemcpyDeviceToDevice);

        // Launch sobel kernel.
        sobelKernel<<<gridDim, blockDim, sharedMemSize>>>(d_in, d_out, w, h);
        cudaDeviceSynchronize();

        // Copy final result back to host.
        cudaMemcpy(local_img.p[j], d_out, sizeBytes, cudaMemcpyDeviceToHost);

        cudaFree(d_in);
        cudaFree(d_out);
#if SOBELF_DEBUG
        // Print a sample pixel value from the processed frame.
        printf("Rank %d: Processed frame %d, sample pixel at (0,0): (%d,%d,%d)\n",
               rank, j, local_img.p[j][0].r, local_img.p[j][0].g, local_img.p[j][0].b);
#endif
    }

    // Gather processed frames back to rank 0.
    if (rank == 0) {
        for (int r = 1; r < size; r++) {
            int r_start = r * frames_per_proc + (r < remainder ? r : remainder);
            int r_count = frames_per_proc + (r < remainder ? 1 : 0);
            for (int i = r_start, j = 0; i < r_start + r_count; i++, j++) {
                int w, h;
                MPI_Recv(&w, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&h, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                int npixels = w * h;
                MPI_Recv(image->p[i], npixels * sizeof(pixel), MPI_BYTE, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
    } else {
        for (int j = 0; j < count; j++) {
            MPI_Send(&local_img.width[j], 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(&local_img.height[j], 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            int npixels = local_img.width[j] * local_img.height[j];
            MPI_Send(local_img.p[j], npixels * sizeof(pixel), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
        }
    }

    // Free local memory.
    for (int j = 0; j < count; j++) {
        free(local_img.p[j]);
    }
    free(local_img.p);
    free(local_img.width);
    free(local_img.height);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t1, NULL);
        store_pixels(output_filename, image);
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
        printf("Rank 0: Export done in %lf s in file %s\n", duration, output_filename);
        // (Free image memory as needed)
    }

    MPI_Finalize();
    return 0;
}
