#include "cuda_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


__global__ void grayscale_kernel(pixel *d_pixels, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int moy = (d_pixels[idx].r + d_pixels[idx].g + d_pixels[idx].b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;
        d_pixels[idx].r = moy;
        d_pixels[idx].g = moy;
        d_pixels[idx].b = moy;
    }
}

__global__ void blur_kernel(pixel *d_in, pixel *d_out, int width, int height, int size) {
    extern __shared__ pixel sharedMem[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int x = bx + tx, y = by + ty;
    int sharedWidth = blockDim.x + 2 * size;
    int sharedHeight = blockDim.y + 2 * size;
    for (int j = ty; j < sharedHeight; j += blockDim.y) {
        int y_global = by + j - size;
        for (int i = tx; i < sharedWidth; i += blockDim.x) {
            int x_global = bx + i - size;
            int valid_x = max(0, min(width - 1, x_global));
            int valid_y = max(0, min(height - 1, y_global));
            sharedMem[j * sharedWidth + i] = d_in[valid_y * width + valid_x];
        }
    }
    __syncthreads();
    if (x < width && y < height) {
        int t_r = 0, t_g = 0, t_b = 0;
        int count = 0;
        for (int j = -size; j <= size; j++) {
            for (int i = -size; i <= size; i++) {
                int sx = tx + i + size;
                int sy = ty + j + size;
                t_r += sharedMem[sy * sharedWidth + sx].r;
                t_g += sharedMem[sy * sharedWidth + sx].g;
                t_b += sharedMem[sy * sharedWidth + sx].b;
                count++;
            }
        }
        int idx = y * width + x;
        d_out[idx].r = t_r / count;
        d_out[idx].g = t_g / count;
        d_out[idx].b = t_b / count;
    }
}

__global__ void check_convergence_kernel(pixel *d_prev, pixel *d_curr, int width, int height, int threshold, int *d_continue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < width && y < height) {
        int idx = y * width + x;
        int diff_r = abs(d_curr[idx].r - d_prev[idx].r);
        int diff_g = abs(d_curr[idx].g - d_prev[idx].g);
        int diff_b = abs(d_curr[idx].b - d_prev[idx].b);
        if (diff_r > threshold || diff_g > threshold || diff_b > threshold)
            *d_continue = 1;
    }
}

__global__ void sobel_kernel(pixel *d_in, pixel *d_out, int width, int height) {
    extern __shared__ pixel sharedMem[];
    int tx = threadIdx.x, ty = threadIdx.y;
    int bx = blockIdx.x * blockDim.x, by = blockIdx.y * blockDim.y;
    int x = bx + tx, y = by + ty;
    int sharedWidth = blockDim.x + 2;
    int sharedHeight = blockDim.y + 2;
    for (int j = ty; j < sharedHeight; j += blockDim.y) {
        int y_global = by + j - 1;
        for (int i = tx; i < sharedWidth; i += blockDim.x) {
            int x_global = bx + i - 1;
            int valid_x = max(0, min(width - 1, x_global));
            int valid_y = max(0, min(height - 1, y_global));
            sharedMem[j * sharedWidth + i] = d_in[valid_y * width + valid_x];
        }
    }
    __syncthreads();
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        int idx = y * width + x;
        int p00 = sharedMem[(ty) * sharedWidth + (tx)].r;
        int p02 = sharedMem[(ty) * sharedWidth + (tx+2)].r;
        int p10 = sharedMem[(ty+1) * sharedWidth + (tx)].r;
        int p12 = sharedMem[(ty+1) * sharedWidth + (tx+2)].r;
        int p20 = sharedMem[(ty+2) * sharedWidth + (tx)].r;
        int p22 = sharedMem[(ty+2) * sharedWidth + (tx+2)].r;
        float Gx = -p00 + p02 - 2*p10 + 2*p12 - p20 + p22;
        int p00_val = sharedMem[(ty) * sharedWidth + (tx)].r;
        int p01_val = sharedMem[(ty) * sharedWidth + (tx+1)].r;
        int p02_val = sharedMem[(ty) * sharedWidth + (tx+2)].r;
        int p20_val = sharedMem[(ty+2) * sharedWidth + (tx)].r;
        int p21_val = sharedMem[(ty+2) * sharedWidth + (tx+1)].r;
        int p22_val = sharedMem[(ty+2) * sharedWidth + (tx+2)].r;
        float Gy = -p00_val - 2*p01_val - p02_val + p20_val + 2*p21_val + p22_val;
        float val = sqrtf(Gx * Gx + Gy * Gy) / 4.0f;
        if (val > 15)
        {
            d_out[idx].r = 255;
            d_out[idx].g = 255;
            d_out[idx].b = 255;
        } else {
            d_out[idx].r = 0;
            d_out[idx].g = 0;
            d_out[idx].b = 0;
        }
    } else if (x < width && y < height) {
        d_out[y * width + x] = d_in[y * width + x];
    }
}


scatter_info* create_scatter_info(int n_images, int size) {
    scatter_info* info = (scatter_info*)malloc(sizeof(scatter_info));
    if (!info) return NULL;
    info->sendcounts = (int*)calloc(size, sizeof(int));
    info->displs = (int*)calloc(size, sizeof(int));
    info->image_counts = (int*)calloc(size, sizeof(int));
    info->image_displs = (int*)calloc(size, sizeof(int));
    info->scatter_byte_counts = (int*)calloc(size, sizeof(int));
    info->scatter_byte_displs = (int*)calloc(size, sizeof(int));
    if (!info->sendcounts || !info->displs || !info->image_counts || !info->image_displs ||
        !info->scatter_byte_counts || !info->scatter_byte_displs) {
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
        return NULL;
    }
    int base_count = n_images / size;
    int remainder = n_images % size;
    for (int i = 0; i < size; i++) {
        info->image_counts[i] = base_count + (i < remainder ? 1 : 0);
        if (i > 0)
            info->image_displs[i] = info->image_displs[i-1] + info->image_counts[i-1];
    }
    return info;
}

void free_scatter_info(scatter_info *info) {
    if (info) {
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
    }
}

void calculate_pixel_counts(scatter_info* scatter_data, int* widths, int* heights, int size) {
    if (!scatter_data || !widths || !heights) return;
    for (int i = 0; i < size; i++) {
        int start_img = scatter_data->image_displs[i];
        int end_img = start_img + scatter_data->image_counts[i];
        scatter_data->sendcounts[i] = 0;
        for (int j = start_img; j < end_img; j++) {
            scatter_data->sendcounts[i] += widths[j] * heights[j];
        }
        if (i > 0)
            scatter_data->displs[i] = scatter_data->displs[i-1] + scatter_data->sendcounts[i-1];
        scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
        scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
    }
}


void free_resources(pixel **p, int n_images) {
    if (p) {
        for (int i = 0; i < n_images; i++) {
            if (p[i]) free(p[i]);
        }
        free(p);
    }
}
