/*
 * INF560 - Image Filtering Project (MPI Domain Decomposition version – Single Pass Filters)
 *
 * This code loads a GIF and for each image (frame) it:
 *   1. Domain decomposes the image by rows among MPI processes.
 *   2. Exchanges ghost cells (one extra row at the top and/or bottom when available).
 *   3. Applies one pass of filters (grayscale, blur, Sobel) on the local real region.
 *      The blur and Sobel filters process only pixels having a full 3x3 neighborhood.
 *   4. Gathers the processed pieces and writes out the final GIF.
 *
 * Compile with:
 *    mpicc -o sobelf_mpi_domain sobelf_mpi_domain.c -lm -lgif -fopenmp
 *
 * Run with, for example:
 *    mpirun -np 4 ./sobelf_mpi_domain input.gif output.gif
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "gif_lib.h"

#define SOBELF_DEBUG 0

typedef struct pixel {
    int r;
    int g;
    int b;
} pixel;

typedef struct animated_gif {
    int n_images;
    int *width;
    int *height;
    pixel **p;  // Each image as a flat row-major array
    GifFileType *g; // Do not modify
} animated_gif;

animated_gif * load_pixels(char * filename) {
    GifFileType * g;
    ColorMapObject * colmap;
    int error;
    int n_images;
    int *width;
    int *height;
    pixel **p;
    int i, j;
    animated_gif * image;

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
    return image;
}

int output_modified_read_gif(char * filename, GifFileType * g) {
    GifFileType * g2;
    int error2;
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

/* ===== Local Filter Functions =====
   Assume the local block is a flat array with dimensions:
       local_alloc_rows x width,
   where local_alloc_rows = local_real_rows + ghost_top + ghost_bottom.
   The "real" region is from row ghost_top to ghost_top+local_real_rows-1.
   For the convolution filters, we update only pixels that have a full 3x3 neighborhood.
*/

// Grayscale: update every pixel in the real region.
void apply_gray_filter_local(pixel *block, int ghost_top, int local_real_rows, int width) {
    int i, j;
    for (i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (j = 0; j < width; j++) {
            int idx = i * width + j;
            int moy = (block[idx].r + block[idx].g + block[idx].b) / 3;
            if (moy < 0) moy = 0;
            if (moy > 255) moy = 255;
            block[idx].r = moy;
            block[idx].g = moy;
            block[idx].b = moy;
        }
    }
}

// Single-pass blur: process only rows with a full 3x3 neighborhood.
void apply_blur_filter_local_once(pixel *block, int ghost_top, int local_real_rows, int width, int size) {
    int start_row = ghost_top + size; 
    int end_row = ghost_top + local_real_rows - size;
    pixel *newBlock = (pixel *)malloc((local_real_rows + ghost_top) * width * sizeof(pixel));
    if (!newBlock) return;
    // Copy ghost rows unchanged
    for (int i = 0; i < ghost_top; i++) {
        memcpy(newBlock + i * width, block + i * width, width * sizeof(pixel));
    }
    // Process real region: avoid first and last column
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < width - 1; j++) {
            int sum_r = 0, sum_g = 0, sum_b = 0, count = 0;
            for (int di = -size; di <= size; di++) {
                for (int dj = -size; dj <= size; dj++) {
                    int idx = (i + di) * width + (j + dj);
                    sum_r += block[idx].r;
                    sum_g += block[idx].g;
                    sum_b += block[idx].b;
                    count++;
                }
            }
            int idx = i * width + j;
            newBlock[idx].r = sum_r / count;
            newBlock[idx].g = sum_g / count;
            newBlock[idx].b = sum_b / count;
        }
    }
    // Copy the processed region back
    for (int i = start_row; i < end_row; i++) {
        memcpy(block + i * width, newBlock + i * width, width * sizeof(pixel));
    }
    free(newBlock);
}

// Single-pass Sobel: process only rows with a full 3x3 neighborhood.
void apply_sobel_filter_local_once(pixel *block, int ghost_top, int local_real_rows, int width) {
    int start_row = ghost_top + 1;
    int end_row = ghost_top + local_real_rows - 1;
    pixel *result = (pixel *)malloc((local_real_rows) * width * sizeof(pixel));
    if (!result) return;
    // Initialize result to zero.
    memset(result, 0, (local_real_rows) * width * sizeof(pixel));
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < width - 1; j++) {
            int b_no = block[(i - 1)*width + (j - 1)].b;
            int b_n  = block[(i - 1)*width + j].b;
            int b_ne = block[(i - 1)*width + (j + 1)].b;
            int b_o  = block[i*width + (j - 1)].b;
            int b_e  = block[i*width + (j + 1)].b;
            int b_so = block[(i + 1)*width + (j - 1)].b;
            int b_s  = block[(i + 1)*width + j].b;
            int b_se = block[(i + 1)*width + (j + 1)].b;
            float deltaX = -b_no + b_ne - 2 * b_o + 2 * b_e - b_so + b_se;
            float deltaY = b_se + 2 * b_s + b_so - b_ne - 2 * b_n - b_no;
            float mag = sqrt(deltaX * deltaX + deltaY * deltaY) / 4.0;
            int idx = (i - ghost_top) * width + j;
            if (mag > 50)
                result[idx].r = result[idx].g = result[idx].b = 255;
            else
                result[idx].r = result[idx].g = result[idx].b = 0;
        }
    }
    for (int i = start_row; i < end_row; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx = i * width + j;
            int ridx = (i - ghost_top) * width + j;
            block[idx].r = result[ridx].r;
            block[idx].g = result[ridx].g;
            block[idx].b = result[ridx].b;
        }
    }
    free(result);
}

/* ===================== Main ===================== */
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
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
               input_filename, image->n_images, duration);
    }
    
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Process each image using domain decomposition */
    for (int img = 0; img < n_images; img++) {
        int w, h;
        if (rank == 0) {
            w = image->width[img];
            h = image->height[img];
        }
        MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        /* Domain decomposition by rows */
        int rows_per_proc = h / size;
        int remainder = h % size;
        int start = rank * rows_per_proc + (rank < remainder ? rank : remainder);
        int local_real_rows = rows_per_proc + (rank < remainder ? 1 : 0);
        /* Ghost rows: if not first process, ghost_top = 1; if not last, ghost_bottom = 1 */
        int ghost_top = (rank == 0) ? 0 : 1;
        int ghost_bottom = (rank == size - 1) ? 0 : 1;
        int local_alloc_rows = local_real_rows + ghost_top + ghost_bottom;
        
        /* Allocate local block */
        pixel *local_block = (pixel *)malloc(local_alloc_rows * w * sizeof(pixel));
        if (!local_block) {
            fprintf(stderr, "Rank %d: Unable to allocate local block\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        memset(local_block, 0, local_alloc_rows * w * sizeof(pixel));
        
        /* Distribute image rows */
        if (rank == 0) {
            memcpy(local_block + ghost_top * w, image->p[img], local_real_rows * w * sizeof(pixel));
            for (int r = 1; r < size; r++) {
                int r_start = r * rows_per_proc + (r < remainder ? r : remainder);
                int r_rows = rows_per_proc + (r < remainder ? 1 : 0);
                MPI_Send(image->p[img] + r_start * w, r_rows * w * sizeof(pixel), MPI_BYTE, r, 0, MPI_COMM_WORLD);
            }
        } else {
            MPI_Recv(local_block + ghost_top * w, local_real_rows * w * sizeof(pixel), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        /* Exchange ghost rows */
        if (rank > 0) {
            // Send first real row to rank-1; receive ghost row into row 0.
            MPI_Sendrecv(local_block + ghost_top * w, w * sizeof(pixel), MPI_BYTE,
                         rank - 1, 1,
                         local_block, w * sizeof(pixel), MPI_BYTE,
                         rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        if (rank < size - 1) {
            // Send last real row to rank+1; receive ghost row into row ghost_top+local_real_rows.
            MPI_Sendrecv(local_block + (ghost_top + local_real_rows - 1) * w, w * sizeof(pixel), MPI_BYTE,
                         rank + 1, 0,
                         local_block + (ghost_top + local_real_rows) * w, w * sizeof(pixel), MPI_BYTE,
                         rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        
        /* Apply filters on the real region */
        apply_gray_filter_local(local_block, ghost_top, local_real_rows, w);
        apply_blur_filter_local_once(local_block, ghost_top, local_real_rows, w, 1);
        apply_sobel_filter_local_once(local_block, ghost_top, local_real_rows, w);
        
        /* Gather processed real rows back to rank 0 */
        if (rank == 0) {
            memcpy(image->p[img], local_block + ghost_top * w, local_real_rows * w * sizeof(pixel));
            for (int r = 1; r < size; r++) {
                int r_start = r * rows_per_proc + (r < remainder ? r : remainder);
                int r_rows = rows_per_proc + (r < remainder ? 1 : 0);
                MPI_Recv(image->p[img] + r_start * w, r_rows * w * sizeof(pixel), MPI_BYTE, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        } else {
            MPI_Send(local_block + ghost_top * w, local_real_rows * w * sizeof(pixel), MPI_BYTE, 0, 3, MPI_COMM_WORLD);
        }
        
        free(local_block);
    } // End for each image
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        gettimeofday(&t1, NULL);
        store_pixels(output_filename, image);
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
        printf("Export done in %lf s in file %s\n", duration, output_filename);
        // Free image memory as needed...
    }
    
    MPI_Finalize();
    return 0;
}
