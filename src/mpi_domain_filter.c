#include "mpi_domain_filter.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "gif_lib.h"
#include "gif_io.h"

#ifndef SOBELF_DEBUG
#define SOBELF_DEBUG 0
#endif



/* Convert each pixel to grayscale in [ghost_top..ghost_top+local_real_rows-1]. */
void apply_gray_filter_local(pixel *block, int ghost_top, int local_real_rows, int width)
{
    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            int moy = (block[idx].r + block[idx].g + block[idx].b) / 3;
            if (moy < 0)   moy = 0;
            if (moy > 255) moy = 255;
            block[idx].r = moy;
            block[idx].g = moy;
            block[idx].b = moy;
        }
    }
}

/* Single-pass blur with a 3×3 kernel (size=1). */
void apply_blur_filter_local_once(pixel *block, int ghost_top, int local_real_rows, int width, int size)
{
    int total_rows = ghost_top + local_real_rows;
    pixel *newBlock = (pixel*) malloc(total_rows * width * sizeof(pixel));
    if (!newBlock) return;

    memcpy(newBlock, block, total_rows * width * sizeof(pixel));

    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
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

    memcpy(block, newBlock, total_rows * width * sizeof(pixel));
    free(newBlock);
}

/* Single-pass Sobel with a 3×3 neighborhood. */
void apply_sobel_filter_local_once(pixel *block, int ghost_top, int local_real_rows, int width)
{
    int total_rows = ghost_top + local_real_rows;
    pixel *result = (pixel*) malloc(local_real_rows * width * sizeof(pixel));
    if (!result) return;
    memset(result, 0, local_real_rows * width * sizeof(pixel));

    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (int j = 1; j < width - 1; j++) {
            int b_no = block[(i-1)*width + (j-1)].b;
            int b_n  = block[(i-1)*width + j    ].b;
            int b_ne = block[(i-1)*width + (j+1)].b;
            int b_o  = block[i*width + (j-1)].b;
            int b_e  = block[i*width + (j+1)].b;
            int b_so = block[(i+1)*width + (j-1)].b;
            int b_s  = block[(i+1)*width + j    ].b;
            int b_se = block[(i+1)*width + (j+1)].b;

            float deltaX = -b_no + b_ne - 2*b_o + 2*b_e - b_so + b_se;
            float deltaY =  b_se + 2*b_s + b_so - b_ne - 2*b_n - b_no;
            float mag = sqrtf(deltaX*deltaX + deltaY*deltaY) / 4.0f;

            int ridx = (i - ghost_top) * width + j;
            if (mag > 50.0f)
                result[ridx].r = result[ridx].g = result[ridx].b = 255;
            else
                result[ridx].r = result[ridx].g = result[ridx].b = 0;
        }
    }

    for (int i = ghost_top; i < ghost_top + local_real_rows; i++) {
        for (int j = 1; j < width - 1; j++) {
            int idx  = i * width + j;
            int ridx = (i - ghost_top) * width + j;
            block[idx].r = result[ridx].r;
            block[idx].g = result[ridx].g;
            block[idx].b = result[ridx].b;
        }
    }
    free(result);
}

/* ------------------------------------------------------------------
 *                     Helper: Exchange Ghost Rows
 * ------------------------------------------------------------------ */
void exchange_ghost_rows(pixel *local_block, int ghost_top, int local_real_rows, int width, int rank, int size) {
    if (rank > 0) {
        MPI_Sendrecv(local_block + ghost_top * width,
                     width * sizeof(pixel), MPI_BYTE,
                     rank - 1, 200,
                     local_block,
                     width * sizeof(pixel), MPI_BYTE,
                     rank - 1, 300,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(local_block, local_block + ghost_top * width, width * sizeof(pixel));
    }
    if (rank < size - 1) {
        MPI_Sendrecv(local_block + (ghost_top + local_real_rows - 1) * width,
                     width * sizeof(pixel), MPI_BYTE,
                     rank + 1, 300,
                     local_block + (ghost_top + local_real_rows) * width,
                     width * sizeof(pixel), MPI_BYTE,
                     rank + 1, 200,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
        memcpy(local_block + (ghost_top + local_real_rows) * width,
               local_block + (ghost_top + local_real_rows - 1) * width,
               width * sizeof(pixel));
    }
}

int run_mpi_domain_filter(char *input_filename, char *output_filename)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double load_time = 0.0;
    double filter_time = 0.0;
    double export_time = 0.0;
    animated_gif *image = NULL;
    int n_images = 0;

    struct timeval t1, t2;
    if (rank == 0) {
        gettimeofday(&t1, NULL);
        image = load_pixels(input_filename);
        if (!image) {
            fprintf(stderr, "Error loading GIF %s\n", input_filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        n_images = image->n_images;
        gettimeofday(&t2, NULL);
        load_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1e6;
        printf("GIF loaded from file %s with %d image(s) in %.3f s\n",
               input_filename, n_images, load_time);
        fflush(stdout);
    }

    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

    struct timeval tf1, tf2;
    if (rank == 0) {
        gettimeofday(&tf1, NULL);
    }

    for (int img = 0; img < n_images; img++) {
        int w = 0, h = 0;
        if (rank == 0) {
            w = image->width[img];
            h = image->height[img];
        }
        MPI_Bcast(&w, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&h, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int rows_per_proc = h / size;
        int remainder = h % size;
        int global_start;
        if (rank < remainder) {
            global_start = rank * (rows_per_proc + 1);
        } else {
            global_start = remainder * (rows_per_proc + 1) + (rank - remainder) * rows_per_proc;
        }
        int local_real_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;

        int ghost_top = 1;
        int ghost_bottom = 1;
        int local_alloc_rows = local_real_rows + ghost_top + ghost_bottom;

        pixel *local_block = (pixel*) calloc(local_alloc_rows * w, sizeof(pixel));
        if (!local_block) {
            fprintf(stderr, "Rank %d: cannot allocate local_block\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }

        if (rank == 0) {
            memcpy(local_block + ghost_top * w, image->p[img], local_real_rows * w * sizeof(pixel));
            int offset = local_real_rows;
            for (int r = 1; r < size; r++) {
                int r_rows = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
                MPI_Send(image->p[img] + offset * w,
                         r_rows * w * sizeof(pixel),
                         MPI_BYTE,
                         r, 100,
                         MPI_COMM_WORLD);
                offset += r_rows;
            }
        } else {
            MPI_Recv(local_block + ghost_top * w,
                     local_real_rows * w * sizeof(pixel),
                     MPI_BYTE,
                     0, 100,
                     MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        }

        exchange_ghost_rows(local_block, ghost_top, local_real_rows, w, rank, size);
        apply_gray_filter_local(local_block, ghost_top, local_real_rows, w);
        exchange_ghost_rows(local_block, ghost_top, local_real_rows, w, rank, size);
        apply_blur_filter_local_once(local_block, ghost_top, local_real_rows, w, 1);
        exchange_ghost_rows(local_block, ghost_top, local_real_rows, w, rank, size);
        apply_sobel_filter_local_once(local_block, ghost_top, local_real_rows, w);

        if (rank == 0) {
            memcpy(image->p[img], local_block + ghost_top * w, local_real_rows * w * sizeof(pixel));
            int offset = local_real_rows;
            for (int r = 1; r < size; r++) {
                int r_rows = (r < remainder) ? (rows_per_proc + 1) : rows_per_proc;
                MPI_Recv(image->p[img] + offset * w,
                         r_rows * w * sizeof(pixel),
                         MPI_BYTE,
                         r, 999,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                offset += r_rows;
            }
        } else {
            MPI_Send(local_block + ghost_top * w,
                     local_real_rows * w * sizeof(pixel),
                     MPI_BYTE,
                     0, 999,
                     MPI_COMM_WORLD);
        }

        free(local_block);
    }

    if (rank == 0) {
        gettimeofday(&tf2, NULL);
        filter_time = (tf2.tv_sec - tf1.tv_sec) + (tf2.tv_usec - tf1.tv_usec) / 1e6;
        printf("SOBEL done in %.3f s\n", filter_time);
        fflush(stdout);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        struct timeval te1, te2;
        gettimeofday(&te1, NULL);
        store_pixels(output_filename, image);
        gettimeofday(&te2, NULL);
        export_time = (te2.tv_sec - te1.tv_sec) + (te2.tv_usec - te1.tv_usec) / 1e6;
        printf("Export done in %.3f s in file %s\n", export_time, output_filename);
        fflush(stdout);
    }

    return 0;
}
/* End of mpi_domain_filter.c */
