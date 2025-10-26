/*
 * Adaptive Image Filtering Project
 * Usage:
 *   adaptive_filter [options] input.gif output.gif [mode]
 *
 * Options:
 *   -h, --help   Show this help message and exit.
 *
 * Modes (optional, default is "auto"):
 *   auto         Automatically select based on image and hardware (default)
 *   cuda         Use CUDA only
 *   mpi          Use MPI only
 *   omp          Use OpenMP only
 *   mpi_omp      Use MPI + OpenMP
 *   cuda_mpi     Use CUDA + MPI
 *   cuda_omp_mpi Use CUDA + OpenMP + MPI
 *   seq          Use sequential processing
 *
 * Compile (for example):
 *   mpicc -o adaptive_filter adaptive_filter.c dgif_lib.c egif_lib.c gif_err.c gif_font.c gif_hash.c \
 *       gifalloc.c openbsd-reallocarray.c quantize.c -lm -fopenmp -lcuda -lstdc++
 *
 * Run with:
 *   ./adaptive_filter input.gif output.gif [mode]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "sequential_filter.h"
#include "mpi_domain_filter.h"
#include "omp_filter.h"
#include "cuda_filter.h"
#include "hybrid_openmp_mpi_filter.h"
#include "cuda_mpi_filter.h"
#include "cuda_omp_mpi_filter.h"
#include "gif_io.h"


void print_help(const char *progname) {
    printf("\nAdaptive Image Filtering Project\n");
    printf("Usage: %s [options] input.gif output.gif [mode]\n", progname);
    printf("\nOptions:\n");
    printf("  -h, --help        Show this help message and exit.\n");
    printf("\nModes (default: auto):\n");
    printf("  auto              Automatically select the best parallelization approach\n");
    printf("  cuda              Use CUDA only\n");
    printf("  mpi               Use MPI only\n");
    printf("  omp               Use OpenMP only\n");
    printf("  mpi_omp           Use MPI + OpenMP\n");
    printf("  cuda_mpi          Use CUDA + MPI\n");
    printf("  cuda_omp_mpi      Use CUDA + OpenMP + MPI\n");
    printf("  seq               Use sequential processing\n");
    printf("\nExample:\n");
    printf("  mpirun -np 4 %s input.gif output.gif auto\n\n", progname);
}

/**
 * Detect hardware resources: how many MPI ranks, how many OMP threads,
 * and how many CUDA devices.
 */
void detect_hardware_resources(int *mpi_ranks, int *omp_threads, int *cuda_gpus) {
    int flag;
    MPI_Initialized(&flag);
    if (flag) {
        MPI_Comm_size(MPI_COMM_WORLD, mpi_ranks);
    } else {
        *mpi_ranks = 1;
    }

    *omp_threads = omp_get_max_threads();

    cudaError_t cudaStatus = cudaGetDeviceCount(cuda_gpus);
    if (cudaStatus != cudaSuccess) {
        *cuda_gpus = 0;
    }
}

/**
 * Evaluate the input GIF by loading it briefly (just to read dimensions and frame count),
 * then freeing it.
 */
void evaluate_input(const char *filename, int *num_frames, double *avg_width, double *avg_height) {
    animated_gif *img = load_pixels((char*)filename);
    if (!img) {
        fprintf(stderr, "Error: Could not load input GIF '%s'\n", filename);
        exit(1);
    }
    *num_frames = img->n_images;

    long sum_w = 0, sum_h = 0;
    for (int i = 0; i < img->n_images; i++) {
        sum_w += img->width[i];
        sum_h += img->height[i];
    }
    *avg_width = (double)sum_w / img->n_images;
    *avg_height = (double)sum_h / img->n_images;

    // Free image resources
    for (int i = 0; i < img->n_images; i++) {
        free(img->p[i]);
    }
    free(img->p);
    free(img->width);
    free(img->height);
    free(img);
}


int main(int argc, char **argv) {
    // Check for help first
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        print_help(argv[0]);
        return 0;
    }

    // Initialize MPI with thread support
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "Error: Insufficient MPI thread support. Requested MPI_THREAD_FUNNELED, got %d\n", provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Check minimal args
    if (argc < 3) {
        fprintf(stderr, "Error: Missing required arguments.\n");
        print_help(argv[0]);
        MPI_Finalize();
        return 1;
    }

    char *input_filename  = argv[1];
    char *output_filename = argv[2];
    char *mode            = (argc > 3) ? argv[3] : "auto";

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        printf("\n------------------------------------------\n");
        printf("Adaptive Image Filtering - Starting...\n");
        printf("Input file: %s\n", input_filename);
        printf("Output file: %s\n", output_filename);
        printf("Selected mode: %s\n", mode);
        printf("------------------------------------------\n\n");
    }

    // Detect hardware
    int mpi_ranks = 1, omp_threads = 1, cuda_gpus = 0;
    detect_hardware_resources(&mpi_ranks, &omp_threads, &cuda_gpus);

    if (world_rank == 0) {
        if (mpi_ranks > 1) {
            printf("Hardware Resources Detected: %d MPI ranks, %d OpenMP threads, %d CUDA GPU(s)\n",
                   mpi_ranks, omp_threads, cuda_gpus);
        } else {
            printf("Hardware Resources Detected: single MPI rank, %d OpenMP threads, %d CUDA GPU(s)\n",
                   omp_threads, cuda_gpus);
        }
    }

    // Evaluate input
    int    num_frames = 0;
    double avg_w      = 0.0;
    double avg_h      = 0.0;
    evaluate_input(input_filename, &num_frames, &avg_w, &avg_h);

    if (world_rank == 0) {
        printf("Input GIF has %d frame(s) with average dimensions %.0f x %.0f\n\n",
               num_frames, avg_w, avg_h);
    }

    int ret = 0;

    // Auto logic now prefers CUDA-based hybrids if GPU is present and data is large:
    if (strcmp(mode, "auto") == 0) {
        int large_img = ((avg_w * avg_h) > 1000000); // 1 million pixel heuristic
        int multiple_frames = (num_frames > 1);

        if (cuda_gpus > 0) {
            // If we have at least one GPU:
            if (multiple_frames || large_img) {
                // If multiple frames or single large => prefer hybrid MPI+OpenMP+CUDA if ranks>1
                if (mpi_ranks > 1 && omp_threads > 1) {
                    printf("Auto-selected: CUDA+OpenMP+MPI (multiple frames or large) because GPU is available.\n");
                    ret = run_cuda_omp_mpi_filter(input_filename, output_filename, omp_threads);
                }
                else if (mpi_ranks > 1) {
                    printf("Auto-selected: CUDA+MPI approach (GPU available, multiple frames or large).\n");
                    ret = run_cuda_mpi_filter(input_filename, output_filename);
                }
                else {
                    // Single rank but GPU => just CUDA
                    printf("Auto-selected: CUDA approach (GPU available, single rank).\n");
                    ret = run_cuda_filter(input_filename, output_filename);
                }
            }
            else {
                
                printf("Auto-selected: CUDA approach (small or single frame but GPU is present).\n");
                ret = run_cuda_filter(input_filename, output_filename);
            }
        } 
        else {
            // If no GPU, fallback to CPU-based logic
            if (multiple_frames) {
                if (mpi_ranks > 1 && omp_threads > 1) {
                    printf("Auto-selected: MPI+OpenMP (multiple frames, no GPU).\n");
                    ret = run_hybrid_openmp_mpi_filter(input_filename, output_filename, omp_threads);
                }
                else if (mpi_ranks > 1) {
                    printf("Auto-selected: MPI domain decomposition (multiple frames, no GPU).\n");
                    ret = run_mpi_domain_filter(input_filename, output_filename);
                }
                else if (omp_threads > 1) {
                    printf("Auto-selected: OpenMP (multiple frames, single rank).\n");
                    ret = run_omp_filter(input_filename, output_filename, omp_threads);
                }
                else {
                    printf("Auto-selected: Sequential (no GPU, single rank, single thread).\n");
                    ret = run_sequential_filter(input_filename, output_filename);
                }
            }
            else {
                // Single frame, no GPU
                if (large_img && mpi_ranks > 1 && omp_threads > 1) {
                    printf("Auto-selected: MPI+OpenMP (large single image, CPU only).\n");
                    ret = run_hybrid_openmp_mpi_filter(input_filename, output_filename, omp_threads);
                }
                else if (large_img && mpi_ranks > 1) {
                    printf("Auto-selected: MPI domain decomposition (large single image, CPU only).\n");
                    ret = run_mpi_domain_filter(input_filename, output_filename);
                }
                else if (large_img && omp_threads > 1) {
                    printf("Auto-selected: OpenMP (large single image, single rank).\n");
                    ret = run_omp_filter(input_filename, output_filename, omp_threads);
                }
                else if (omp_threads > 1) {
                    printf("Auto-selected: OpenMP (small single image, multi-thread CPU).\n");
                    ret = run_omp_filter(input_filename, output_filename, omp_threads);
                }
                else {
                    printf("Auto-selected: Sequential fallback.\n");
                    ret = run_sequential_filter(input_filename, output_filename);
                }
            }
        }
    }
    // choice
    else if (strcmp(mode, "cuda") == 0) {
        printf("User selected: CUDA approach.\n");
        ret = run_cuda_filter(input_filename, output_filename);
    }
    else if (strcmp(mode, "mpi") == 0) {
        printf("User selected: MPI approach.\n");
        ret = run_mpi_domain_filter(input_filename, output_filename);
    }
    else if (strcmp(mode, "omp") == 0) {
        printf("User selected: OpenMP approach.\n");
        ret = run_omp_filter(input_filename, output_filename, omp_threads);
    }
    else if (strcmp(mode, "mpi_omp") == 0) {
        printf("User selected: MPI+OpenMP hybrid approach.\n");
        ret = run_hybrid_openmp_mpi_filter(input_filename, output_filename, omp_threads);
    }
    else if (strcmp(mode, "cuda_mpi") == 0) {
        printf("User selected: CUDA+MPI hybrid approach.\n");
        ret = run_cuda_mpi_filter(input_filename, output_filename);
    }
    else if (strcmp(mode, "cuda_omp_mpi") == 0) {
        printf("User selected: CUDA+OpenMP+MPI hybrid approach.\n");
        ret = run_cuda_omp_mpi_filter(input_filename, output_filename, omp_threads);
    }
    else if (strcmp(mode, "seq") == 0) {
        printf("User selected: Sequential approach.\n");
        ret = run_sequential_filter(input_filename, output_filename);
    }
    else {
        // If unknown, fallback to the new auto logic
        printf("Unknown mode: %s. Defaulting to auto selection.\n", mode);
        
        char *fake_argv[] = { argv[0], input_filename, output_filename, "auto" };
       
        main(4, fake_argv); 
        MPI_Finalize();
        return 0;
    }

    if (world_rank == 0) {
        printf("\nAdaptive filtering completed with return code %d.\n", ret);
        printf("Exiting Adaptive Filter.\n");
    }

    MPI_Finalize();
    return ret;
}
