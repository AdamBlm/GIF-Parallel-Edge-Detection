#ifndef CUDA_OMP_MPI_FILTER_H
#define CUDA_OMP_MPI_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run_hybrid_cuda_omp_mpi_filter - Runs the hybrid CUDA+OpenMP+MPI filtering pipeline.
 * @input_filename: Path to the input GIF.
 * @output_filename: Path to the output GIF.
 * @num_threads: Number of OpenMP threads to use.
 *
 * Returns 0 on success, nonzero on failure.
 */
int run_cuda_omp_mpi_filter(char *input_filename, char *output_filename, int num_threads);

#ifdef __cplusplus
}
#endif

#endif // CUDA_OMP_MPI_FILTER_H
