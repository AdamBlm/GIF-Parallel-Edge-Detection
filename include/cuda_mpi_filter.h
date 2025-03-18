#ifndef CUDA_MPI_FILTER_H
#define CUDA_MPI_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run_hybrid_cuda_mpi_filter - Runs the hybrid CUDA+MPI filtering pipeline.
 * @input_filename: Path to the input GIF.
 * @output_filename: Path to the output GIF.
 *
 * Returns 0 on success, nonzero on failure.
 */
int run_cuda_mpi_filter(char *input_filename, char *output_filename);

#ifdef __cplusplus
}
#endif

#endif // CUDA_MPI_FILTER_H
