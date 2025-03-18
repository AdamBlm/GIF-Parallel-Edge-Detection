#ifndef HYBRID_OPENMP_MPI_FILTER_H
#define HYBRID_OPENMP_MPI_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run_hybrid_filter - Runs the optimized Hybrid OpenMP+MPI filtering pipeline.
 * @input_filename: Path to the input GIF file.
 * @output_filename: Path to the output GIF file.
 * @num_threads: Number of OpenMP threads to use in each MPI process 
 *               (if <= 0, the default number is used).
 *
 * Returns 0 on success, nonzero on failure.
 */
int run_hybrid_openmp_mpi_filter(char *input_filename, char *output_filename, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* HYBRID_OPENMP_MPI_FILTER_H */
