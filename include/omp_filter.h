#ifndef OMP_FILTER_H
#define OMP_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run_omp_filter - Runs the optimized OpenMP filtering pipeline.
 * @input_filename: Path to the input GIF file.
 * @output_filename: Path to the output GIF file.
 * @num_threads: If >0, the number of OpenMP threads to use; otherwise the default is used.
 *
 * Returns 0 on success, nonzero on error.
 */
int run_omp_filter(char *input_filename, char *output_filename, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* OMP_FILTER_H */
