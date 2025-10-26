#ifndef MPI_DOMAIN_FILTER_H
#define MPI_DOMAIN_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run_mpi_domain_filter - Process a GIF file using the MPI domain-decomposition approach.
 * @input_filename: Path to the input GIF.
 * @output_filename: Path to the output GIF.
 *
 * Returns 0 on success, nonzero on error.
 */
int run_mpi_domain_filter(char *input_filename, char *output_filename);

#ifdef __cplusplus
}
#endif

#endif  /* MPI_DOMAIN_FILTER_H */
