#ifndef CUDA_OMP_MPI_FILTER_H
#define CUDA_OMP_MPI_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif


int run_cuda_omp_mpi_filter(char *input_filename, char *output_filename, int num_threads);

#ifdef __cplusplus
}
#endif

#endif // CUDA_OMP_MPI_FILTER_H
