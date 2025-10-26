#ifndef OMP_FILTER_H
#define OMP_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif


int run_omp_filter(char *input_filename, char *output_filename, int num_threads);

#ifdef __cplusplus
}
#endif

#endif /* OMP_FILTER_H */
