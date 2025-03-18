#ifndef CUDA_FILTER_H
#define CUDA_FILTER_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * run_cuda_filter - Runs the CUDA filtering pipeline.
 * @input_filename: Path to the input GIF file.
 * @output_filename: Path to the output GIF file.
 *
 * This function loads the input GIF (using the common I/O functions), processes each image
 * frame in tiles using CUDA kernels (grayscale, blur, and Sobel), and writes out the result.
 *
 * Returns 0 on success, nonzero on error.
 */
int run_cuda_filter(char *input_filename, char *output_filename);

#ifdef __cplusplus
}
#endif

#endif /* CUDA_FILTER_H */
