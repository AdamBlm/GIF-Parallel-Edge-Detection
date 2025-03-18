#ifndef GIF_IO_H
#define GIF_IO_H

#include "gif_lib.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------
 * Data Structures
 * ------------------------------------------------------------------ */

/* Represent one pixel from the image */
typedef struct pixel {
    int r;  /* Red */
    int g;  /* Green */
    int b;  /* Blue */
} pixel;

/* Holds all frames (images) from one GIF */
typedef struct animated_gif {
    int n_images;      /* Number of images */
    int *width;        /* Width of each image */
    int *height;       /* Height of each image */
    pixel **p;         /* For each image i, p[i] is a flat array of W×H pixels in row-major order */
    GifFileType *g;    /* Internal GIF structure for rewriting */
} animated_gif;

/* ------------------------------------------------------------------
 * Function Prototypes
 * ------------------------------------------------------------------ */

/**
 * load_pixels - Loads a GIF image from a file and converts it into an animated_gif structure.
 * @filename: Path to the input GIF.
 *
 * Returns a pointer to an animated_gif structure, or NULL on error.
 */
animated_gif * load_pixels(char *filename);

/**
 * output_modified_read_gif - Writes the final GIF using the modified internal GIF structure.
 * @filename: Path to the output GIF.
 * @g: Pointer to the internal GifFileType structure.
 *
 * Returns 1 on success, 0 on failure.
 */
int output_modified_read_gif(char *filename, GifFileType *g);

/**
 * store_pixels - Re-encodes the final pixel arrays into the GIF's raster bits and writes the output GIF.
 * @filename: Path to the output GIF.
 * @image: Pointer to the animated_gif structure.
 *
 * Returns 1 on success, 0 on failure.
 */
int store_pixels(char *filename, animated_gif *image);

#ifdef __cplusplus
}
#endif

#endif /* GIF_IO_H */
