/*
 * INF560
 *
 * GIF Utilities for sobelf project - Header File
 */
#ifndef GIF_UTILS_H
#define GIF_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "gif_lib.h"

/* Represent one pixel from the image */
typedef struct pixel
{
    int r; /* Red */
    int g; /* Green */
    int b; /* Blue */
} pixel;

/* Represent one GIF image (animated or not) */
typedef struct animated_gif
{
    int n_images; /* Number of images */
    int *width;   /* Width of each image */
    int *height;  /* Height of each image */
    pixel **p;    /* Pixels of each image */
    GifFileType *g; /* Internal representation. DO NOT MODIFY */
} animated_gif;

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif* load_pixels(char* filename);

/* Output GIF image */
int output_modified_read_gif(char* filename, GifFileType* g);

/* Store pixels in GIF file */
int store_pixels(char* filename, animated_gif* image);

#ifdef __cplusplus
}
#endif

#endif /* GIF_UTILS_H */ 