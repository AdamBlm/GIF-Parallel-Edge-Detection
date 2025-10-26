#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "gif_io.h"

/* ------------------------------------------------------------------
 * Function Definitions
 * ------------------------------------------------------------------ */

animated_gif * load_pixels(char *filename)
{
    int error;
    GifFileType *g = DGifOpenFileName(filename, &error);
    if (g == NULL) {
        fprintf(stderr, "Error opening input GIF %s\n", filename);
        return NULL;
    }
    error = DGifSlurp(g);
    if (error != GIF_OK) {
        fprintf(stderr, "Error DGifSlurp: %d <%s>\n", error, GifErrorString(g->Error));
        return NULL;
    }

    int n_images = g->ImageCount;
    int *width = (int*) malloc(n_images * sizeof(int));
    int *height = (int*) malloc(n_images * sizeof(int));
    pixel **p = (pixel**) malloc(n_images * sizeof(pixel*));

    /* Convert color indices -> RGB in p[i] */
    ColorMapObject *colmap = g->SColorMap;
    if (!colmap) {
        fprintf(stderr, "Error: global colormap is NULL\n");
        return NULL;
    }

    for (int i = 0; i < n_images; i++) {
        width[i]  = g->SavedImages[i].ImageDesc.Width;
        height[i] = g->SavedImages[i].ImageDesc.Height;
        p[i] = (pixel*) malloc(width[i] * height[i] * sizeof(pixel));

        for (int j = 0; j < width[i] * height[i]; j++) {
            int c = g->SavedImages[i].RasterBits[j];
            p[i][j].r = colmap->Colors[c].Red;
            p[i][j].g = colmap->Colors[c].Green;
            p[i][j].b = colmap->Colors[c].Blue;
        }
    }

    animated_gif *image = (animated_gif*) malloc(sizeof(animated_gif));
    image->n_images = n_images;
    image->width = width;
    image->height = height;
    image->p = p;
    image->g = g;
    return image;
}

int output_modified_read_gif(char *filename, GifFileType *g)
{
    int error2;
    GifFileType *g2 = EGifOpenFileName(filename, false, &error2);
    if (!g2) {
        fprintf(stderr, "Error EGifOpenFileName %s\n", filename);
        return 0;
    }
    g2->SWidth = g->SWidth;
    g2->SHeight = g->SHeight;
    g2->SColorResolution = g->SColorResolution;
    g2->SBackGroundColor = g->SBackGroundColor;
    g2->AspectByte = g->AspectByte;
    g2->SColorMap = g->SColorMap;
    g2->ImageCount = g->ImageCount;
    g2->SavedImages = g->SavedImages;
    g2->ExtensionBlockCount = g->ExtensionBlockCount;
    g2->ExtensionBlocks = g->ExtensionBlocks;

    error2 = EGifSpew(g2);
    if (error2 != GIF_OK) {
        fprintf(stderr, "Error after EGifSpew: %d <%s>\n", error2, GifErrorString(g2->Error));
        return 0;
    }
    return 1;
}

int store_pixels(char *filename, animated_gif *image)
{
    int n_colors = 0;
    GifColorType *colormap = (GifColorType*) malloc(256 * sizeof(GifColorType));
    if (!colormap) {
        fprintf(stderr, "Cannot allocate 256-colormap\n");
        return 0;
    }
    /* White by default */
    for (int i = 0; i < 256; i++) {
        colormap[i].Red = 255;
        colormap[i].Green = 255;
        colormap[i].Blue = 255;
    }

    /* Put background color at index 0 (example) */
    {
        int bg = image->g->SBackGroundColor;
        int r = image->g->SColorMap->Colors[bg].Red;
        int g = image->g->SColorMap->Colors[bg].Green;
        int b = image->g->SColorMap->Colors[bg].Blue;
        int moy = (r + g + b) / 3;
        if (moy < 0) moy = 0;
        if (moy > 255) moy = 255;
        colormap[0].Red = moy;
        colormap[0].Green = moy;
        colormap[0].Blue = moy;
        image->g->SBackGroundColor = 0;
    }
    n_colors++;

    /* Gather all final colors */
    for (int i = 0; i < image->n_images; i++) {
        int W = image->width[i];
        int H = image->height[i];
        for (int j = 0; j < W * H; j++) {
            pixel px = image->p[i][j];
            int found = -1;
            for (int c = 0; c < n_colors; c++) {
                if (px.r == colormap[c].Red &&
                    px.g == colormap[c].Green &&
                    px.b == colormap[c].Blue) {
                    found = c;
                    break;
                }
            }
            if (found < 0) {
                if (n_colors >= 256) {
                    fprintf(stderr, "Error: too many colors\n");
                    return 0;
                }
                colormap[n_colors].Red = px.r;
                colormap[n_colors].Green = px.g;
                colormap[n_colors].Blue = px.b;
                n_colors++;
            }
        }
    }

    /* Round up to next power-of-2 if needed */
    int real_n = (1 << GifBitSize(n_colors));
    if (real_n != n_colors) {
        n_colors = real_n;
    }

    ColorMapObject *cmo = GifMakeMapObject(n_colors, colormap);
    if (!cmo) {
        fprintf(stderr, "Error building ColorMapObject\n");
        return 0;
    }
    image->g->SColorMap = cmo;

    /* Rewrite each frame's RasterBits from p[i] */
    for (int i = 0; i < image->n_images; i++) {
        int W = image->width[i];
        int H = image->height[i];
        for (int j = 0; j < W * H; j++) {
            pixel px = image->p[i][j];
            int idx_found = -1;
            for (int c = 0; c < n_colors; c++) {
                if (px.r == cmo->Colors[c].Red &&
                    px.g == cmo->Colors[c].Green &&
                    px.b == cmo->Colors[c].Blue) {
                    idx_found = c;
                    break;
                }
            }
            if (idx_found < 0) {
                fprintf(stderr, "Error: pixel not found in new colormap\n");
                return 0;
            }
            image->g->SavedImages[i].RasterBits[j] = idx_found;
        }
    }

    if (!output_modified_read_gif(filename, image->g)) {
        return 0;
    }
    return 1;
}
