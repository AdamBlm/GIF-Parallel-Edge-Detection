/******************************************************************************
 * sobelf.c
 *
 * A standalone C program that reads a GIF (single or multi-frame), converts it
 * to grayscale, applies a blur filter, then applies a Sobel filter, and writes
 * the result as an output GIF. The reading/writing is done sequentially, but
 * the pixel-intensive operations (grayscale, blur, sobel) are parallelized
 * using OpenMP.
 *
 * Compile with:
 *    gcc -o sobelf sobelf.c -fopenmp -lgif -lm
 *
 * Usage:
 *    ./sobelf input.gif output.gif
 *
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include <gif_lib.h>

/******************************************************************************
 * Helper Macros/Functions
 ******************************************************************************/

/* Clamps an integer value between min and max. */
static inline int clamp(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

/******************************************************************************
 * load_gif_frames
 *
 * Reads an entire GIF file using DGifSlurp (giflib), returning the GifFileType*
 * which contains all frames (SavedImages). The caller is responsible for
 * closing the GifFileType with DGifCloseFile() eventually.
 *
 * Return: pointer to loaded GifFileType, or NULL on error.
 ******************************************************************************/
GifFileType* load_gif_frames(const char *filename) {
    int error = 0;
    GifFileType *gif = DGifOpenFileName(filename, &error);
    if (!gif) {
        fprintf(stderr, "DGifOpenFileName failed: %s\n", GifErrorString(error));
        return NULL;
    }

    if (DGifSlurp(gif) != GIF_OK) {
        fprintf(stderr, "DGifSlurp failed: %s\n", GifErrorString(gif->Error));
        DGifCloseFile(gif, &error);
        return NULL;
    }

    return gif;
}

/******************************************************************************
 * write_gif_frames
 *
 * Writes frames out to a GIF file. We assume each frame has the same dimensions
 * as the input.  We create a simple grayscale palette or re-use an existing one.
 *
 * The frames are given by a 2D buffer (height x width) of 0..255 grayscale
 * intensities for each frame.
 *
 * Params:
 *  out_filename   - name of the output GIF
 *  frames_gray    - array of pointers to grayscale data (1 byte per pixel),
 *                   frames_gray[f][row * width + col].
 *  width, height  - dimensions of each frame
 *  nframes        - number of frames
 *  delay_csecs    - optional: an array of delays (in centiseconds) per frame
 *                   or NULL if not available.
 *
 * Returns 0 on success, nonzero on error.
 ******************************************************************************/
int write_gif_frames(const char *out_filename,
                     unsigned char **frames_gray,
                     int width, int height,
                     int nframes,
                     int *delay_csecs)
{
    int error = 0;
    GifFileType *gif_out = EGifOpenFileName(out_filename, false, &error);
    if (!gif_out) {
        fprintf(stderr, "EGifOpenFileName failed: %s\n", GifErrorString(error));
        return 1;
    }

    // We create a global color map for grayscale (256 entries).
    // Each entry i is (i, i, i).
    ColorMapObject *grayscale_cmap = GifMakeMapObject(256, NULL);
    if (!grayscale_cmap) {
        fprintf(stderr, "Failed to allocate color map.\n");
        EGifCloseFile(gif_out, &error);
        return 1;
    }
    for (int i = 0; i < 256; i++) {
        grayscale_cmap->Colors[i].Red   = i;
        grayscale_cmap->Colors[i].Green = i;
        grayscale_cmap->Colors[i].Blue  = i;
    }

    // Write initial screen descriptor (width, height, color resolution).
    if (EGifPutScreenDesc(gif_out, width, height, 8, 0, grayscale_cmap) == GIF_ERROR) {
        fprintf(stderr, "EGifPutScreenDesc failed: %s\n", GifErrorString(gif_out->Error));
        GifFreeMapObject(grayscale_cmap);
        EGifCloseFile(gif_out, &error);
        return 1;
    }

    // For each frame, we create a SavedImage. We'll also write a Graphics
    // Control Extension (for delay, etc.) if needed.
    for (int f = 0; f < nframes; f++) {
        // Graphics Control Extension (for per-frame delay).
        if (delay_csecs) {
            // 4 bytes: disposal method, user input, transparent color
            // Then 2 bytes for the delay, 1 byte for transparent color index
            //   1 byte block terminator
            unsigned char gce[4 + 2 + 1 + 1] = {0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
            // disposal: 0
            // user input: 0
            // transparent color: 0
            // Next 2 bytes are the delay in cs, in little-endian
            int d = delay_csecs[f];
            gce[4] = d & 0xFF;
            gce[5] = (d >> 8) & 0xFF;
            // no transparent color
            // block terminator
            if (EGifPutExtensionLeader(gif_out, GRAPHICS_EXT_FUNC_CODE) == GIF_ERROR ||
                EGifPutExtensionBlock(gif_out, sizeof(gce), gce) == GIF_ERROR ||
                EGifPutExtensionTrailer(gif_out) == GIF_ERROR) {
                fprintf(stderr, "Failed to write GCE for frame %d\n", f);
            }
        }

        // Now put an image descriptor
        if (EGifPutImageDesc(gif_out, 0, 0, width, height, false, NULL) == GIF_ERROR) {
            fprintf(stderr, "EGifPutImageDesc failed for frame %d: %s\n", f, GifErrorString(gif_out->Error));
            GifFreeMapObject(grayscale_cmap);
            EGifCloseFile(gif_out, &error);
            return 1;
        }

        // We now write out the pixels row by row.
        for (int y = 0; y < height; y++) {
            if (EGifPutLine(gif_out,
                            (GifPixelType*)(&frames_gray[f][y * width]),
                            width) == GIF_ERROR) {
                fprintf(stderr, "EGifPutLine failed on frame %d row %d: %s\n", 
                        f, y, GifErrorString(gif_out->Error));
                GifFreeMapObject(grayscale_cmap);
                EGifCloseFile(gif_out, &error);
                return 1;
            }
        }
    }

    // Close out the GIF file
    GifFreeMapObject(grayscale_cmap);
    if (EGifCloseFile(gif_out, &error) == GIF_ERROR) {
        fprintf(stderr, "EGifCloseFile failed: %s\n", GifErrorString(error));
        return 1;
    }
    return 0;
}

/******************************************************************************
 * main
 ******************************************************************************/
int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.gif> <output.gif>\n", argv[0]);
        return 1;
    }
    const char *in_filename  = argv[1];
    const char *out_filename = argv[2];

    double t0, t1, t_read, t_process, t_write;

    // -------------------------------------------------------------------------
    // 1. Read (sequential)
    // -------------------------------------------------------------------------
    t0 = omp_get_wtime();
    GifFileType *gif_in = load_gif_frames(in_filename);
    t1 = omp_get_wtime();
    t_read = t1 - t0;
    if (!gif_in) {
        fprintf(stderr, "Failed to read GIF: %s\n", in_filename);
        return 1;
    }

    int width      = gif_in->SWidth;
    int height     = gif_in->SHeight;
    int nframes    = gif_in->ImageCount;
    // Some GIFs store a per-frame delay in the global extension blocks or in
    // the Graphics Control Extension. For simplicity, we do partial reading of it:
    // we will store each frame's delay in an array (if found).
    int *frame_delays = (int*)calloc(nframes, sizeof(int)); // in centiseconds

    // We'll allocate an array of frames in 8-bit grayscale form:
    // frames_gray[f][y * width + x].
    unsigned char **frames_gray = (unsigned char**)malloc(nframes * sizeof(unsigned char*));
    for (int f = 0; f < nframes; f++) {
        frames_gray[f] = (unsigned char*)calloc(width * height, sizeof(unsigned char));
    }

    // Extract any per-frame pixel data and also read any GCE delays if present.
    // We'll do the extraction in parallel at the *frame* level or we can do it sequentially.
    // Here we do it *sequentially* to keep it simple, as it's just memory copy:
    for (int f = 0; f < nframes; f++) {
        SavedImage *frame = &gif_in->SavedImages[f];
        // Attempt to read a GCE for the delay time:
        for (int e = 0; e < frame->ExtensionBlockCount; e++) {
            ExtensionBlock *ext = &frame->ExtensionBlocks[e];
            if (ext->Function == GRAPHICS_EXT_FUNC_CODE && ext->ByteCount == 4) {
                // Bytes 1-2 store the delay in 1/100 sec
                int delay = ext->Bytes[1] | (ext->Bytes[2] << 8);
                frame_delays[f] = delay;
            }
        }

        // If the frame has its own local color map, use it; otherwise use global
        ColorMapObject *cmap = (frame->ImageDesc.ColorMap ? frame->ImageDesc.ColorMap
                                   : gif_in->SColorMap);

        // The frame's `RasterBits` are indices into the color map
        GifByteType *bits = frame->RasterBits;
        for (int i = 0; i < width * height; i++) {
            int idx = bits[i]; // color index
            GifColorType color = cmap->Colors[idx];
            // We'll store in frames_gray as grayscale (0..255).
            // A simple approach is: gray = 0.299*R + 0.587*G + 0.114*B.
            unsigned char gray = (unsigned char)(0.299 * color.Red +
                                                 0.587 * color.Green +
                                                 0.114 * color.Blue);
            frames_gray[f][i] = gray;
        }
    }

    // -------------------------------------------------------------------------
    // 2. Process (in parallel): grayscale (already done), blur, sobel
    //    - Grayscale is done above, but we can re-check or do other transformations.
    // -------------------------------------------------------------------------
    t0 = omp_get_wtime();

    // We'll do a blur, then a sobel, for each frame.
    // We can parallelize across frames or within each frame's pixel loops.
    // Let's do BOTH for demonstration:
    //   #pragma omp parallel for
    //   for (f = 0; f < nframes; f++) { ... }

    // For demonstration, do a parallel-for over frames:
    #pragma omp parallel for
    for (int f = 0; f < nframes; f++) {
        unsigned char *src_frame = frames_gray[f];

        // ------------------
        // Blur step
        // We will create a temporary buffer for the blurred result,
        // skip the border to avoid out-of-bounds.
        // ------------------
        unsigned char *blur_frame = (unsigned char*)calloc(width * height, sizeof(unsigned char));

        #pragma omp parallel for
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int sum = 0;
                // naive 3x3 blur
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        sum += src_frame[yy * width + xx];
                    }
                }
                int val = sum / 9;
                blur_frame[y * width + x] = (unsigned char)val;
            }
        }

        // Optionally handle edges as unblurred or replicate, but let's keep them as 0 for simplicity
        // or copy from src_frame
        // We'll just copy the border from src_frame to blur_frame
        for (int x = 0; x < width; x++) {
            blur_frame[0 * width + x]            = src_frame[0 * width + x];
            blur_frame[(height - 1) * width + x] = src_frame[(height - 1) * width + x];
        }
        for (int y = 0; y < height; y++) {
            blur_frame[y * width + 0]            = src_frame[y * width + 0];
            blur_frame[y * width + (width - 1)]  = src_frame[y * width + (width - 1)];
        }

        // ------------------
        // Sobel step
        // We'll overwrite src_frame with the sobel result (final).
        // Create Gx, Gy using the standard Sobel operators:
        //    Gx = [-1  0  +1
        //          -2  0  +2
        //          -1  0  +1]
        //    Gy = [-1 -2 -1
        //           0  0  0
        //          +1 +2 +1]
        // We'll skip the border for simplicity.
        // ------------------
        #pragma omp parallel for
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int gx = 0, gy = 0;
                // row above
                gx += -1 * blur_frame[(y-1) * width + (x-1)];
                gx +=  0 * blur_frame[(y-1) * width + (x  )];
                gx +=  1 * blur_frame[(y-1) * width + (x+1)];
                gy += -1 * blur_frame[(y-1) * width + (x-1)];
                gy += -2 * blur_frame[(y  ) * width + (x-1)];
                gy += -1 * blur_frame[(y+1) * width + (x-1)];

                // row center
                gx += -2 * blur_frame[(y  ) * width + (x-1)];
                gx +=  2 * blur_frame[(y  ) * width + (x+1)];
                // row below
                gx += -1 * blur_frame[(y+1) * width + (x-1)];
                gx +=  0 * blur_frame[(y+1) * width + (x  )];
                gx +=  1 * blur_frame[(y+1) * width + (x+1)];
                gy +=  1 * blur_frame[(y-1) * width + (x+1)];
                gy +=  2 * blur_frame[(y  ) * width + (x+1)];
                gy +=  1 * blur_frame[(y+1) * width + (x+1)];

                int mag = (int)(sqrt((double)(gx * gx + gy * gy)));
                mag = clamp(mag, 0, 255);
                src_frame[y * width + x] = (unsigned char)mag;
            }
        }

        // For the border, just copy from the blurred version or set to 0
        for (int x = 0; x < width; x++) {
            src_frame[0 * width + x]            = blur_frame[0 * width + x];
            src_frame[(height - 1) * width + x] = blur_frame[(height - 1) * width + x];
        }
        for (int y = 0; y < height; y++) {
            src_frame[y * width + 0]            = blur_frame[y * width + 0];
            src_frame[y * width + (width - 1)]  = blur_frame[y * width + (width - 1)];
        }

        free(blur_frame);
    }

    t1 = omp_get_wtime();
    t_process = t1 - t0;

    // -------------------------------------------------------------------------
    // 3. Write (sequential)
    // -------------------------------------------------------------------------
    t0 = omp_get_wtime();
    int ret = write_gif_frames(out_filename, frames_gray, width, height, nframes, frame_delays);
    t1 = omp_get_wtime();
    t_write = t1 - t0;

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    // Free frames_gray
    for (int f = 0; f < nframes; f++) {
        free(frames_gray[f]);
    }
    free(frames_gray);
    free(frame_delays);

    // Close input GIF
    int error;
    DGifCloseFile(gif_in, &error);

    if (ret != 0) {
        fprintf(stderr, "Failed to write output GIF: %s\n", out_filename);
        return 1;
    }

    // Print timing info
    printf("Reading time:   %.3f s\n", t_read);
    printf("Processing time:%.3f s\n", t_process);
    printf("Writing time:   %.3f s\n", t_write);
    printf("Total time:     %.3f s\n", t_read + t_process + t_write);

    return 0;
}
