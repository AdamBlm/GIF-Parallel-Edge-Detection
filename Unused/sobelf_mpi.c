/*
 * INF560 - Image Filtering Project (MPI version)
 *
 * This code loads a GIF, applies several filters (grayscale, blur, Sobel)
 * to each frame, and writes out the modified GIF.
 *
 * MPI parallelization is done by distributing the frames among MPI ranks:
 *
 *   1. Rank 0 loads the entire GIF.
 *   2. Rank 0 determines the distribution of frames and sends each rank
 *      its assigned frames (width, height, and pixel data).
 *   3. Each rank applies the filters on its subset.
 *   4. Each non-zero rank sends its filtered frames back to rank 0.
 *   5. Rank 0 gathers all frames and writes out the final GIF.
 *
 * Compile with:
 *    mpicc -o sobelf_mpi sobelf_mpi.c -lm -lgif
 *
 * Run with, for example:
 *    mpirun -np 4 ./sobelf_mpi input.gif output.gif
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>
#include "gif_lib.h"
#include "gif_utils.h"  /* Include the shared gif utilities */
/* Set this macro to 1 to enable debugging information */
#define SOBELF_DEBUG 0

/* Define MPI tags for messaging */
#define MPI_TAG_IMAGE_DATA 100
#define MPI_TAG_IMAGE_DIMS 101
#define MPI_TAG_RESULT 102

/* Helper for collective operations */
typedef struct scatter_info {
    int *sendcounts;  /* Number of elements to send to each process */
    int *displs;      /* Displacement for each process */
    int *image_counts; /* Number of images per process */
    int *image_displs; /* Displacement for image indices */
    int *scatter_byte_counts; /* Number of bytes to send to each process - cached for reuse */
    int *scatter_byte_displs; /* Byte displacement for each process - cached for reuse */
} scatter_info;

/* Create scatter/gather information for collective operations */
scatter_info* create_scatter_info(int n_images, int size) {
    scatter_info* info = (scatter_info*)malloc(sizeof(scatter_info));
    if (!info) return NULL;
    
    info->sendcounts = (int*)calloc(size, sizeof(int));
    info->displs = (int*)calloc(size, sizeof(int));
    info->image_counts = (int*)calloc(size, sizeof(int));
    info->image_displs = (int*)calloc(size, sizeof(int));
    info->scatter_byte_counts = (int*)calloc(size, sizeof(int));
    info->scatter_byte_displs = (int*)calloc(size, sizeof(int));
    
    if (!info->sendcounts || !info->displs || 
        !info->image_counts || !info->image_displs ||
        !info->scatter_byte_counts || !info->scatter_byte_displs) {
        // Free allocated memory if any allocation failed
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
        return NULL;
    }
    
    // Distribute images evenly among processes
    int base_count = n_images / size;
    int remainder = n_images % size;
    
    for (int i = 0; i < size; i++) {
        info->image_counts[i] = base_count + (i < remainder ? 1 : 0);
        if (i > 0) {
            info->image_displs[i] = info->image_displs[i-1] + info->image_counts[i-1];
        }
    }
    
    return info;
}

/* Free scatter/gather information */
void free_scatter_info(scatter_info *info) {
    if (info) {
        if (info->sendcounts) free(info->sendcounts);
        if (info->displs) free(info->displs);
        if (info->image_counts) free(info->image_counts);
        if (info->image_displs) free(info->image_displs);
        if (info->scatter_byte_counts) free(info->scatter_byte_counts);
        if (info->scatter_byte_displs) free(info->scatter_byte_displs);
        free(info);
    }
}

/* Calculate pixel counts for scatter/gather operations */
void calculate_pixel_counts(scatter_info* scatter_data, int* widths, int* heights, int size) {
    if (!scatter_data || !widths || !heights) return;
    
    // Calculate displacements and pixel counts
    for (int i = 0; i < size; i++) {
        int start_img = scatter_data->image_displs[i];
        int end_img = start_img + scatter_data->image_counts[i];
        
        scatter_data->sendcounts[i] = 0;
        for (int j = start_img; j < end_img; j++) {
            scatter_data->sendcounts[i] += widths[j] * heights[j];
        }
        
        if (i > 0) {
            scatter_data->displs[i] = scatter_data->displs[i-1] + scatter_data->sendcounts[i-1];
        }
        
        // Pre-calculate byte counts and displacements to avoid redundant calculations
        scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
        scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
    }
}

/* Forward declarations for functions from gif_utils.h */
/*
 * Load a GIF image from a file and return a
 * structure of type animated_gif.
 */
animated_gif * load_pixels(char * filename);

/*
 * Output GIF image
 */
int output_modified_read_gif(char * filename, GifFileType * g);

/*
 * Store pixels in GIF file
 */
int store_pixels(char * filename, animated_gif * image);

void
apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ )
        {
            int moy ;

            moy = (p[i][j].r + p[i][j].g + p[i][j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[i][j].r = moy ;
            p[i][j].g = moy ;
            p[i][j].b = moy ;
        }
    }
}

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

/* Define tile size for tiling optimization */
#define TILE_SIZE 32

/* Simple min function for tile boundaries */
static inline int min(int a, int b) {
    return (a < b) ? a : b;
}

void
apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i;
    pixel ** p;
    pixel * buffer;
    
    p = image->p;
    
    /* Process each image */
    for (i = 0; i < image->n_images; i++)
    {
        int width = image->width[i];
        int height = image->height[i];
        int end = 0;
        int n_iter = 0;
        
        /* Allocate memory for blur buffer */
        buffer = (pixel *)malloc(width * height * sizeof(pixel));
        if (buffer == NULL) {
            fprintf(stderr, "Unable to allocate memory for blur buffer\n");
            continue;  /* Skip this image */
        }
        
        /* Perform blur iterations until convergence */
        do
        {
            end = 1;
            n_iter++;
            
            /* Initialize buffer with original pixel values */
            for (int j = 0; j < height; j++) {
                for (int k = 0; k < width; k++) {
                    int index = CONV(j, k, width);
                    buffer[index].r = p[i][index].r;
                    buffer[index].g = p[i][index].g;
                    buffer[index].b = p[i][index].b;
                }
            }
            
            /* Apply blur on the entire image except borders, using tiling for better cache utilization */
            for (int jj = size; jj < height - size; jj += TILE_SIZE)
            {
                for (int kk = size; kk < width - size; kk += TILE_SIZE)
                {
                    /* Process a tile */
                    for (int j = jj; j < min(jj + TILE_SIZE, height - size); j++)
                    {
                        for (int k = kk; k < min(kk + TILE_SIZE, width - size); k++)
                        {
                            int t_r = 0;
                            int t_g = 0;
                            int t_b = 0;
                            
                            /* Compute average over the neighborhood */
                            for (int stencil_j = -size; stencil_j <= size; stencil_j++)
                            {
                                for (int stencil_k = -size; stencil_k <= size; stencil_k++)
                                {
                                    int idx = CONV(j+stencil_j, k+stencil_k, width);
                                    t_r += p[i][idx].r;
                                    t_g += p[i][idx].g;
                                    t_b += p[i][idx].b;
                                }
                            }
                            
                            /* Calculate the average value for the pixel */
                            const int denom = (2*size+1)*(2*size+1);
                            buffer[CONV(j, k, width)].r = t_r / denom;
                            buffer[CONV(j, k, width)].g = t_g / denom;
                            buffer[CONV(j, k, width)].b = t_b / denom;
                        }
                    }
                }
            }
            
            /* Check for convergence and update pixels */
            int continue_flag = 0;
            
            for (int j = size; j < height - size; j++)
            {
                for (int k = size; k < width - size; k++)
                {
                    int index = CONV(j, k, width);
                    int diff_r = buffer[index].r - p[i][index].r;
                    int diff_g = buffer[index].g - p[i][index].g;
                    int diff_b = buffer[index].b - p[i][index].b;
                    
                    if (abs(diff_r) > threshold || abs(diff_g) > threshold || abs(diff_b) > threshold)
                    {
                        continue_flag = 1;
                    }
                    
                    /* Update the pixel value with the blurred value */
                    p[i][index].r = buffer[index].r;
                    p[i][index].g = buffer[index].g;
                    p[i][index].b = buffer[index].b;
                }
            }
            
            end = !continue_flag;
        }
        while (threshold > 0 && !end);
        
        /* Free buffer */
        free(buffer);
    }
}

void
apply_sobel_filter( animated_gif * image )
{
    int i;
    pixel ** p;
    pixel * buffer;
    
    p = image->p;
    
    /* Process each image */
    for (i = 0; i < image->n_images; i++)
    {
        int width = image->width[i];
        int height = image->height[i];
        
        /* Allocate memory for sobel result */
        buffer = (pixel *)malloc(width * height * sizeof(pixel));
        if (buffer == NULL) {
            fprintf(stderr, "Unable to allocate memory for sobel buffer\n");
            continue;  /* Skip this image */
        }
        
        /* First, copy the original image to buffer to preserve it */
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                int index = CONV(j, k, width);
                buffer[index].r = p[i][index].r;
                buffer[index].g = p[i][index].g;
                buffer[index].b = p[i][index].b;
            }
        }
        
        /* Apply Sobel filter with improved memory access pattern */
        for (int j = 1; j < height - 1; j++)
        {
            /* Calculate row offsets for better cache locality */
            const int row_prev = CONV(j-1, 0, width);
            const int row_curr = CONV(j, 0, width);
            const int row_next = CONV(j+1, 0, width);
            
            for (int k = 1; k < width - 1; k++)
            {
                /* Get pixel values from the 3x3 neighborhood for each channel */
                /* Red channel */
                int pixel_red_no = p[i][row_prev + k-1].r;
                int pixel_red_n  = p[i][row_prev + k].r;
                int pixel_red_ne = p[i][row_prev + k+1].r;
                int pixel_red_o  = p[i][row_curr + k-1].r;
                int pixel_red_e  = p[i][row_curr + k+1].r;
                int pixel_red_so = p[i][row_next + k-1].r;
                int pixel_red_s  = p[i][row_next + k].r;
                int pixel_red_se = p[i][row_next + k+1].r;
                
                /* Green channel */
                int pixel_green_no = p[i][row_prev + k-1].g;
                int pixel_green_n  = p[i][row_prev + k].g;
                int pixel_green_ne = p[i][row_prev + k+1].g;
                int pixel_green_o  = p[i][row_curr + k-1].g;
                int pixel_green_e  = p[i][row_curr + k+1].g;
                int pixel_green_so = p[i][row_next + k-1].g;
                int pixel_green_s  = p[i][row_next + k].g;
                int pixel_green_se = p[i][row_next + k+1].g;
                
                /* Blue channel */
                int pixel_blue_no = p[i][row_prev + k-1].b;
                int pixel_blue_n  = p[i][row_prev + k].b;
                int pixel_blue_ne = p[i][row_prev + k+1].b;
                int pixel_blue_o  = p[i][row_curr + k-1].b;
                int pixel_blue_e  = p[i][row_curr + k+1].b;
                int pixel_blue_so = p[i][row_next + k-1].b;
                int pixel_blue_s  = p[i][row_next + k].b;
                int pixel_blue_se = p[i][row_next + k+1].b;
                
                /* Calculate Sobel gradients for each channel */
                /* Red channel */
                float deltaX_red = -pixel_red_no + pixel_red_ne - 2*pixel_red_o + 2*pixel_red_e - pixel_red_so + pixel_red_se;
                float deltaY_red = pixel_red_se + 2*pixel_red_s + pixel_red_so - pixel_red_ne - 2*pixel_red_n - pixel_red_no;
                float val_red = sqrt(deltaX_red * deltaX_red + deltaY_red * deltaY_red)/4;
                
                /* Green channel */
                float deltaX_green = -pixel_green_no + pixel_green_ne - 2*pixel_green_o + 2*pixel_green_e - pixel_green_so + pixel_green_se;
                float deltaY_green = pixel_green_se + 2*pixel_green_s + pixel_green_so - pixel_green_ne - 2*pixel_green_n - pixel_green_no;
                float val_green = sqrt(deltaX_green * deltaX_green + deltaY_green * deltaY_green)/4;
                
                /* Blue channel */
                float deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;
                float deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;
                float val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;
                
                /* Calculate maximum gradient value across all channels for better edge detection */
                float max_val = val_red;
                if (val_green > max_val) max_val = val_green;
                if (val_blue > max_val) max_val = val_blue;
                
                /* Apply binary thresholding for edge detection */
                if (max_val > 15)  /* Adjusted threshold to get visible edges */
                {
                    buffer[row_curr + k].r = 255;
                    buffer[row_curr + k].g = 255;
                    buffer[row_curr + k].b = 255;
                }
                else
                {
                    buffer[row_curr + k].r = 0;
                    buffer[row_curr + k].g = 0;
                    buffer[row_curr + k].b = 0;
                }
            }
        }
        
        /* Copy the result back to the original image */
        for (int j = 0; j < height; j++)
        {
            for (int k = 0; k < width; k++)
            {
                int index = CONV(j, k, width);
                p[i][index].r = buffer[index].r;
                p[i][index].g = buffer[index].g;
                p[i][index].b = buffer[index].b;
            }
        }
        
        /* Free buffer */
        free(buffer);
    }
}

/* Helper function to free allocated resources */
void free_resources(pixel **p, int n_images) {
    if (p) {
        for (int i = 0; i < n_images; i++) {
            if (p[i]) free(p[i]);
        }
        free(p);
    }
}

int main(int argc, char **argv) {
    /* Initialize MPI */
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    char *input_filename, *output_filename;
    animated_gif *image = NULL;
    int n_images = 0;
    scatter_info *scatter_data = NULL;
    struct timeval t1, t2;
    double duration;
    
    /* Create MPI datatype for pixel */
    MPI_Datatype MPI_PIXEL;
    int blocklengths[3] = {1, 1, 1};  /* 3 components: r, g, b */
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR, MPI_UNSIGNED_CHAR};
    pixel dummy_pixel;
    
    /* Calculate displacements for the MPI datatype */
    MPI_Aint base_address;
    MPI_Get_address(&dummy_pixel, &base_address);
    MPI_Get_address(&dummy_pixel.r, &displacements[0]);
    MPI_Get_address(&dummy_pixel.g, &displacements[1]);
    MPI_Get_address(&dummy_pixel.b, &displacements[2]);
    
    /* Make displacements relative */
    displacements[0] = displacements[0] - base_address;
    displacements[1] = displacements[1] - base_address;
    displacements[2] = displacements[2] - base_address;
    
    /* Create and commit the MPI datatype */
    MPI_Type_create_struct(3, blocklengths, displacements, types, &MPI_PIXEL);
    MPI_Type_commit(&MPI_PIXEL);
    
    /* Check command-line arguments */
    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s input.gif output.gif\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    input_filename = argv[1];
    output_filename = argv[2];
    
    /* Process 0 (master) loads the image */
    if (rank == 0) {
        gettimeofday(&t1, NULL);
        image = load_pixels(input_filename);
        if (!image) {
            fprintf(stderr, "Error loading GIF\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        n_images = image->n_images;
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec) / 1e6);
        printf("GIF loaded from file %s with %d image(s) in %lf s\n", 
               input_filename, image->n_images, duration);
        
        /* Create distribution information */
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Failed to allocate scatter info\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Calculate pixel counts for distribution */
        calculate_pixel_counts(scatter_data, image->width, image->height, size);
    }
    
    /* Broadcast the number of images to all processes */
    MPI_Bcast(&n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Allocate memory for image dimensions on non-root processes */
    int *image_widths = NULL;
    int *image_heights = NULL;
    
    if (rank == 0) {
        image_widths = image->width;
        image_heights = image->height;
    } else {
        image_widths = (int *)malloc(n_images * sizeof(int));
        image_heights = (int *)malloc(n_images * sizeof(int));
        if (!image_widths || !image_heights) {
            fprintf(stderr, "Process %d: Memory allocation failed\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        scatter_data = create_scatter_info(n_images, size);
        if (!scatter_data) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Broadcast the image dimensions to all processes */
    MPI_Bcast(image_widths, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(image_heights, n_images, MPI_INT, 0, MPI_COMM_WORLD);
    
    /* Broadcast scatter info arrays in a single operation */
    int total_scatter_info_size = size * 4;
    int* all_scatter_info = NULL;
    
    if (rank == 0) {
        all_scatter_info = (int*)malloc(total_scatter_info_size * sizeof(int));
        if (!all_scatter_info) {
            fprintf(stderr, "Process 0: Failed to allocate scatter info buffer\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        // Pack all scatter arrays into a single buffer for efficient transfer
        memcpy(&all_scatter_info[0], scatter_data->image_counts, size * sizeof(int));
        memcpy(&all_scatter_info[size], scatter_data->image_displs, size * sizeof(int));
        memcpy(&all_scatter_info[size*2], scatter_data->sendcounts, size * sizeof(int));
        memcpy(&all_scatter_info[size*3], scatter_data->displs, size * sizeof(int));
    } else {
        all_scatter_info = (int*)malloc(total_scatter_info_size * sizeof(int));
        if (!all_scatter_info) {
            fprintf(stderr, "Process %d: Failed to allocate scatter info buffer\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    // One broadcast instead of multiple
    MPI_Bcast(all_scatter_info, total_scatter_info_size, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        // Unpack the received data
        memcpy(scatter_data->image_counts, &all_scatter_info[0], size * sizeof(int));
        memcpy(scatter_data->image_displs, &all_scatter_info[size], size * sizeof(int));
        memcpy(scatter_data->sendcounts, &all_scatter_info[size*2], size * sizeof(int));
        memcpy(scatter_data->displs, &all_scatter_info[size*3], size * sizeof(int));
        
        // Calculate byte counts and displacements
        for (int i = 0; i < size; i++) {
            scatter_data->scatter_byte_counts[i] = scatter_data->sendcounts[i] * sizeof(pixel);
            scatter_data->scatter_byte_displs[i] = scatter_data->displs[i] * sizeof(pixel);
        }
    }
    
    free(all_scatter_info);
    
    /* Get information for this process */
    int my_n_images = scatter_data->image_counts[rank];
    int my_start_image = scatter_data->image_displs[rank];
    
    /* Create a local animated_gif structure for processing */
    animated_gif local_img;
    local_img.n_images = my_n_images;
    local_img.width = (int *)malloc(my_n_images * sizeof(int));
    local_img.height = (int *)malloc(my_n_images * sizeof(int));
    local_img.p = (pixel **)malloc(my_n_images * sizeof(pixel *));
    local_img.g = NULL;  // Not used for processing
    
    if (!local_img.width || !local_img.height || !local_img.p) {
        fprintf(stderr, "Process %d: Memory allocation failed for local image structure\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    /* Initialize local image dimensions and allocate pixel buffers */
    for (int i = 0; i < my_n_images; i++) {
        int img_idx = my_start_image + i;
        local_img.width[i] = image_widths[img_idx];
        local_img.height[i] = image_heights[img_idx];
        int npixels = local_img.width[i] * local_img.height[i];
        local_img.p[i] = (pixel *)malloc(npixels * sizeof(pixel));
        if (!local_img.p[i]) {
            fprintf(stderr, "Process %d: Memory allocation failed for local image %d\n", rank, i);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    
    /* Create contiguous buffers for scatter/gather operations */
    pixel *all_pixels = NULL;
    pixel *scattered_pixels = NULL;
    int sendcount = scatter_data->sendcounts[rank];
    
    if (rank == 0) {
        /* Calculate total pixels for all processes */
        int total_pixels = 0;
        for (int i = 0; i < size; i++) {
            total_pixels += scatter_data->sendcounts[i];
        }
        
        /* Allocate and fill buffer for all pixels */
        all_pixels = (pixel*)malloc(total_pixels * sizeof(pixel));
        if (!all_pixels) {
            fprintf(stderr, "Process 0: Memory allocation failed for all_pixels\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Copy pixels to contiguous buffer for scattering */
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(&all_pixels[pixel_idx], image->p[i], 
                   image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
    }
    
    /* Allocate receive buffer for scatter operation */
    scattered_pixels = (pixel*)malloc(sendcount * sizeof(pixel));
    if (!scattered_pixels && sendcount > 0) {
        fprintf(stderr, "Process %d: Memory allocation failed for scattered_pixels\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    /* Use MPI_Scatterv to distribute pixels to all processes */
    MPI_Barrier(MPI_COMM_WORLD); // Synchronize before timing
    gettimeofday(&t1, NULL);
    
    /* Debugging info */
    if (rank == 0 && SOBELF_DEBUG) {
        printf("Scattering pixels: total images=%d\n", n_images);
        for (int i = 0; i < size; i++) {
            printf("Rank %d: images=%d, sendcount=%d\n", 
                  i, scatter_data->image_counts[i], 
                  scatter_data->sendcounts[i]);
        }
    }
    
    /* Use MPI_Scatterv to distribute pixels to all processes */
    MPI_Scatterv(all_pixels, scatter_data->sendcounts, scatter_data->displs, MPI_PIXEL,
                 scattered_pixels, sendcount, MPI_PIXEL,
                 0, MPI_COMM_WORLD);
    
    /* Copy scattered pixels to local image buffers */
    if (my_n_images > 0) {
        int pixel_offset = 0;
        for (int i = 0; i < my_n_images; i++) {
            int npixels = local_img.width[i] * local_img.height[i];
            memcpy(local_img.p[i], &scattered_pixels[pixel_offset], npixels * sizeof(pixel));
            pixel_offset += npixels;
        }
    }
    
    /* Free scattered pixels buffer, no longer needed */
    if (scattered_pixels) free(scattered_pixels);
    
    /* Apply filters to local images */
    apply_gray_filter(&local_img);
    apply_blur_filter(&local_img, 5, 20);
    apply_sobel_filter(&local_img);
    
    /* Prepare buffer for gathering results */
    pixel *gathered_pixels = NULL;
    if (sendcount > 0) {
        gathered_pixels = (pixel*)malloc(sendcount * sizeof(pixel));
        if (!gathered_pixels) {
            fprintf(stderr, "Process %d: Memory allocation failed for gathered_pixels\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        
        /* Copy processed pixels to contiguous buffer for gathering */
        int pixel_offset = 0;
        for (int i = 0; i < my_n_images; i++) {
            int npixels = local_img.width[i] * local_img.height[i];
            memcpy(&gathered_pixels[pixel_offset], local_img.p[i], npixels * sizeof(pixel));
            pixel_offset += npixels;
        }
    }
    
    /* Use MPI_Gatherv to collect results back to master */
    MPI_Gatherv(gathered_pixels, sendcount, MPI_PIXEL,
                all_pixels, scatter_data->sendcounts, scatter_data->displs, MPI_PIXEL,
                0, MPI_COMM_WORLD);
    
    /* Master copies results back to image structure */
    if (rank == 0) {
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(image->p[i], &all_pixels[pixel_idx], 
                   image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
        
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
        printf("SOBEL done in %lf s\n", duration);
    }
    
    /* Free local resources */
    if (gathered_pixels) free(gathered_pixels);
    for (int i = 0; i < my_n_images; i++) {
        if (local_img.p[i]) free(local_img.p[i]);
    }
    if (local_img.p) free(local_img.p);
    if (local_img.width) free(local_img.width);
    if (local_img.height) free(local_img.height);
    
    /* Master stores the results */
    if (rank == 0) {
        gettimeofday(&t1, NULL);
        if (!store_pixels(output_filename, image)) {
            fprintf(stderr, "Error writing output file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        gettimeofday(&t2, NULL);
        duration = (t2.tv_sec - t1.tv_sec) + ((t2.tv_usec - t1.tv_usec)/1e6);
        printf("Export done in %lf s in file %s\n", duration, output_filename);
        
        /* Free master resources */
        free_resources(image->p, image->n_images);
        free(image->width);
        free(image->height);
        free(image);
    }
    
    /* Free remaining resources */
    if (all_pixels) free(all_pixels);
    if (rank != 0) {
        free(image_widths);
        free(image_heights);
    }
    free_scatter_info(scatter_data);
    
    /* Free the MPI datatype */
    MPI_Type_free(&MPI_PIXEL);
    
    MPI_Finalize();
    return 0;
} 

