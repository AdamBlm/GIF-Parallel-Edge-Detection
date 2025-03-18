    /* Use MPI_Gatherv with MPI_BYTE for more efficient data collection */
    MPI_Gatherv(gathered_pixels, sendcount * sizeof(pixel), MPI_BYTE,
                all_pixels, scatter_data->scatter_byte_counts, scatter_data->scatter_byte_displs, MPI_BYTE,
                0, MPI_COMM_WORLD);
                
    /* Copy processed data back to the original image structure */
    if (rank == 0 && all_pixels != NULL) {
        /* Copy all processed pixels back to the original image structure */
        int pixel_idx = 0;
        for (int i = 0; i < n_images; i++) {
            memcpy(image->p[i], &all_pixels[pixel_idx], 
                  image_widths[i] * image_heights[i] * sizeof(pixel));
            pixel_idx += image_widths[i] * image_heights[i];
        }
    }
                
    /* FILTER Timer stop */ 
