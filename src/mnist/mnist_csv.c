#include "mnist_csv.h"

#define INITIAL_CAPACITY 10000

MNISTData* load_mnist_csv(const char* images_path, const char* labels_path) {
    FILE* images_file = fopen(images_path, "r");
    if (!images_file) {
        printf("Error: Could not open images file: %s\n", images_path);
        return NULL;
    }

    char line[4096];  // Buffer for reading lines
    // Skip header line
    if (!fgets(line, sizeof(line), images_file)) {
        printf("Error: Could not read header from %s\n", images_path);
        fclose(images_file);
        return NULL;
    }

    int capacity = INITIAL_CAPACITY;
    int num_images = 0;
    int image_size = 784;
    double* images = (double*)malloc(capacity * image_size * sizeof(double));
    int* labels = (int*)malloc(capacity * sizeof(int));
    if (!images || !labels) {
        printf("Error: Failed to allocate memory for images or labels\n");
        if (images) free(images);
        if (labels) free(labels);
        fclose(images_file);
        return NULL;
    }

    while (fgets(line, sizeof(line), images_file)) {
        // Skip blank lines
        char* p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r') continue;

        if (num_images >= capacity) {
            capacity *= 2;
            images = (double*)realloc(images, capacity * image_size * sizeof(double));
            labels = (int*)realloc(labels, capacity * sizeof(int));
            if (!images || !labels) {
                printf("Error: Failed to reallocate memory\n");
                if (images) free(images);
                if (labels) free(labels);
                fclose(images_file);
                return NULL;
            }
        }

        // Parse CSV line
        char* token = strtok(line, ",");
        if (!token) continue; // skip malformed lines
        labels[num_images] = atoi(token);  // First value is the label

        int j;
        for (j = 0; j < image_size; j++) {
            token = strtok(NULL, ",");
            if (!token) break;
            images[num_images * image_size + j] = atof(token) / 255.0;
        }
        if (j != image_size) {
            printf("Warning: Skipping line %d due to insufficient pixel data\n", num_images+2);
            continue;
        }
        num_images++;
    }

    fclose(images_file);

    // Shrink to fit
    images = (double*)realloc(images, num_images * image_size * sizeof(double));
    labels = (int*)realloc(labels, num_images * sizeof(int));

    MNISTData* data = (MNISTData*)malloc(sizeof(MNISTData));
    if (!data) {
        printf("Error: Failed to allocate MNISTData\n");
        free(images);
        free(labels);
        return NULL;
    }
    data->num_images = num_images;
    data->image_size = image_size;
    data->images = images;
    data->labels = labels;
    return data;
}

void free_mnist_data(MNISTData* data) {
    if (data) {
        if (data->images) free(data->images);
        if (data->labels) free(data->labels);
        free(data);
    }
} 