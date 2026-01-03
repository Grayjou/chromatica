// brick_stack.h
#ifndef BRICK_STACK_H
#define BRICK_STACK_H

#include <stdint.h>

// Structure to hold brick stack interval data
typedef struct {
    double* y_boundaries;      // Row boundaries [0, ..., 1]
    int num_rows;              // Number of rows (num_y_boundaries - 1)
    
    double** x_boundaries;     // Array of x-boundary arrays per row
    int* num_bricks_per_row;   // Number of bricks in each row
} BrickStack;

// Result for a single point lookup
typedef struct {
    int brick_idx;
    int row_idx;
    double rel_x;
    double rel_y;
} BrickLookupResult;

// Initialize brick stack from interval data
BrickStack* brick_stack_create(
    const double* y_bounds, int num_y_bounds,
    const double** x_bounds_per_row, const int* num_x_bounds_per_row
);

// Free brick stack memory
void brick_stack_free(BrickStack* stack);

// Lookup single point
BrickLookupResult brick_stack_lookup(const BrickStack* stack, double x, double y);

// Process entire WxHx2 position matrix
void brick_stack_process_matrix(
    const BrickStack* stack,
    const double* positions,    // Flattened WxHx2: [x0,y0, x1,y1, ...]
    int W, int H,
    int32_t* out_pertinency,    // Flattened WxHx2: [brick0,row0, brick1,row1, ...]
    double* out_relative        // Flattened WxHx2: [relx0,rely0, relx1,rely1, ...]
);

#endif // BRICK_STACK_H