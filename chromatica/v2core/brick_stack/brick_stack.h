// brick_stack.h
#ifndef BRICK_STACK_H
#define BRICK_STACK_H

#include <stdint.h>

typedef struct {
    double start;
    double end;
    int brick_count;
    double* brick_ends;  // Cumulative lengths up to 1.0
} BrickRow;

typedef struct {
    int row_count;
    BrickRow* rows;
    double* row_ends;  // Cumulative heights up to 1.0
} BrickStack;

typedef struct {
    int row_idx;
    int brick_idx;
    double rel_x;
    double rel_y;
} BrickPosition;

// Create a brick stack from partition data
BrickStack* create_brick_stack(int row_count, double* row_ends, 
                               int* brick_counts, double** brick_arrays);

// Free brick stack memory
void free_brick_stack(BrickStack* stack);

// Find brick position for normalized coordinates
BrickPosition find_brick_position(const BrickStack* stack, double x, double y);

// Process scrambled matrix in parallel
void process_scrambled_matrix(const BrickStack* stack, 
                             double* scrambled_coords,  // WxHx2
                             int width, int height,
                             int* brick_indices,        // WxHx2 output
                             double* rel_positions);    // WxHx2 output

#endif