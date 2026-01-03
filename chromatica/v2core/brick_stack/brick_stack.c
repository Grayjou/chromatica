// brick_stack.c
#include "brick_stack.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>

BrickStack* create_brick_stack(int row_count, double* row_ends,
                              int* brick_counts, double** brick_arrays) {
    BrickStack* stack = malloc(sizeof(BrickStack));
    stack->row_count = row_count;
    stack->rows = malloc(row_count * sizeof(BrickRow));
    stack->row_ends = malloc(row_count * sizeof(double));
    
    memcpy(stack->row_ends, row_ends, row_count * sizeof(double));
    
    for (int i = 0; i < row_count; i++) {
        stack->rows[i].start = (i == 0) ? 0.0 : row_ends[i-1];
        stack->rows[i].end = row_ends[i];
        stack->rows[i].brick_count = brick_counts[i];
        stack->rows[i].brick_ends = malloc(brick_counts[i] * sizeof(double));
        memcpy(stack->rows[i].brick_ends, brick_arrays[i], 
               brick_counts[i] * sizeof(double));
    }
    
    return stack;
}

void free_brick_stack(BrickStack* stack) {
    for (int i = 0; i < stack->row_count; i++) {
        free(stack->rows[i].brick_ends);
    }
    free(stack->rows);
    free(stack->row_ends);
    free(stack);
}

BrickPosition find_brick_position(const BrickStack* stack, double x, double y) {
    BrickPosition pos = {-1, -1, 0.0, 0.0};
    
    // Find row
    for (int i = 0; i < stack->row_count; i++) {
        if (y < stack->row_ends[i] + 1e-12) {
            pos.row_idx = i;
            pos.rel_y = (y - stack->rows[i].start) / 
                       (stack->rows[i].end - stack->rows[i].start);
            break;
        }
    }
    
    if (pos.row_idx == -1) return pos;
    
    // Find brick in row
    BrickRow* row = &stack->rows[pos.row_idx];
    for (int j = 0; j < row->brick_count; j++) {
        if (x < row->brick_ends[j] + 1e-12) {
            pos.brick_idx = j;
            double brick_start = (j == 0) ? 0.0 : row->brick_ends[j-1];
            pos.rel_x = (x - brick_start) / 
                       (row->brick_ends[j] - brick_start);
            break;
        }
    }
    
    return pos;
}

// SIMD/parallel implementation for processing matrix
#ifdef _OPENMP
#include <omp.h>
#endif

void process_scrambled_matrix(const BrickStack* stack,
                             double* scrambled_coords,
                             int width, int height,
                             int* brick_indices,
                             double* rel_positions) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int idx = (i * width + j) * 2;
            double x = scrambled_coords[idx];
            double y = scrambled_coords[idx + 1];
            
            BrickPosition pos = find_brick_position(stack, x, y);
            
            // Store results
            brick_indices[idx] = pos.brick_idx;
            brick_indices[idx + 1] = pos.row_idx;
            rel_positions[idx] = pos.rel_x;
            rel_positions[idx + 1] = pos.rel_y;
        }
    }
}