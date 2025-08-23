#include "winograd.cuh"

// Transformation matrices for F(2x2, 3x3)
__constant__ float G[4][3] = {
    {1.0f, 0.0f, 0.0f}, 
    {0.5f, 0.5f, 0.5f}, 
    {0.5f, -0.5f, 0.5f}, 
    {0.0f, 0.0f, 1.0f}
};

__constant__ float B_T[4][4] = {
    {1.0f, 0.0f, -1.0f, 0.0f}, 
    {0.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, -1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, 0.0f, -1.0f}
};

__constant__ float B[4][4] = {
    {1.0f,  0.0f,  0.0f,  0.0f}, 
    {0.0f,  1.0f, -1.0f,  1.0f}, 
    {-1.0f, 1.0f,  1.0f,  0.0f}, 
    {0.0f,  0.0f,  0.0f, -1.0f}
};

__constant__ float A_T[2][4] = {
    {1.0f, 1.0f, 1.0f, 0.0f}, 
    {0.0f, 1.0f, -1.0f, -1.0f}
};

// Kernel to precompute filter transformations
__global__
void filter_transform_kernel(const float* __restrict__ filter,
                             float* __restrict__ U,
                             int K, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_filters = K * C;
    if (idx >= total_filters) return;
    
    int k = idx / C;
    int c = idx % C;
    
    // Get pointer to the 3x3 filter for (k, c)
    const float* g = filter + (k * C + c) * 9;
    
    // Get pointer to output 4x4 transformed filter
    float* u_kc = U + (k * C + c) * 16;
    
    // Filter Transform: U = G * g * G^T
    float temp_g[4][3];
    
    // First step: temp_g = G * g
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            temp_g[i][j] = G[i][0] * g[0 * 3 + j] + G[i][1] * g[1 * 3 + j] + G[i][2] * g[2 * 3 + j];
        }
    }
    
    // Second step: u_kc = temp_g * G^T (manually computed G^T multiplication)
    for (int i = 0; i < 4; ++i) {
        u_kc[i * 4 + 0] = temp_g[i][0];
        u_kc[i * 4 + 1] = 0.5f * (temp_g[i][0] + temp_g[i][1] + temp_g[i][2]);
        u_kc[i * 4 + 2] = 0.5f * (temp_g[i][0] - temp_g[i][1] + temp_g[i][2]);
        u_kc[i * 4 + 3] = temp_g[i][2];
    }
}

// Fused kernel for Winograd convolution F(2x2, 3x3) using precomputed filter transforms
__global__
void winograd_conv_kernel(const float* __restrict__ image,
                          const float* __restrict__ filter,
                          float* __restrict__ output,
                          int N, int C, int H, int W, int K, int outH, int outW) {
    // Optimized 3D thread mapping: x=tile_x, y=tile_y, z=batch*output_channel
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    int nk_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    int tiles_x = (outW + 1) / 2;
    int tiles_y = (outH + 1) / 2;
    
    // Check bounds
    if (tile_x >= tiles_x || tile_y >= tiles_y || nk_idx >= N * K) return;
    
    // Decompose nk_idx into batch and output channel indices
    int k = nk_idx % K;
    int n = nk_idx / K;

    // Optimized: Use single accumulator array instead of m[4][4]
    float accumulator[16] = {0.0f};

    // Loop over input channels
    for (int c = 0; c < C; ++c) {
        // --- Load Precomputed Filter Transform ---
        const float* u_kc = filter + (k * C + c) * 16;
        
        // --- Image Transform (optimized to use less registers) ---
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        
        // Optimized: Reuse temp array for both intermediate steps
        float temp[16];
        
        // Step 1: Load input data and apply B_T transform
        // temp = B_T * d
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp[i * 4 + j] = 
                    B_T[i][0] * image[(n * C + c) * H * W + (h_start + 0) * W + (w_start + j)] +
                    B_T[i][1] * image[(n * C + c) * H * W + (h_start + 1) * W + (w_start + j)] +
                    B_T[i][2] * image[(n * C + c) * H * W + (h_start + 2) * W + (w_start + j)] +
                    B_T[i][3] * image[(n * C + c) * H * W + (h_start + 3) * W + (w_start + j)];
            }
        }
        
        // Step 2: Apply B transform and compute element-wise product
        // v = temp * B, then accumulate m += u * v
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                float v_val = 
                    temp[i * 4 + 0] * B[0][j] +
                    temp[i * 4 + 1] * B[1][j] +
                    temp[i * 4 + 2] * B[2][j] +
                    temp[i * 4 + 3] * B[3][j];
                
                accumulator[i * 4 + j] += u_kc[i * 4 + j] * v_val;
            }
        }
    }

    // --- Output Transform (optimized to use minimal registers) ---
    // Compute Y = A_T * accumulator * A
    // Step 1: temp = A_T * accumulator
    float temp_out[8]; // 2x4 result
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            temp_out[i * 4 + j] = 
                A_T[i][0] * accumulator[0 * 4 + j] +
                A_T[i][1] * accumulator[1 * 4 + j] +
                A_T[i][2] * accumulator[2 * 4 + j] +
                A_T[i][3] * accumulator[3 * 4 + j];
        }
    }
    
    // Step 2: Compute final output and write directly
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            float Y_val;
            if (j == 0) {
                Y_val = temp_out[i * 4 + 0] + temp_out[i * 4 + 1] + temp_out[i * 4 + 2];
            } else {
                Y_val = temp_out[i * 4 + 1] - temp_out[i * 4 + 2] - temp_out[i * 4 + 3];
            }
            
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[((n * K + k) * outH + h) * outW + w] = Y_val;
            }
        }
    }
}

void winograd_conv(thrust::device_vector<float>& image,
                   thrust::device_vector<float>& filter, 
                   thrust::device_vector<float>& out,
                   thrust::device_vector<float>& U,
                   thrust::device_vector<float>& V, 
                   thrust::device_vector<float>& M,
                   int H, int W, int C, int K, int N) {
    const int outH = H - 2;
    const int outW = W - 2;
    
    // Step 1: Precompute filter transformations
    const int threads_per_block_filter = 256;
    int total_filters = K * C;
    int grid_size_filter = (total_filters + threads_per_block_filter - 1) / threads_per_block_filter;
    
    filter_transform_kernel<<<grid_size_filter, threads_per_block_filter>>>(
        filter.data().get(), U.data().get(), K, C
    );
    
    // Step 2: Optimized 3D blocking for better memory access pattern
    int tiles_x = (outW + 1) / 2;  // Number of tiles in X direction
    int tiles_y = (outH + 1) / 2;  // Number of tiles in Y direction
    int total_nk = N * K;          // Total batch * output_channel combinations
    
    // Choose block dimensions that work well with new mapping
    // blockDim.x = tile_x, blockDim.y = tile_y, blockDim.z = batch*output_channel
    dim3 blockDim(8, 8, 8);  // 512 threads per block, good for occupancy
    
    // Calculate grid dimensions to cover all (tile_x, tile_y, N*K) combinations
    dim3 gridDim(
        (tiles_x + blockDim.x - 1) / blockDim.x,  // Tile X dimension
        (tiles_y + blockDim.y - 1) / blockDim.y,  // Tile Y dimension  
        (total_nk + blockDim.z - 1) / blockDim.z  // Batch * Output channel dimension
    );

    winograd_conv_kernel<<<gridDim, blockDim>>>(
        image.data().get(), U.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}