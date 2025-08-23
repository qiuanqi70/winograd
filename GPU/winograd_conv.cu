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
    // 3D thread mapping for better memory access
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nk_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    int tiles_x = (outW + 1) / 2;
    int tiles_y = (outH + 1) / 2;
    int total_spatial_tiles = tiles_x * tiles_y;
    
    // Check bounds
    if (spatial_idx >= total_spatial_tiles || nk_idx >= N * K) return;
    
    // Decompose indices
    int tile_y = spatial_idx / tiles_x;
    int tile_x = spatial_idx % tiles_x;
    int k = nk_idx % K;
    int n = nk_idx / K;

    float m[4][4] = {{0.0f}};

    // Loop over input channels
    for (int c = 0; c < C; ++c) {
        // --- Load Precomputed Filter Transform ---
        // Note: filter parameter now points to precomputed U matrix
        // U[k][c]
        const float* u_kc = filter + (k * C + c) * 16;
        
        // --- Image Transform ---
        int h_start = tile_y * 2;
        int w_start = tile_x * 2;
        float d[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                d[i][j] = image[(n * C + c) * H * W + (h_start + i) * W + (w_start + j)];
            }
        }
        float v_ncp[4][4];
        float temp_d[4][4];
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                temp_d[i][j] = B_T[i][0] * d[0][j] + B_T[i][1] * d[1][j] + B_T[i][2] * d[2][j] + B_T[i][3] * d[3][j];
            }
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                v_ncp[i][j] = temp_d[i][0] * B[0][j] + temp_d[i][1] * B[1][j] + temp_d[i][2] * B[2][j] + temp_d[i][3] * B[3][j];
            }
        }

        // --- Element-wise product and accumulate ---
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                m[i][j] += u_kc[i * 4 + j] * v_ncp[i][j];
            }
        }
    }

    // --- Output Transform ---
    float temp_m[2][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            temp_m[i][j] = A_T[i][0] * m[0][j] + A_T[i][1] * m[1][j] + A_T[i][2] * m[2][j] + A_T[i][3] * m[3][j];
        }
    }
    float Y[2][2];
    for (int i = 0; i < 2; ++i) {
        Y[i][0] = temp_m[i][0] + temp_m[i][1] + temp_m[i][2];
        Y[i][1] = temp_m[i][1] - temp_m[i][2] - temp_m[i][3];
    }

    // --- Write output ---
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int h = tile_y * 2 + i;
            int w = tile_x * 2 + j;
            if (h < outH && w < outW) {
                output[((n * K + k) * outH + h) * outW + w] = Y[i][j];
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
    
    // Step 2: 3D blocking for better memory access pattern
    int tiles_x = (outW + 1) / 2;  // Number of tiles in X direction
    int tiles_y = (outH + 1) / 2;  // Number of tiles in Y direction
    
    // Choose block dimensions that work well with tile distribution
    dim3 blockDim(16, 16, 1);  // 256 threads per block
    
    // Calculate grid dimensions to cover all (N, K, tile_y, tile_x) combinations
    int total_spatial_tiles = tiles_x * tiles_y;
    int total_nk = N * K;
    
    dim3 gridDim(
        (total_spatial_tiles + blockDim.x - 1) / blockDim.x,  // Spatial tiles dimension
        (total_nk + blockDim.y - 1) / blockDim.y,             // N*K dimension  
        1
    );

    winograd_conv_kernel<<<gridDim, blockDim>>>(
        image.data().get(), U.data().get(), out.data().get(),
        N, C, H, W, K, outH, outW
    );

    cudaDeviceSynchronize();
}