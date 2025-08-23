#include "conv.cuh"

#define a(_n, _x, _y, _c) a[(_n) * H * W * C + (_x) * W * C + (_y) * C + (_c)]
#define w(_k, _x, _y, _c) w[(_k) * R * S * C + (_x) * S * C + (_y) * C + (_c)]
#define b(_n, _x, _y, _k) b[(_n) * H * W * K + (_x) * W * K + (_y) * K + (_k)]

static constexpr int BLOCK = 16;  

// INT8 - Optimized with tiling and channel-wise DP4A
template <>
KernelConfig get_kernel_config<int8_t>() {
    KernelConfig config;
    // 增加K维度的并行，每个block处理4个输出通道
    constexpr int K_PER_BLOCK = 4;
    config.grid = dim3((H + BLOCK - 1) / BLOCK,     
                      (W + BLOCK - 1) / BLOCK, 
                      (K + K_PER_BLOCK - 1) / K_PER_BLOCK);
    config.block = dim3(BLOCK, BLOCK);
    // 计算共享内存大小：输入 tile (4个通道) + 权重 (4个通道 * K_PER_BLOCK)
    int padded_tile = BLOCK + R - 1;
    int input_shared_size = padded_tile * padded_tile * 4 * sizeof(int8_t);  // 4个通道
    int weight_shared_size = R * S * 4 * K_PER_BLOCK * sizeof(int8_t);  // 4个通道的权重 * K_PER_BLOCK
    config.shared_memory_size = input_shared_size + weight_shared_size;
    return config;
}

template <>
__global__ void conv2d_cuda_kernel<int8_t, int>(const int8_t *__restrict__ a, const int8_t *__restrict__ w,
                                                int8_t *__restrict__ b) {
    // 共享内存布局：输入数据 tile (4通道) + 权重 (4通道 * K_PER_BLOCK)
    extern __shared__ int8_t shared_memory[];
    const int padded_tile = BLOCK + R - 1;
    constexpr int K_PER_BLOCK = 4;
    int8_t* shared_input = shared_memory;  // [padded_tile][padded_tile][4]
    int8_t* shared_weights = shared_memory + padded_tile * padded_tile * 4;  // [K_PER_BLOCK][R][S][4]
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;  // 输出通道组索引
    const int tid = tx * BLOCK + ty;
    
    // 全局输出位置
    const int out_x = bx * BLOCK + tx;
    const int out_y = by * BLOCK + ty;
    
    // 计算当前block处理的输出通道范围
    const int k_start = bz * K_PER_BLOCK;
    const int k_end = min(K, k_start + K_PER_BLOCK);
    
    // 计算需要加载的输入数据范围（包含 padding）
    const int start_x = bx * BLOCK - R / 2;
    const int start_y = by * BLOCK - S / 2;
    
    // 计算通道处理的分组数量
    const int channel_groups = (C + 3) / 4;
    
    // 处理每个 batch
    for (int n = 0; n < N; ++n) {
        // 为当前block的每个输出通道初始化结果
        int results[K_PER_BLOCK];
        for (int k_idx = 0; k_idx < K_PER_BLOCK; ++k_idx) {
            results[k_idx] = 0;
        }
        
        // 以4个通道为一组处理输入通道
        for (int cg = 0; cg < channel_groups; ++cg) {
            int c_base = cg * 4;
            int channels_in_group = min(4, C - c_base);  // 当前组的实际通道数
            
            __syncthreads();
            
            // 并行加载当前4个通道的权重到共享内存（为所有K_PER_BLOCK个输出通道）
            for (int weight_idx = tid; weight_idx < R * S * 4 * K_PER_BLOCK; weight_idx += BLOCK * BLOCK) {
                int k_idx = weight_idx / (R * S * 4);  // 输出通道在block内的索引
                int temp = weight_idx % (R * S * 4);
                int r = temp / (S * 4);
                int temp2 = temp % (S * 4);
                int s = temp2 / 4;
                int c_offset = temp2 % 4;
                int c_actual = c_base + c_offset;
                int k_actual = k_start + k_idx;
                
                if (c_actual < C && k_actual < K) {
                    shared_weights[weight_idx] = w(k_actual, r, s, c_actual);
                } else {
                    shared_weights[weight_idx] = 0;  // 填充0
                }
            }
            
            __syncthreads();
            
            // 并行加载当前4个通道的输入数据到共享内存
            for (int load_idx = tid; 
                 load_idx < padded_tile * padded_tile * 4; 
                 load_idx += BLOCK * BLOCK) {
                
                int temp = load_idx;
                int c_offset = temp % 4;
                temp /= 4;
                int load_y = temp % padded_tile;
                int load_x = temp / padded_tile;
                
                int global_x = start_x + load_x;
                int global_y = start_y + load_y;
                int c_actual = c_base + c_offset;
                
                // 边界检查和加载数据
                if (global_x >= 0 && global_x < size && global_y >= 0 && global_y < size && c_actual < C) {
                    shared_input[load_idx] = a(n, global_x, global_y, c_actual);
                } else {
                    shared_input[load_idx] = 0;  // Zero padding
                }
            }
            
            __syncthreads();
            
            // 计算卷积，如果当前线程在有效输出范围内
            if (out_x < size && out_y < size) {
                // 为每个输出通道计算卷积
                for (int k_idx = 0; k_idx < k_end - k_start; ++k_idx) {
                    // 使用 DP4A 在通道维度上进行计算
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int shared_x = tx + r;
                            int shared_y = ty + s;
                            
                            // 获取4个通道的输入值和权重值
                            int input_base_idx = (shared_x * padded_tile + shared_y) * 4;
                            int weight_base_idx = (k_idx * R * S + r * S + s) * 4;
                            
                            if (channels_in_group == 4) {
                                // 完整的4个通道，使用DP4A
                                int input_packed = *reinterpret_cast<int*>(&shared_input[input_base_idx]);
                                int weight_packed = *reinterpret_cast<int*>(&shared_weights[weight_base_idx]);
                                results[k_idx] = __dp4a(input_packed, weight_packed, results[k_idx]);
                            } else {
                                // 不足4个通道，逐个计算
                                for (int c_offset = 0; c_offset < channels_in_group; ++c_offset) {
                                    int8_t input_val = shared_input[input_base_idx + c_offset];
                                    int8_t weight_val = shared_weights[weight_base_idx + c_offset];
                                    results[k_idx] += static_cast<int>(input_val) * static_cast<int>(weight_val);
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // 写入结果
        if (out_x < size && out_y < size) {
            for (int k_idx = 0; k_idx < k_end - k_start; ++k_idx) {
                int k_actual = k_start + k_idx;
                b(n, out_x, out_y, k_actual) = static_cast<int8_t>(results[k_idx]);
            }
        }
    }
}

// HALF - Optimized with tiling
template <>
KernelConfig get_kernel_config<half_t>() {
    KernelConfig config;
    config.grid = dim3((H + BLOCK - 1) / BLOCK, (W + BLOCK - 1) / BLOCK);
    config.block = dim3(BLOCK, BLOCK);
    // 计算共享内存大小：输入 tile + 权重
    int padded_tile = BLOCK + R - 1;
    int input_shared_size = padded_tile * padded_tile * sizeof(half_t);
    int weight_shared_size = R * S * C * sizeof(half_t);
    config.shared_memory_size = input_shared_size + weight_shared_size;
    return config;
}

template <>
__global__ void conv2d_cuda_kernel<half_t, float>(const half_t *__restrict__ a, const half_t *__restrict__ w,
                                                  half_t *__restrict__ b) {
    // 共享内存布局：输入数据 tile + 权重
    extern __shared__ half_t shared_memory_half[];
    const int padded_tile = BLOCK + R - 1;
    half_t* shared_input = shared_memory_half;
    half_t* shared_weights = shared_memory_half + padded_tile * padded_tile;
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tid = tx * BLOCK + ty;  // 线程在块内的唯一ID
    
    // 全局输出位置
    const int out_x = bx * BLOCK + tx;
    const int out_y = by * BLOCK + ty;
    
    // 计算需要加载的输入数据范围（包含 padding）
    const int start_x = bx * BLOCK - R / 2;
    const int start_y = by * BLOCK - S / 2;
    
    // 处理每个 batch 和每个输出通道
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            // 并行加载当前输出通道的所有权重到共享内存
            __syncthreads();
            for (int weight_idx = tid; weight_idx < R * S * C; weight_idx += BLOCK * BLOCK) {
                int r = weight_idx / (S * C);
                int temp = weight_idx % (S * C);
                int s = temp / C;
                int c = temp % C;
                shared_weights[weight_idx] = w(k, r, s, c);
            }
            __syncthreads();
            
            float result = 0;
            
            // 处理每个输入通道
            for (int c = 0; c < C; ++c) {
                __syncthreads();
                
                // 协作加载输入数据到共享内存
                for (int load_idx = tid; 
                     load_idx < padded_tile * padded_tile; 
                     load_idx += BLOCK * BLOCK) {
                    
                    int load_x = load_idx / padded_tile;
                    int load_y = load_idx % padded_tile;
                    int global_x = start_x + load_x;
                    int global_y = start_y + load_y;
                    
                    // 边界检查和加载数据
                    if (global_x >= 0 && global_x < size && global_y >= 0 && global_y < size) {
                        shared_input[load_idx] = a(n, global_x, global_y, c);
                    } else {
                        shared_input[load_idx] = static_cast<half_t>(0.0f);  // Zero padding
                    }
                }
                
                __syncthreads();
                
                // 计算卷积，如果当前线程在有效输出范围内
                if (out_x < size && out_y < size) {
                    for (int r = 0; r < R; ++r) {
                        for (int s = 0; s < S; ++s) {
                            int shared_x = tx + r;
                            int shared_y = ty + s;
                            half_t input_val = shared_input[shared_x * padded_tile + shared_y];
                            half_t weight_val = shared_weights[r * S * C + s * C + c];
                            result += static_cast<float>(input_val) * static_cast<float>(weight_val);
                        }
                    }
                }
            }
            
            // 写入结果
            if (out_x < size && out_y < size) {
                b(n, out_x, out_y, k) = static_cast<half_t>(result);
            }
        }
    }
}
