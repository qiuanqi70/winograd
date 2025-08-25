#include "winograd.h"
#include <arm_neon.h>
#include <omp.h>

// Transformation matrices for Winograd F(2x2, 3x3)
const float G[4][3] = {
    {1.0, 0.0, 0.0}, 
    {0.5, 0.5, 0.5}, 
    {0.5, -0.5, 0.5}, 
    {0.0, 0.0, 1.0}
};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, 
    {0.0, 0.5, -0.5, 0.0}, 
    {0.0, 0.5, 0.5, 1.0}
};
const float B[4][4] = {
    {1, 0, 0, 0}, 
    {0, 1, -1, 1}, 
    {-1, 1, 1, 0}, 
    {0, 0, 0, -1}
};
const float B_T[4][4] = {
    {1, 0, -1, 0}, 
    {0, 1, 1, 0}, 
    {0, -1, 1, 0}, 
    {0, 1, 0, -1}
};
const float A[4][2] = {
    {1, 0}, 
    {1, 1}, 
    {1, -1}, 
    {0, -1}
};
const float A_T[2][4] = {
    {1, 1, 1, 0}, 
    {0, 1, -1, -1}
};

void sgemm_neon(const float* A, const float* B, float* out, int M, int K, int N) {
    int j;
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for(j = 0; j < N; j+=4) {
            float32x4_t acc = vdupq_n_f32(0.0f);
            for (int k = 0; k < K; ++k) {
                float32x4_t b_vec = vld1q_f32(B + k * N +j); // 加载B的4个元素
                float32x4_t a_val = vdupq_n_f32(A[i * K + k]); // A的一个元素广播
                acc = vmlaq_f32(acc, a_val, b_vec); // acc += a_val * b_vec
            }
            vst1q_f32(out + i * N + j, acc); // 存回结果
        }
        //尾处理
        #pragma omp parallel for
        for (int jj = j; jj < N; ++jj) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + jj];
            out[i * N + jj] = sum;
        }
    }
}

/** 
 * @brief Winograd Implementation of F(2x2, 3x3)
 * @param image: [batch * C * inHeight * inWidth]
 * @param filter: [K * C * 3 * 3]
 * @param out: Output result. Shape is [batch * K * outHeight * outWidth]
 * @param U: [4 * 4 * K * C], intermediate transformed filters
 * @param V: [4 * 4 * C * P], intermediate transformed image
 * @param M: [4 * 4 * K * P], intermediate result
 * @param inHeight: Height of input image
 * @param inWidth: Width of input image
 * @param C: Number of channels in input image
 * @param K: Number of filters
 * @param N: Batch size
 */
void winograd_conv(const float* restrict image, const float* restrict filter, float* restrict out,
                   float* restrict U, float* restrict V, float* restrict M,
                   const int inHeight, const int inWidth, const int C, const int K, const int N) {
    const int outHeight = inHeight - 2; // output height
    const int outWidth = inWidth - 2; // output width
    const int sizeI = inHeight * inWidth; // size of input image
    const int sizeF = 3 * 3; // size of filter
    const int sizeO = outHeight * outWidth; // size of output
    const int P = outHeight / 2 * outWidth / 2 * N; // size of output in blocks

    //将filter变换和image变换改为任务并行
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            float tmp_u[12]; // 4 * 3
            float u[16];     // 4 * 4;
            
            // Transform filters and scatter to U
            // U[:, :, k, c] = G * filters[k, c, :, :] * G^T
            #pragma omp parallel for private(tmp_u, u)
            for (int k = 0; k < K; ++k) {
                for (int c = 0; c < C; ++c) {
                    const float* filters_ptr = filter + (k * C + c) * sizeF;
                    
                    for(int i=0;i<3;i++){
                        tmp_u[i]=filters_ptr[i];
                        tmp_u[3+i]=0.5*(filters_ptr[i]+filters_ptr[3+i]+filters_ptr[3*2+i]);
                        tmp_u[3*2+i]=0.5*(filters_ptr[i]-filters_ptr[3+i]+filters_ptr[3*2+i]);
                        tmp_u[3*3+i]=filters_ptr[2*3+i];
                    }
                    for(int i=0;i<4;i++){
                        u[4*i]=tmp_u[3*i];
                        u[4*i+1]=0.5*(tmp_u[3*i]+tmp_u[3*i+1]+tmp_u[3*i+2]);
                        u[4*i+2]=0.5*(tmp_u[3*i]-tmp_u[3*i+1]+tmp_u[3*i+2]);
                        u[4*i+3]=tmp_u[3*i+2];
                    }
                    
                    for (int xi = 0; xi < 4; ++xi) {
                        for (int nu = 0; nu < 4; ++nu) {
                            U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu];
                        }
                    }
                }
            }
        }
        #pragma omp section
        {
            // Transform image and scatter to V
            // V[:, :, c, p] = B^T * image[c, b, :, :] * B
            float tmp_v[16];
            float d[16]; // d: [4 * 4];
            float v[16]; // v: [4 * 4];
            #pragma omp parallel for collapse(2) private(tmp_v, d, v)
            for (int n = 0; n < N; ++n) {
                for (int c = 0; c < C; ++c) {
                    for (int y = 0; y < outHeight / 2; ++y) {
                        for (int x = 0; x < outWidth / 2; ++x) {
                            // Generate d_cb
                            for (int iy = 0; iy < 4; ++iy) {
                                // 使用NEON加载4个连续的float值
                                float32x4_t vec = vld1q_f32(image + (n * C + c) * sizeI + (y * 2 + iy) * inWidth + (x * 2));
                                // 将向量数据存储到目标数组
                                vst1q_f32(d + iy * 4, vec);
                            }

                            for(int i=0;i<4;i++){
                                tmp_v[i]=d[i]-d[4*2+i];
                                tmp_v[4+i]=d[4+i]+d[4*2+i];
                                tmp_v[4*2+i]=-d[4+i]+d[4*2+i];
                                tmp_v[4*3+i]=d[4+i]-d[4*3+i];
                            }
                            for(int i=0;i<4;i++){
                                v[4*i]=tmp_v[4*i]-tmp_v[4*i+2];
                                v[4*i+1]=tmp_v[4*i+1]+tmp_v[4*i+2];
                                v[4*i+2]=-tmp_v[4*i+1]+tmp_v[4*i+2];
                                v[4*i+3]=tmp_v[4*i+1]-tmp_v[4*i+3];
                            }
                            
                            int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
                            for (int xi = 0; xi < 4; ++xi) {
                                for (int nu = 0; nu < 4; ++nu) {
                                    V[((xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu];
                                }
                            }
                        }
                    }    
                }
            }
        }
    }

    // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int xi = 0; xi < 4; ++xi) {
        for (int nu = 0; nu < 4; ++nu) {
            float* M_ptr = M + (xi * 4 + nu) * K * P;
            float* U_ptr = U + (xi * 4 + nu) * K * C;
            float* V_ptr = V + (xi * 4 + nu) * C * P;
            sgemm_neon(U_ptr, V_ptr, M_ptr, K, C, P);
        }
    }

    // Gather output and apply inverse transformation
    // Y = A^T * m * A
    float mm[16];      // 4 * 4
    float tmp_m[8];    // 2 * 4
    float temp_out[4]; // 2 * 2
    #pragma omp parallel for collapse(2) private(mm, temp_out, tmp_m)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
                    for (int xi = 0; xi < 4; ++xi) {
                        for (int nu = 0; nu < 4; ++nu) {
                            mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
                        }
                    }
                    
                    for(int i=0;i<4;i++){
                        tmp_m[i]=mm[i]+mm[4+i]+mm[4*2+i];
                        tmp_m[4+i]=mm[4+i]-mm[4*2+i]-mm[4*3+i];
                    }
                    for(int i=0;i<2;i++){
                        temp_out[i*2]=tmp_m[4*i]+tmp_m[4*i+1]+tmp_m[4*i+2];
                        temp_out[2*i+1]=tmp_m[4*i+1]-tmp_m[4*i+2]-tmp_m[4*i+3];
                    }
                    
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            out[((n * K + k) * outHeight + y * 2 + i) * outWidth +
                                x * 2 + j] = temp_out[i * 2 + j];
                        }
                    }
                }
            }
        }
    }
}