#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>

#define N 8192
#define M 8192

#define TPB 16
#define ATOMIC 1

using namespace std;

void fill_mat (vector<float> & mat){

    static random_device rd;
    static mt19937 gen(rd());

    static uniform_real_distribution<> distr(-100.0, 100.0); 


    size_t i;

    for(i = 0; i < mat.size(); i++){
        mat[i] = distr(gen);        
    } 
}


inline float sq (float k){
    return k * k; 
}

__global__ void matrix_error_Global(const float *A, 
                                     const float *B, 
                                     float *sum, 
                                     int N, 
                                     int M)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x, 
        idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if(idx < N && idy < M){
        int index = idx * M + idy;
        atomicAdd(sum, sq(A[index] - B[index]));
    }

}

__global__ void matrix_error_Shared(const float *d_A, 
                                     const float *d_B, 
                                     float *sum, 
                                     int N, 
                                     int M)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x, 
        idy = threadIdx.y + blockIdx.y * blockDim.y;
    
    int local_idx = threadIdx.x, 
        local_idy = threadIdx.y;
    
    __shared__ float s_A[16][16]; 
    __shared__ float s_B[16][16]; 

    if (idx <  N && idy < M){
        s_A[local_idx][local_idy] = d_A[idx * M + idy]; 
        s_B[local_idx][local_idy] = d_B[idx * M + idy]; 
    } else {
        s_A[local_idx][local_idy] = 0.0; 
        s_B[local_idx][local_idy] = 0.0; 
    }

    __syncthreads(); 

    float local_sum = 0.0; 
    if(idx < N && idy < M){
        local_sum = sq(s_A[local_idx][local_idy] - s_B[local_idx][local_idy]); 
    }

    atomicAdd(sum, local_sum); 
}

int main(void){
    //Matrices que se alojan en el host
    vector<float> h_A(N * M),
                  h_B(N * M);
    
    fill_mat(h_A);
    fill_mat(h_B);

    float *d_A, *d_B, *d_sum;
    size_t size = N * M * sizeof(float);

    //---------------------Inicio: Implementacion con memoria global
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    cudaMemset(d_sum, 0, sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+threadsPerBlock.x -1) / threadsPerBlock.x, 
                        (M+threadsPerBlock.y -1)/ threadsPerBlock.y);

    matrix_error_Global<<<blocksPerGrid, threadsPerBlock>>> (d_A, d_B, N, M);

    float h_sum  = 0; 
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Error en ejecucion con memoria global: " << h_sum << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_sum);

    //---------------------Fin: Implementacion con memoria global 
   //---------------------Inicio: Implementacion con memoria global 
    cudaMalloc(&d_A, size); 
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_sum, sizeof(float));  

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    cudaMemset(d_sum, 0, sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N+threadsPerBlock.x -1) / threadsPerBlock.x, 
                        (M+threadsPerBlock.y -1)/ threadsPerBlock.y);

    matrix_error_Shared<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_sum, N, M); 

    float h_sum = 0.0; 
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost); 

    cout << "Error en ejecucion con memoria compartida: " << h_sum << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_sum);

    //---------------------Fin: Implementacion con memoria compartida

    return 0;
}