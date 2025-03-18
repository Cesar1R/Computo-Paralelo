#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <random>

#define N_SIZE 1024
#define M_SIZE 1024


using namespace std;

void fill_mat (vector<float> & mat){

    static random_device rd;
    static mt19937 gen(rd());

    static uniform_real_distribution<> distr(-1.0, 1.0); 


    size_t i;

    for(i = 0; i < mat.size(); i++){
        mat[i] = distr(gen);        
    } 
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
        atomicAdd(sum, (A[index] - B[index]) * (A[index] - B[index]));
    }

}



int main(void){
    //Matrices que se alojan en el host
    vector<float> h_A(N_SIZE * M_SIZE),
                  h_B(N_SIZE * M_SIZE);
    
    fill_mat(h_A);
    fill_mat(h_B);

    float *d_A, *d_B;
    float *d_sum;
    size_t size = N_SIZE * M_SIZE * sizeof(float);

    //---------------------Inicio: Implementacion con memoria global
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_A, h_A.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size, cudaMemcpyHostToDevice);

    cudaMemset(d_sum, 0, sizeof(float));

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N_SIZE+threadsPerBlock.x -1) / threadsPerBlock.x, 
                        (M_SIZE+threadsPerBlock.y -1)/ threadsPerBlock.y);

    matrix_error_Global<<<blocksPerGrid, threadsPerBlock>>> (d_A, d_B, d_sum,N_SIZE, M_SIZE);

    float h_sum  = 0; 
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);

    cout << "Error en ejecucion con memoria global: " << h_sum << endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_sum);

    //---------------------Fin: Implementacion con memoria global 


    return 0;
}