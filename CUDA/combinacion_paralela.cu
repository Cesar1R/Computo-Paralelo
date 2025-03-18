%%writefile combinacion_paralelo.cu
#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>


__global__ void combinacion_img(const uchar3* imagen, const uchar3* fondo, const float* alpha, uchar3* res, int imgW, int imgH) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgW && y < imgH) {
        int idx = y * imgW + x;
        uchar3 a = imagen[idx];
        uchar3 b = fondo[idx];

        float alpha_v = alpha[idx];

        res[idx].x = a.x * alpha_v + b.x * (1.0f - alpha_v);
        res[idx].y = a.y * alpha_v + b.y * (1.0f - alpha_v);
        res[idx].z = a.z * alpha_v + b.z * (1.0f - alpha_v);
    }
}

int main() {
    cv::Mat fondo = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/fondo.bmp", cv::IMREAD_COLOR);
    cv::Mat imagen = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/greenscreen.bmp", cv::IMREAD_COLOR);
    cv::Mat imgalpha = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/greenscreenMask.bmp", cv::IMREAD_GRAYSCALE);

        //Verificamos que no estn vacias 
    if (imagen.empty() || fondo.empty() || imgalpha.empty()) {
        std::cerr << "Error al cargar las imágenes" << std::endl;
        return -1;
    }

    cv::Mat alpha;
    imgalpha.convertTo(alpha, CV_32FC1, 1.0 / 255);

    //Creamos el contenedor del resultado

    cv::Mat res = cv::Mat::zeros(imagen.size(), imagen.type());

    int imgW = imagen.cols;
    int imgH = imagen.rows;

    size_t imgSize = imgW * imgH * sizeof(uchar3);
    size_t alphaSize = imgW * imgH * sizeof(float);

    uchar3* d_imagen; 
    uchar3* d_fondo; 
    uchar3*  d_res;
    float* d_alpha;

    //-----Gestion de memoria del dispotivo 
    cudaMalloc(&d_imagen, imgSize);
    cudaMalloc(&d_fondo, imgSize);
    cudaMalloc(&d_res, imgSize);
    cudaMalloc(&d_alpha, alphaSize);

    //Comunicacion entre el dispositivo y el host 
    cudaMemcpy(d_imagen, imagen.ptr<uchar3>(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fondo, fondo.ptr<uchar3>(), imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha.ptr<float>(), alphaSize, cudaMemcpyHostToDevice);


    //Caracteristicas de la red
    dim3 block(16, 16);
    dim3 grid((imgW + block.x - 1) / block.x, (imgH + block.y - 1) / block.y);

    combinacion_img<<<grid, block>>>(d_imagen, d_fondo, d_alpha, d_res, imgW, imgH);

    //Accion para colab
    cudaDeviceSynchronize();

    //Comunicacion entre el host y el dispositivo. 
    cudaMemcpy(res.ptr<uchar3>(), d_res, imgSize, cudaMemcpyDeviceToHost);


    //---Gestion de memoria
    cudaFree(d_alpha);
    cudaFree(d_fondo);
    cudaFree(d_imagen);
    cudaFree(d_res);

    cv::imwrite("imagen_combinada_paralelo.bmp", res);

    return 0;
}
