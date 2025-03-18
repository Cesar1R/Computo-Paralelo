%%writefile imagenMG_paralelo.cu

#include <opencv2/opencv.hpp>
#include <cmath>
#include <cuda_runtime.h>


__global__ void get_img(const unsigned char * d_img, float * f1, float *f2, int imgW, int imgH, unsigned char * d_MG, unsigned char * d_MGT, const float* filtro1, const float* filtro2, int filtroW, int filtroH, int T) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < imgW && y < imgH) {
        float sum1 = 0.0f, sum2 = 0.0f;

        for(int i = -filtroH/2; i <= filtroH/2; i++) {
            for(int j = -filtroW/2; j <= filtroW/2; j++) {
                int imgX = min(max(x + j, 0), imgW - 1); 
                int imgY = min(max(y + i, 0), imgH - 1); 
                float imgValue = static_cast<float>(d_img[imgY * imgW + imgX]);
                sum1 += imgValue * filtro1[(i + filtroH/2) * filtroW + (j + filtroW/2)];
                sum2 += imgValue * filtro2[(i + filtroH/2) * filtroW + (j + filtroW/2)];
            }
        }
        f1[y * imgW + x] = sum1; f2[y * imgW + x] = sum2;
        
        float gx = sum1, gy = sum2;

        float grad_magnitude = sqrtf(gx * gx + gy * gy);
        
        d_MG[y * imgW + x] = static_cast<unsigned char>(grad_magnitude);
        d_MGT[y * imgW + x] = (grad_magnitude > T) ? 255 : 0;
    }
}



int main() {
    cv::Mat img = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/pinzas_gray.png", cv::IMREAD_GRAYSCALE); 

    if(img.empty()) {
        std::cerr << "Imagen vacia" << std::endl; 
        return -1; 
    }

    //Creamos las matrices correspondientes a los ciclos
    float K1[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1}; 
    float K2[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1}; 

    //Reservamos las matrices del tipo Mat para guardar las imagenes 
    cv::Mat res_K1(img.size(), CV_32FC1);
    cv::Mat res_K2(img.size(), CV_32FC1);
    cv::Mat MG(img.size(), CV_8UC1);
    cv::Mat MGT(img.size(), CV_8UC1);

    //Apuntadores para la memoria de los dispositivos 
    unsigned char *d_img, *d_MG, *d_MGT;
    float *d_res_K1, *d_res_K2;
    float *d_K1, *d_K2;


    //Informacion sobre el size de las imagenes y los filtros
    int imgW = img.cols;
    int imgH = img.rows;
    size_t imgSize = imgW * imgH * sizeof(unsigned char);
    size_t filtroSize = 9 * sizeof(float); 


    //Reserva de memoria en los dispositivos
    cudaMalloc(&d_img, imgSize);
    cudaMalloc(&d_res_K1, imgW * imgH * sizeof(float));
    cudaMalloc(&d_res_K2, imgW * imgH * sizeof(float));
    cudaMalloc(&d_MG, imgW * imgH * sizeof(unsigned char));
    cudaMalloc(&d_MGT, imgW * imgH * sizeof(unsigned char));
    cudaMalloc(&d_K1, filtroSize);
    cudaMalloc(&d_K2, filtroSize);

    //Enviamos la informacion a los dispositivos 
    cudaMemcpy(d_img, img.data, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K1, K1, filtroSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K2, K2, filtroSize, cudaMemcpyHostToDevice);

    //Caracteristicas de la red 
    dim3 block(16, 16);
    dim3 grid((imgW + block.x - 1) / block.x, (imgH + block.y - 1) / block.y);
    
    //Llamada a la funcion 
    int T = 100;
    get_img<<<grid, block>>>(d_img, d_res_K1, d_res_K2, imgW, imgH, d_MG, d_MGT, d_K1, d_K2, 3, 3, T);

    //Accion para colab
    cudaDeviceSynchronize();


    //Recuperamos la informacion de inters
    cudaMemcpy(res_K1.ptr<float>(), d_res_K1, imgW * imgH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(res_K2.ptr<float>(), d_res_K2, imgW * imgH * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(MG.ptr<unsigned char>(), d_MG, imgW * imgH * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(MGT.ptr<unsigned char>(), d_MGT, imgW * imgH * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    //Liberamos memoria
    cudaFree(d_img);
    cudaFree(d_res_K1);
    cudaFree(d_res_K2);
    cudaFree(d_MG);
    cudaFree(d_MGT);
    cudaFree(d_K1);
    cudaFree(d_K2);


    //Guardamos las imagenes de interes
    cv::normalize(MG, MG, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("imagen_MG_paralelo.bmp", MG);
    cv::imwrite("imagen_MGT_paralelo.bmp", MGT);
    return 0;
}
