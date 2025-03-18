%%writefile imagenMG.cpp 

#include <opencv2/opencv.hpp>
#include <cmath>

int main(){

    cv::Mat img = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/pinzas_gray.png", cv::IMREAD_GRAYSCALE); 

    if(img.empty()){
        std::cerr <<"Imagen vacia" << std::endl; 
        return -1; 
    }

    //Matrices correspondientes a los filtros
    cv::Mat K1 = (cv::Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
    cv::Mat K2 = (cv::Mat_<float>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

    cv::Mat imgX, imgY; 

    //Aplicamos los filtros y recueramos el resultado
    cv::filter2D(img, imgX, CV_32F, K1);
    cv::filter2D(img, imgY, CV_32F, K2);

    cv::Mat MG = cv::Mat::zeros(img.size(), CV_8U);


    //Aplicamos la primer transformacion para obtener la imagen MG
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            float gx = imgX.at<float>(y, x);
            float gy = imgY.at<float>(y, x);
            MG.at<uchar>(y, x) = static_cast<uchar>(cv::sqrt(gx*gx + gy*gy));
        }
    }

    cv::normalize(MG, MG, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("imagen_MG.bmp", MG);

    

    cv::Mat MGT = cv::Mat::zeros(img.size(), CV_8U);

    //Definimos el umbral
    int T = 100; 
    //Aplicamos la segunda transformacion 
    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            float comp = MG.at<uchar>(y, x);
            MGT.at<uchar>(y, x) = static_cast<uchar>((comp > T) ? 255 : 0);
        }
    }


    cv::normalize(MGT, MGT, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite("imagen_MGT.bmp", MGT);

    return 0;

}