%%writefile combinacion_secuencial.cpp
#include <opencv2/opencv.hpp>
#include <iostream>

using cv::imread;
using cv::IMREAD_COLOR;

int main() {
    cv::Mat fondo = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/fondo.bmp", cv::IMREAD_COLOR);
    cv::Mat imagen = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/greenscreen.bmp", cv::IMREAD_COLOR);
    cv::Mat mask = cv::imread("/content/drive/MyDrive/Demat6/Computo Paralelo/Tarea4/greenscreenMask.bmp", cv::IMREAD_GRAYSCALE);

    //Verificamos que no estn vacias 
    if (fondo.empty() || imagen.empty() || mask.empty()) {
        std::cerr << "Error al cargar las imágenes" << std::endl;
        return -1;
    }

    //Creamos el contenedor del resultado
    cv::Mat imgResult = cv::Mat::zeros(imagen.size(), imagen.type());

    //Aplicamos el procedimiento en secuencial 
    
    for (int y = 0; y < imagen.rows; y++) {
        for (int x = 0; x < imagen.cols; x++) {
            cv::Vec3b b = fondo.at<cv::Vec3b>(y, x);
            cv::Vec3b a = imagen.at<cv::Vec3b>(y, x);
            float alpha = mask.at<uchar>(y, x) / 255.0;

            cv::Vec3b c;
            c[0] = a[0] * alpha + (1 - alpha) * b[0];
            c[1] = a[1] * alpha + (1 - alpha) * b[1];
            c[2] = a[2] * alpha + (1 - alpha) * b[2];

            imgResult.at<cv::Vec3b>(y, x) = c;
        }
    }

    
    cv::imwrite("imagen_combinada.bmp", imgResult);

    return 0;
}
