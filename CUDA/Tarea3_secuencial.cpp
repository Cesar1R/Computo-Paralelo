#include <bits/stdc++.h>
#include <chrono> 
#include <random>

using namespace std;

#define N 512 
#define M 512


void fill_mat (vector<vector<float>> & mat){
    size_t i, j;

    static random_device rd;
    static mt19937 gen(rd());

    static uniform_real_distribution<> distr(-100.0, 100.0); 

    for(i = 0; i < mat.size(); i++){
        for(j = 0; j < mat[0].size(); j++){
            mat[i][j] = distr(gen);
        }
    } 
}

float sq (float k){
    return k * k; 
}

int main(){

    vector<vector<float>> I1 (N, vector<float> (M)), 
                          I2 (N, vector<float> (M));

    size_t i, j; 

    float sum = 0;

    fill_mat(I1);
    fill_mat(I2);

    auto start = std::chrono::high_resolution_clock::now();


    for(i = 0; i < N; i++){
        for (j = 0; j < M; j++){
            sum += sq(I1[i][j] - I2[i][j]);  
        }
    }


    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> tiempo = end - start; 
    
    std::cout << "Tiempo del codigo en secuencial: " << tiempo.count() << endl; 


    cout << "Diferencia de cuadrados: " << sum << endl; 

    return 0;
}