#include <stdio.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define N 16 // tamaño de los vectores
#define BLOCK 5 // tamaño del bloque

// Declaración de funciones
__global__ void suma(float* a, float* b, float* c)
{
    int myID = threadIdx.x + blockDim.x * blockIdx.x;
    // Sólo trabajan N hilos
    if (myID < N)
    {
        c[myID] = a[myID] + b[myID];
    }
}

__global__ void resta(float* a, float* b, float* c)
{
    int myID = threadIdx.x + blockDim.x * blockIdx.x;
    if (myID < N)
    {
        c[myID] = a[myID] - b[myID];
    }
}

__global__ void multiplicacion(float* a, float* b, float* c)
{
    int myID = threadIdx.x + blockDim.x * blockIdx.x;
    if (myID < N)
    {
        c[myID] = a[myID] * b[myID];
    }
}

// MAIN: rutina principal ejecutada en el host
int main(int argc, char** argv)
{
    // Declaraciones
    float* vector1, * vector2, * resultado_suma, * resultado_resta, * resultado_multiplicacion;
    float* dev_vector1, * dev_vector2, * dev_resultado_suma, * dev_resultado_resta, * dev_resultado_multiplicacion;

    // Reserva memoria en el host
    vector1 = (float*)malloc(N * sizeof(float));
    vector2 = (float*)malloc(N * sizeof(float));
    resultado_suma = (float*)malloc(N * sizeof(float));
    resultado_resta = (float*)malloc(N * sizeof(float));
    resultado_multiplicacion = (float*)malloc(N * sizeof(float));

    // Reserva memoria en el device
    cudaMalloc((void**)&dev_vector1, N * sizeof(float));
    cudaMalloc((void**)&dev_vector2, N * sizeof(float));
    cudaMalloc((void**)&dev_resultado_suma, N * sizeof(float));
    cudaMalloc((void**)&dev_resultado_resta, N * sizeof(float));
    cudaMalloc((void**)&dev_resultado_multiplicacion, N * sizeof(float));

    // Inicialización de vectores con valores aleatorios
    for (int i = 0; i < N; i++)
    {
        vector1[i] = (float)rand() / RAND_MAX;
        vector2[i] = (float)rand() / RAND_MAX;
    }

    // Copia de datos hacia el device
    cudaMemcpy(dev_vector1, vector1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vector2, vector2, N * sizeof(float), cudaMemcpyHostToDevice);

    // Lanzamiento del kernel
    // Calculamos el número de bloques necesario para un tamaño de bloque fijo
    int nBloques = (N + BLOCK - 1) / BLOCK;
    int hilosB = BLOCK;

    printf("Vector de %d elementos\n", N);
    printf("Lanzamiento con %d bloques (%d hilos)\n", nBloques, nBloques * hilosB);

    // Suma
    suma <<<nBloques, hilosB>>> (dev_vector1, dev_vector2, dev_resultado_suma);
    cudaMemcpy(resultado_suma, dev_resultado_suma, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Resta
    resta <<<nBloques, hilosB>>> (dev_vector1, dev_vector2, dev_resultado_resta);
    cudaMemcpy(resultado_resta, dev_resultado_resta, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Multiplicación
    multiplicacion <<<nBloques, hilosB>>> (dev_vector1, dev_vector2, dev_resultado_multiplicacion);
    cudaMemcpy(resultado_multiplicacion, dev_resultado_multiplicacion, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Impresión de resultados en el host
    printf("> vector1:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", vector1[i]);
    }
    printf("\n");

    printf("> vector2:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", vector2[i]);
    }
    printf("\n");

    printf("> SUMA:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", resultado_suma[i]);
    }
    printf("\n");

    printf("> RESTA:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", resultado_resta[i]);
    }
    printf("\n");

    printf("> MULTIPLICACIÓN:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", resultado_multiplicacion[i]);
    }
    printf("\n");

    // Liberamos memoria en el device
    cudaFree(dev_vector1);
    cudaFree(dev_vector2);
    cudaFree(dev_resultado_suma);
    cudaFree(dev_resultado_resta);
    cudaFree(dev_resultado_multiplicacion);

    // Salida
    free(vector1);
    free(vector2);
    free(resultado_suma);
    free(resultado_resta);
    free(resultado_multiplicacion);

    printf("\npulsa INTRO para finalizar...");
    fflush(stdin);
    char tecla = getchar();

    cudaDeviceSynchronize();
    return 0;
}
