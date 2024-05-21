#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

// Declaración de funciones
// GLOBAL: función llamada desde el host y ejecutada en el device (kernel)
__global__ void operaciones_GPU(int a, int b, int* suma, int* resta, int* multiplicacion)
{
    *suma = a + b;
    *resta = a - b;
    *multiplicacion = a * b;
}

// HOST: función llamada y ejecutada desde el host
__host__ void operaciones_CPU(int a, int b, int* suma, int* resta, int* multiplicacion)
{
    *suma = a + b;
    *resta = a - b;
    *multiplicacion = a * b;
}

int main()
{
    // Declaraciones
    int n1 = 1, n2 = 2;
    int suma_cpu, resta_cpu, multiplicacion_cpu;
    int suma_gpu, resta_gpu, multiplicacion_gpu;
    int* dev_resultados;

    // Reserva en el device
    cudaMalloc((void**)&dev_resultados, 3 * sizeof(int));

    // Llamada a la función operaciones_CPU
    operaciones_CPU(n1, n2, &suma_cpu, &resta_cpu, &multiplicacion_cpu);

    // Llamada a la función operaciones_GPU
    operaciones_GPU << <1, 1 >> > (n1, n2, &suma_gpu, &resta_gpu, &multiplicacion_gpu);

    // Recogida de datos desde el device
    int resultados[3];
    cudaMemcpy(resultados, dev_resultados, 3 * sizeof(int), cudaMemcpyDeviceToHost);

    // Resultados CPU
    printf("CPU:\n");
    printf("%d + %d = %d\n", n1, n2, suma_cpu);
    printf("%d - %d = %d\n", n1, n2, resta_cpu);
    printf("%d * %d = %d\n", n1, n2, multiplicacion_cpu);

    // Resultados GPU
    printf("GPU:\n");
    printf("%d + %d = %d\n", n1, n2, suma_gpu);
    printf("%d - %d = %d\n", n1, n2, resta_gpu);
    printf("%d * %d = %d\n", n1, n2, multiplicacion_gpu);

    // Salida
    printf("\nPulsa INTRO para finalizar...");
    fflush(stdin);
    char tecla = getchar();

    // Liberar memoria
    cudaFree(dev_resultados);

    return 0;
}

