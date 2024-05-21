#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;
//-- Funcion device recursiva
__device__ int Factorial(int x) {
	if (x == 0)
		return 1;
	else
		return x * Factorial(x - 1);

}
//--kernel
__global__ void MyKernel(int n, int* fact) {
	*fact = Factorial(n);
}
int main() {
	int n, resultado, * d_f;
	//--Leer un entero
	//cout << "Ingrese un número: ";
	cin >> n;
	//--reservar memoria en el device
	cudaMalloc(&d_f, sizeof(int));
	//--Lanzar el kernel
	MyKernel << <1, 1 >> > (n, d_f); //--1 bloque y un hilo por bloque
	//--Copiar el resultado al host
	cudaMemcpy(&resultado, d_f, sizeof(int), cudaMemcpyDeviceToHost);
	// Print out
	cout << "El resultado es: " << resultado << endl;
	cudaFree(d_f); //--libera la memoria del host
	system("pause");
	return 0;

}