/*
	This is just a little example program that multiplies two matrices A and B,
	using cuBLAS for CUDA 7.5

	Compile and link with 
		nvcc mm.c -lcublas
*/
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// Multiply matrices on the CPU if you desire (faster for small matrices like we are using)
void multiplyMatrices(double* a, int rows_a, int cols_a, double* b, int rows_b, int cols_b, double* c, double alpha){
	for (int i=0; i<rows_a; i++){
		for (int j=0; j<cols_b; j++){
			for (int k=0; k<cols_a; k++){
				c[i*cols_b + j] += a[i*cols_a + k] * b[k*cols_b + j];
			}
		}
	}
}

// Prints a matrix m
void printMat(double* m, int rows, int cols){
	for (int i=0; i<rows; i++){
		for (int j=0; j<cols; j++){
			printf("%+5.4f  ", m[i*cols + j]);
		}
		printf("\n");
	}
}

void cuMultiplyMatrices(
	double* h_a, int rows_a, int cols_a, // matrix A on host
	double* h_b, int rows_b, int cols_b, // matrix B on host
	double* h_c, // matrix C on host with assumed dimensions
	double alpha // scalar multiplication
){
	cublasHandle_t handle;
	double *d_a, *d_b, *d_c;
	const double beta = 0.0;

	cublasCreate(&handle);

	// Allocate space for matrices on device
	cudaMalloc(&d_a, rows_a*cols_a*sizeof(double));
	cudaMalloc(&d_b, rows_b*cols_b*sizeof(double));
	cudaMalloc(&d_c, rows_a*cols_b*sizeof(double));

	// Copy A and B to device
	cudaMemcpy(d_a, h_a, rows_a*cols_a*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, rows_b*cols_b*sizeof(double), cudaMemcpyHostToDevice);

	// Call the cublas routine with the matrices in reverse order because cublas is column primary
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, cols_b, rows_a, cols_a, &alpha, d_b, cols_b, d_a, cols_a, &beta, d_c, cols_b);

	// Copy solution back to host
	cudaMemcpy(h_c, d_c, rows_a*cols_b*sizeof(double), cudaMemcpyDeviceToHost);

	cublasDestroy(handle);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

int main(void){
	double a[] = {
		1,2,
		4,5,
		6,7
	};
	double b[] = {
		1, 1, 0, 2,
		1, 0, 1, -1
	};
	double c[] = {
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	multiplyMatrices(a, 3, 2, b, 2, 4, c, 1.0);
	printMat(c,3,4);
}