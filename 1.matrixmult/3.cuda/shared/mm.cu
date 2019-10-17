/**
 * Autor: Grupo GRID
 * Fecha: Julio 2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4
#define _BLOCK_SIZE_ 2

 __global__ void MultiplocarMatrices(int *A, int *B, int *C, int n)
{

  const uint wA = n;
  const uint wB = n;  
  
  const uint bx = blockIdx.x;
  const uint by = blockIdx.y;

  const uint tx = threadIdx.x;
  const uint ty = threadIdx.y;

  const uint aBegin = wA * _BLOCK_SIZE_ * by;
  const uint aEnd = aBegin + wA - 1;
  const uint aStep = _BLOCK_SIZE_;

  const uint bBegin = _BLOCK_SIZE_ * bx;
  const uint bStep = _BLOCK_SIZE_ * wB;

  float Csub = 0;

  for (int a = aBegin, b = bBegin;
       a <= aEnd;
       a += aStep, b += bStep) 
    {
      __shared__ float As[_BLOCK_SIZE_][_BLOCK_SIZE_];
      __shared__ float Bs[_BLOCK_SIZE_][_BLOCK_SIZE_];

      As[ty][tx] = A[a + wA * ty + tx];
      Bs[ty][tx] = B[b + wB * ty + tx];

      __syncthreads();

      for (int k = 0; k < _BLOCK_SIZE_; ++k)
        Csub += As[ty][k] * Bs[k][tx];

      __syncthreads();
    }

  const uint c = wB * _BLOCK_SIZE_ * by + _BLOCK_SIZE_ * bx;
  C[c + wB * ty + tx] = Csub;
}



int main( int argc, char** argv) 
{
    int *h_a, *h_b, *h_c;
    int *d_a, *d_b, *d_c;
    char validacion[10];
    

    int    n = N; // Valor por defecto

    if ( argc > 1 )
    {
       n = atoi (argv[1]);
       if ( n > 1024 )
       {
          n = 1024;
       }
    }

    size_t memSize = n * n * sizeof( int );

    h_a = (int *) malloc( memSize );
    h_b = (int *) malloc( memSize );
    h_c = (int *) malloc( memSize );

    if ( h_a == NULL || h_b == NULL || h_c == NULL )
    {
       perror("Memoria insuficiente\n");
       exit(-1);
    }

    cudaMalloc( (void**) &d_a, memSize );  
    cudaMalloc( (void**) &d_b, memSize );
    cudaMalloc( (void**) &d_c, memSize );
    
    if ( d_a == NULL || d_b == NULL || d_c == NULL )
    {
       perror("Memoria insuficiente en la GPU\n");
       exit(-1);
    }

    
    for( int i = 0 ; i < n ; i++ )
    {
      for( int j = 0 ; j < n ; j++ )
      {
         h_a[ n * i + j] = 1;
         h_b[ n * i + j] = 1;
         h_c[ n * i + j] = 0;
      }
    }

    cudaMemcpy( d_a, h_a , memSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_b, h_b , memSize, cudaMemcpyHostToDevice );  
    

    int MATRIX_SIZE = n;
    int TILE_SIZE = 2;

    dim3 dimGrid  ( MATRIX_SIZE / TILE_SIZE, MATRIX_SIZE / TILE_SIZE);
    dim3 dimBlock (TILE_SIZE, TILE_SIZE, 1);

    MultiplocarMatrices<<< dimGrid , dimBlock  >>> (d_a, d_b, d_c, n );

    cudaMemcpy( h_c, d_c, memSize, cudaMemcpyDeviceToHost );

    // Validación de la multiplicación
    // El producto de dos matrices de NxN de unos produce una matriz donde todos los elementos son N 

    strcpy(validacion, "Ok");
    for( int i = 0 ; i < n ; i++ )
    {
      for( int j = 0 ; j < n ; j++ )
      {
        if ( h_c[ n * i + j ] != n )
	 {
            strcpy(validacion, "Error");
	 }
      }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    printf ( "\t %10d \t\t %s \t ", n, validacion );

    return 0;
}

