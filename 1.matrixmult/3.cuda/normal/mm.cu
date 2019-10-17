/**
 * Autor: Grupo GRID
 * Fecha: Julio 2016
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define N 4



__global__ 
void MultiplocarMatrices(int *d_a, int *d_b, int *d_c, int n )
{
  int id, i, j;

  id = blockIdx.x * blockDim.x + threadIdx.x;
  if ( id < n )
  { 
     for( i = 0 ; i < n ; i++ )
     {
        d_c[ id * n + i ] = 0;
        for ( j = 0 ; j < n ; j++)
        {
          d_c[ id * n + i ] += ( d_a[ id * n + j ] * d_b[ j * n + id ] );
        }
      }
  }
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
    
    MultiplocarMatrices<<< 1 , n  >>> (d_a, d_b, d_c, n );

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

