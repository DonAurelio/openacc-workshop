/******************************************************************************
* FILE: mm.c
* DESCRIPTION:
*   Parallelization of square matriz muliplication using data paralelism approach.
*   matrices a,b and c are represened as arrays.  
* AUTHOR: Aurelio Vivas
* LAST REVISED: 15/10/16
******************************************************************************/

# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <omp.h>

int main(int argc, char** argv){
  
  /* Matrices */
  int *a, *b, *c;

  /* Para comprobar el resultado */
  char validacion[10];

  /* Dimensiones de las matrices cuadradas a,b,c */
  int n = 0;

  if(argc > 1){
    n = atoi(argv[1]);
    if(n>1024){
      n = 1024;
    }
  }

  /* Reservando memoria */
  size_t memSize = n * n * sizeof(int);
  a = (int *) malloc(memSize);
  b = (int *) malloc(memSize);
  c = (int *) malloc(memSize);

  if(a == NULL || b == NULL || c == NULL){
    perror("Memoria insuficiente");
    exit(-1);
  }

  /* Llenando las matrices */
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      a[n * i + j] = 1;
      b[n * i + j] = 1;
      c[n * i + j] = 0;
    }
  }

  /* Multiplicando las matrices a y b */
  int num_of_cores = omp_get_num_procs();
  int blocks = n*n / num_of_cores;
  //printf("Matrix lenght => %d\n",n*n);
  //printf("Block size => %d\n",blocks);
  //printf("Thread id \t #Block \t Thread i\t ai \t bi \t\n");
  for(int block=0;block<blocks;block++){
    #pragma omp parallel shared(a,b,c)
    {
      int thread_id = omp_get_thread_num();
      // Primero identificamos los indices en la matriz resultado
      int thread_index = block * num_of_cores + thread_id;
      for(int m=0;m<n;m++){
        // Luego identificamos los indices en las matrices a multiplicar 
        int a_index = ((thread_index / n) * n) + m;
        int b_index = (n * m) + (thread_index % n);
        c[thread_index] += a[a_index] * b[b_index];
        //printf("%d \t\t %d \t\t %d \t\t %d \t %d \t\n",thread_id,block,thread_index,a_index,b_index);
      }
    }
  }


  /* Validacion de la multiplicacion 
    La muliplicacion de dos matrices nxn de unos, da como
    resultado una matriz donde todos su elementos son n.
  */
  strcpy(validacion,"Ok");
  for(int i=0;i<n;i++){
    for(int j=0;j<n;j++){
      if(c[n * i + j] != n){
        strcpy(validacion,"Error");
      }
    }
  }

  /* Liberando memoria */
  free(a);
  free(b);
  free(c);

  printf("\t %10d \t\t %s \t ", n, validacion);
}