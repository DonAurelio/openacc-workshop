/******************************************************************************
* FILE: mm.c
* DESCRIPTION:

* AUTHOR: Aurelio Vivas
* LAST REVISED: 15/10/16
******************************************************************************/

# include <stdio.h>
# include <stdlib.h>
# include <string.h>

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
  for(int k=0;k<n;k++){
    for(int l=0;l<n;l++){
      for(int m=0;m<n;m++){
        c[n * k + l] += a[n * m + l] * b[n * k + m];
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