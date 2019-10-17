#include <stdio.h>      /* Standard I/O Library: printf */
#include <stdlib.h>     /* Standard Library: malloc, calloc, free, ralloc */
#include <string.h>     /* String funtions: strcpy*/

# define N 1000         /* A, B and C rows and cols */

/* Creating an alias for unsigned long int datatype */
typedef unsigned long int long_int;

int main(int argc, char *argv[]){

    /* Requesting dynamic memory for space to allocate A,B,C 
    matrcies are created commonly like A[N][N], B[N][N], C[N][N]. However
    when N is too large it is neccesary to create them in the way
    described as follows.
    */
    long_int ** A = (long_int **) malloc( N * sizeof(long_int *) );
    long_int ** B = (long_int **) malloc( N * sizeof(long_int *) );
    long_int ** C = (long_int **) malloc( N * sizeof(long_int *) );

    for(int i=0; i<N; ++i){
        A[i] = (long_int *) malloc( N * sizeof(long_int *) );
        B[i] = (long_int *) malloc( N * sizeof(long_int *) );
        C[i] = (long_int *) malloc( N * sizeof(long_int *) );
    }

    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            A[i][j] = B[i][j] = 1;
            C[i][j] = 0;           
        }
    }

    
    /* Matrix multiplication C = A x B */
    // Rows loop
    for(int i=0;i<N;i++){
        // Cols loop
        for(int j=0;j<N;j++){
            for(int k=0;k<N;k++){
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }


    /* Checking the answere */
    char validation[10];
    strcpy(validation,"Ok");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            if( C[i][j] != N) strcpy(validation, "Fail");
        }
    }

    printf ( "\nSize %d, Validation %s\n", N, validation );

    /* When we request dynamic memory for large arrays
    it is neccesary to release it before the application finish.
    If you do not do this, the memory will be occupied with the data
    of the arrays A, B and C regardless of whether the program is 
    finished. */
    for(int i=0; i<N; ++i){
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }

    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
};
