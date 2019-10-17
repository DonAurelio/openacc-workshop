#include <stdio.h>      /* Standard I/O Library: printf */
#include <stdlib.h>     /* Standard Library: malloc, calloc, free, ralloc */
#include <string.h>     /* String funtions: strcpy*/

#define N 10000000

/* Creating an alias for unsigned long int datatype */
typedef unsigned long int long_int;

int main(){
    /* Requesting dynamic memory for space to allocate A,B,C 
    Arrays are created commonly like A[N], B[N], C[N]. However
    when N is too large it is neccesary to create them in the way
    described as follows.
    */
    long_int * A = (long_int *) malloc( N * sizeof(long_int) );
    long_int * B = (long_int *) malloc( N * sizeof(long_int) );
    long_int * C = (long_int *) malloc( N * sizeof(long_int) );
    long_int * D = (long_int *) malloc( N * sizeof(long_int) );


    /* Initializing A, B, C */
    for(int i=0; i<N; ++i){ 
        A[i] = 2;
        B[i] = 2;
        C[i] = 0;
        D[i] = 0;

    }

    /* Vector sum  */
    for(int j=0; j<N; ++j) C[j] = A[j] + B[j];

    /* Vector multiplicaion */
    for(int j=0; j<N; ++j) D[j] = A[j] * B[j];
    
    /* Checking the answer */
    char validation[10];
    strcpy(validation,"Ok");
    for(int k=0; k<N; ++k) 
        if (C[k] != 4) strcpy(validation,"Fail");

    for(int k=0; k<N; ++k) 
        if (D[k] != 4) strcpy(validation,"Fail");

    printf ( "\nSize %d, Validation %s\n", N, validation );

    /* When we request dynamic memory for large arrays
    it is neccesary to release it before the application finish.
    If you do not do this, the memory will be occupied with the data
    of the arrays A, B and C regardless of whether the program is 
    finished. */
    free(A);
    free(B);
    free(C);

    return EXIT_SUCCESS;
}
