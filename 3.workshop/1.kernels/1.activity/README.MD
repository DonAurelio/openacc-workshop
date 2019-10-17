# OpenACC Kernels Directive

## Objectives

- [x] You will evidence a case in which the **kernels** directive alone does not work properly
- [x] You will learn how to give the compiler more information about your code.
- [x] You will learn in which cases a code can benefit from the parallelization.

## Introdution

There are cases in which the compiler itself is not able to figure out what we want to do in our code. In most cases it is necessary to give it more information about the code we want to parallelize.

## Exercise 1

Veryfy your code runs correctly in a sequential approach.

* Uncomment this line of your make file

```make
CFLAGS = -std=c++11 -O3
```

* Comment these lines

```make
# CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30 -Minfo=accel
```

```make
# CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30,time -Minfo=accel
```
* Run the *matrixmult.c* code

```bash
make run
```

### Questions

* Take notes of the execution time.
* Take notes of the validation.

## Exercise 2

The purpose of this exercise is to parallelize the matrix multiplication problem. To do this, it is recommended to follow this steps:

1. Identify in the code that región that is intensive computationally.
2. Try to parallelize that section(s) of code with a **kernels región**. 
3. If the code works correctly with **kernels** try to improve its execution time, giving the compiler more information about your code.

Edit the file *matrixmult.c*

* Enclose the following code in a **kernels región** 

```c
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

```
**NOTE:** The part of the code that performs the matrix multiplication is considered to be more expensive computationally.

* Uncomment this line of your make file

```make
CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30 -Minfo=accel
```

* Comment these lines

```make
# CFLAGS = -std=c++11 -O3
```

```make
# CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30,time -Minfo=accel
```
* Run the *matrixmult.c* code

```bash
make run
```

### Questions

* What happened with the execution of the program?

**Discuss this question with the speaker**

### Feedback

In some cases the compiler does not have the necessary information of our code to be able to parallelize it correctly. As is the case in this example. Given this it is neccesary to provide it with more information.

## Exercise 3

Give the compiler more information about the code.

* Place the **loop** directive with the **independent** clause just above the **rows loop**.

```c
// Rows loop
#pragma acc loop independent
for(...){
    for(...){
        ....
    }
};
```

* Run the code

```bash
make run
```

### Questions

* Did the code run correctly?
* Why?
* Take notes about the execution time.
* **Ask the speaker how does this code work?**

## Exercise 4

Give even more information to the compiler about the code.

* Place the **loop** directive with the **independent** clause just above the **cols loop**.

```c
// Cols loop
#pragma acc loop independent
for(...){
    ....
}
```

* Run the code

```bash
make run
```

### Questions

* Did the code run conrrectly?
* Why?
* Take notes about the execution time.
* In what exercise of this section did the code run faster?
* **Why does the runtime improve with parallelization?**

* **Ask the speaker how does this code work**