# OpenACC Kernels Directive

## Objectives

- [x] You will learn how to create **kernels regions** and what is its purpose.
- [x] You will learn how to use the **loop** directive and the **independent** clause
- [x] You will learn why it is not always necessary to parallelize a code

## Introdution

OpenACC works with the same format as OpenMP does

```c
#pragma omp directive-name clauses..
```

in OpenACC

```c
#pragma acc directive-name clauses..
```

Unlike OpenMP which creates code to be executed in CPU, OpenACC creates code to both CPU and GPU. When we program in GPU platforms we must to care about:

1. Send the data to be processed to the GPU memory.
2. When the calculation is finished in GPU, bring the data back to the CPU memory.
3. Check the GPU does what we expect.

The **kernels** directive carries out the point 1 and 2 by us (this is called automatic paralelization). The **kernels** directive:

* **Tries** to identify which data in a **kernels region** is needed in the GPU memory.
* Sends the data to the GPU memory.
* **Tries** to carry out the requested calculations.
* Sends the data and result back to the CPU memory.

So we must to carry out the 3 point, since not always the compiler does what we expect. The format of a **kernels** directive to create a **kernels región** is described as follows:

```c
// Kernels region
#pragma acc kernels
{
    // Code to be parallelized here ..
}
```

## Exercise 1

Veryfy your code runs correctly in a sequential approach.

* Uncomment this line in your make file

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

Parallelize the code with the **kernels** directive.

* Enclose the following section of code in a **kernels region**.

```c
/* Vector sum  */
for(int j=0; j<N; ++j) C[j] = A[j] + B[j];

/* Vector multiplicaion */
for(int j=0; j<N; ++j) D[j] = A[j] * B[j];
```

* Uncomment this line in your make file

```make
CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30,time -Minfo=accel
```

* Comment these lines

```make
# CFLAGS = -std=c++11 -O3
```

```make
# CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30 -Minfo=accel
```

* Run the code

```bash
make run
```

### Questions

* Is the answer correct?
* Does the execution time improve?
* Why not?
* Was the code parallelized?

**Discuss these questions with the speaker**

### Feedback

If the code answer is correct we can assume the data was copied to the GPU, processed and get it back to the CPU memory as we expect.

## Exercise 2

The process of parallelization is an incremental process, ie, while you are performing small changes to the code, at the same time you have to test whether or not that change improves the execution time of the code. Experts recommend starting this process with the **kernels** directive. As in some cases this directive may be the definitive solution, in other cases it may not. **The kernels directive parallelize our code, but not in the way we expect, because the execution time does not improve and we got a compiler feedback that suggest what is the real problem.**.

Let's give the compiler more information about our code. 

* Place the **loop** directive with the **independent** clause just above the vector sum loop.

```c
/* Vector sum  */
#pragma acc loop independent
for(...) .....;
```

* Run the code **vector.c**

```bash
make run
```

### Questions

* Is the answer correct?
* Does the execution time improve?
* Why?
* What does the **loop** directive do?
* What does the **independent** clause do?

**Discuss these questions with the speaker**

## Exercise 3

* Place the **loop** directive with the **independent** clause just above the vector multiplication for loop.

```c
/* Vector multiplication  */
#pragma acc loop independent
for(...) .....;
```

* Run the code **vector.c**

```bash
make run
```

### Questions

* Is the answer correct?
* **Why does not the runtime improve with parallelization?**

**Discuss these questions with the speaker**