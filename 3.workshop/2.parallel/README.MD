# OpenACC Parallel Directive

## Objectives

- [x] You will learn to use the **parallel** directive
- [x] You will learn to use the **data** movement directive to copy data from tehe GPU and bring it back to the CPU memory.
- [x] You will learn to use the **loop** directive and the **independent** clause 

## Introdution

As mentioned in the [1.kernels](https://github.com/DonAurelio/openacc-workshop/tree/master/1.kernels). OpenACC give us two possibilities to express parallelism in our applications. the **kernels** and the **parallel** directives. **This section is dedicated to the parallel directive**.

When we use the **parallel** directive, the compiler relies on our capabilities to parallelize the code (this is called non automatic parallelization). In this way we have to carry out:

1. Send the data to be processed to the GPU memory.
2. When the calculation is finished in GPU, bring the data back to the CPU memory.
3. Check the GPU does what we expect.

Some problems require a more detailed treatment in the way the threads are executed, this detailed treatment is made possible by means of the parallel directive. 

The code **1.demo** is an example of the parallel directive in action.

## Running the 1.demo in a sequential approach

* Uncomment this line in your *Makefile*

```make
CFLAGS = -std=c++11 -O3
```

* Comment this line

```make
# CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30
```
* Run the code 

```bash
make run
```

## Running the 1.demo in a parallel approach

* Uncomment this line in your *Makefile*

```make
CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30
```

* Comment this line

```make
# CFLAGS = -std=c++11 -O3
```
* Run the code 

```bash
make run
```