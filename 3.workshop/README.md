# Introduction to Parallel Porgramming with OpenACC - Workshop

In recent years GPUs have become a disruptive technology to carry out applications related to Deep Learning, Computer Vision, Bioinformatics, problems related to the treatment of matrices and vectors, among others. OpenACC born as an alternative to the popular CUDA Programming Model for programming GPUs. Given the latent dificulty that the latter imposes to the scientific world. To break this barrier OpenACC offers an approach similar to that offered by OpenMP. OpenACC is available in the GNU compiler gcc/g++, however, this workshop will focus on the PGI compiler pgcc/pg++ which offers a better support and the NVIDIA device.

## Methodology

This workshop is guided by a presentation. During the presentation is required to make changes to the code present in this repository. In addition, questions are suggested to discuss during the talk.

## Conclusions

* Some applications only need compiler optimizations to improve their runtime. As is the case in section 1.kernels -(vectors addition and multiplication). Since the amount of work that must be done is few and is not worth incurring in the cost that adds parallelism to the application (data transmission CPU-GPU memory and GPU-CPU memory, threads creation and schedulig).

* Other applications work well with compile optimizations and OpenACC parallelism, as is the case in section 1.kernels - 1.activity (matrix multiplication). Since the amount of work that must be done is high enough and is worth incurring the cost that adds the parallelism to the application (data transmission CPU-GPU memory and GPU-CPU memory, threads creation and schedulig).

## References

[Introduction to OpenACC](http://heather.cs.ucdavis.edu/OpenACCDir/Intros158/FuLinMiller.pdf)

[The OpenACC Aplication Programming Interface](http://www.openacc.org/sites/default/files/inline-files/OpenACC_1_0_specification.pdf)
