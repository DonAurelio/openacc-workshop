# Inspecting the GPU 

Use the following command to get detailed information about your GPU device. This is the most important step to start with GPU programming.

```sh 
pgaccelinfo
```

This command will display something like 

```sh 
CUDA Driver Version:           10020
NVRM version:                  NVIDIA UNIX x86_64 Kernel Module  430.26  Tue Jun  4 17:40:52 CDT 2019

Device Number:                 0
Device Name:                   Quadro K2200
Device Revision Number:        5.0
Global Memory Size:            4238802944
Number of Multiprocessors:     5
Concurrent Copy and Execution: Yes
Total Constant Memory:         65536
Total Shared Memory per Block: 49152
Registers per Block:           65536
Warp Size:                     32
Maximum Threads per Block:     1024
Maximum Block Dimensions:      1024, 1024, 64
Maximum Grid Dimensions:       2147483647 x 65535 x 65535
Maximum Memory Pitch:          2147483647B
Texture Alignment:             512B
Clock Rate:                    1124 MHz
Execution Timeout:             Yes
Integrated Device:             No
Can Map Host Memory:           Yes
Compute Mode:                  default
Concurrent Kernels:            Yes
ECC Enabled:                   No
Memory Clock Rate:             2505 MHz
Memory Bus Width:              128 bits
L2 Cache Size:                 2097152 bytes
Max Threads Per SMP:           2048
Async Engines:                 1
Unified Addressing:            Yes
Managed Memory:                Yes
Concurrent Managed Memory:     No
PGI Default Target:            -ta=tesla:cc50
```

Remember the following information, you will need this later to compile your GPU code efficiently

```sh
PGI Default Target:            -ta=tesla:cc50
```

Use the following command to check the characteristcis of your CPU 

```sh 
lscpu
```

You will get something like 

```sh 
Architecture:          x86_64
CPU op-mode(s):        32-bit, 64-bit
Byte Order:            Little Endian
CPU(s):                8
On-line CPU(s) list:   0-7
Thread(s) per core:    2
Core(s) per socket:    4
Socket(s):             1
NUMA node(s):          1
Vendor ID:             GenuineIntel
CPU family:            6
Model:                 60
Model name:            Intel(R) Core(TM) i7-4790 CPU @ 3.60GHz
Stepping:              3
CPU MHz:               2958.811
CPU max MHz:           4000.0000
CPU min MHz:           800.0000
BogoMIPS:              7183.71
Virtualization:        VT-x
L1d cache:             32K
L1i cache:             32K
L2 cache:              256K
L3 cache:              8192K
NUMA node0 CPU(s):     0-7
```

# Questions 

1. Why the CPU Clock speed is better than the GPU clock speed?
2. Why the GPU have more cores that the CPU?