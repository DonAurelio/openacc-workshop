SHELL := /bin/bash

# FILE
TARGET = matrixmult
EXT = .c
COMPILER = pgc++

# -std=c++11, tells the compiler to use the C++ 11 standard libary
# -O3, tells the compiler to perfome aggressive optimizations (this can chage code results)
# -acc, enable OpenACC pragmas in the source code
# -ta=<gpu arch>, works with OpenACC, tells the compiler the GPU architecure
# -Minfo=accel, tells the compiler to show GPU code generation steps taken

# Uncomment for sequential execution
# CFLAGS = -std=c++11 -O3

# Uncomment to enable OpenACC pragmas in the source code
# CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30 -Minfo=accel

# Uncomment to enable OpenACC pragmas and take GPU execution time
# I recommend use this when the program compile correctly
CFLAGS = -std=c++11 -O3 -acc -ta=tesla:cc30,time -Minfo=accel

run: compile
	@time ./$(TARGET)

compile:
	@$(COMPILER) $(TARGET)$(EXT) -o $(TARGET) $(CFLAGS)

clean:
	@rm -rf $(TARGET)