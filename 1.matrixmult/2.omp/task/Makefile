##
# Autor: Grupo GRID
# Fecha: Julio 2016
##
 
# Nombre del programa
TARGET      = mm

# Argumentos del programa
ARG         = 2 4 8 16 32 64 128 256 512 1024

EXT         = .c

COMPILER    =  gcc

CFLAGS      = -fopenmp -std=c99

all: build run

build: 
	@$(COMPILER) $(CFLAGS) -o $(TARGET) $(TARGET)$(EXT)


run:	
	@clear 
	@date | tee Time.txt
	@echo "\n\t N \t\t\t Estado \t T.Real(sg)\tT.Usuario(sg)\tT.Kernel(sg)\tCPU\tMemoria(KB)" | tee -a Time.txt
	@for argument in $(ARG) ; do /usr/bin/time -f "\t %e \t\t %U \t\t %S \t\t %P \t\t %M" ./$(TARGET) $$argument 2>&1 | tee -a Time.txt ; done


clean:
	@rm -f $(TARGET) $(TARGET).o *~ Time.*
