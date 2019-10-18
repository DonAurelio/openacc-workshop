# OpenACC Workshop 

Requirements:

1. Nvidia GPU 
2. Nvidia Driver Installed
3. Nvidia CUDA Installed 
4. [Docker Nvidia Installed](https://github.com/NVIDIA/nvidia-docker)


## Check the NVIDIA GPU CLOUD

Check the NVIDIA GPU CLOUD for a GPU container for this workshop: 

Use the following link https://ngc.nvidia.com/

## Setting Up

Setting up a docker container to play with the GPU 

```ssh
docker run --runtime nvidia --name $USER -it nvcr.io/hpc/pgi-compilers:ce bash
```

Update the source list

```ssh
apt update 
```

Install the following packages

```sh
apt install git nano wget time
```

To exit from within the container use

```sh
exit
```

To stop your conatiner use 

```sh
docker stop $USER
```

To remove your container use

```sh
docker rm $USER
```