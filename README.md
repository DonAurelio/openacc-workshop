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

Get into the **home** directiry

```sh
cd $HOME
```

Clone this repository into the container

```sh
git clone https://github.com/DonAurelio/openacc-workshop.git
```

Get into the **openacc-workshop** directiry

```sh
cd openacc-workshop
```

List the content of the current directory 

```sh
ls --all
```

You will get something like

```sh
drwxr-xr-x 6 root root 4096 Oct 18 01:32 ./
drwx------ 1 root root 4096 Oct 18 01:32 ../
drwxr-xr-x 8 root root 4096 Oct 18 01:32 .git/
drwxr-xr-x 5 root root 4096 Oct 18 01:32 1.matrixmult/
drwxr-xr-x 2 root root 4096 Oct 18 01:32 2.inspecting_the_gpu/
drwxr-xr-x 4 root root 4096 Oct 18 01:32 3.workshop/
-rw-r--r-- 1 root root 1008 Oct 18 01:32 README.md
```

## Do not use the following commands until the end of the workshop

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
