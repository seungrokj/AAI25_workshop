# Table of content

0. [BEFORE WE BEGIN](#before-we-begin)
1. [Learning Domain Knowledge to Make Qwen-Vision Smarter with LoRA on AMD GPU](#case-1-vllm-v0-benchmarks)
2. [Adding Reason Capability into LLaMa-Vision with Torchtune](#case-2-vllm-v1-benchmarks)

# BEFORE WE BEGIN

In this workshop, we will run two fine-tuning examples to show the power of fine-tuning.


## Connecting to Digital Ocean Cloud Instance

 - 📌 IMPORTANT: Check out Digital Ocean Cloud Quick Start Guide at [digital ocean quick start](../Digital_Ocean_Usage/README.md)

### Use the following ssh cmd to connect to your instance (Window:PowerShell, Linux or Mac:Terminal)

```
ssh root@<YOUR_DIGITAL_OCEAN_INSTANCE_IP>

# Inside the host machine

CONTAINER_NAME=AAI25-finetune-workshop
IMAGE_NAME=rocm/pytorch-training:v25.4

docker run -it \
        --device /dev/dri \
        --device /dev/kfd \
        --ipc host \
        -p 8888:8888 \
        --group-add video \
        --cap-add SYS_PTRACE \
        --security-opt seccomp=unconfined \
        --privileged \
        --env HUGGINGFACE_HUB_CACHE=/root/models\
        --env MODELSCOPE_CACHE=/root/models\
        -v /root:/root \
        --workdir /root \
        --shm-size 32G \
        --name ${CONTAINER_NAME} \
        ${IMAGE_NAME} /bin/bash

```

### Pre-download the related model ckpt by 

```bash

huggingface-cli download --token <YOUR_HUGGINGFACE_KEY> --resume-download Qwen/Qwen2-VL-7B-Instruct
```


### Install Jupyter notebook server and launch it by 
```bash
pip install jupyter
cd /root && git clone https://github.com/seungrokj/AAI25_workshop.git && cd AAI25_workshop/ws_102_Finetuning_AI_Models
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

### Open your browser (e.g. Edge, Chrome, Opera..) and type in the 
```bash
http://<YOUR_DIGITAL_OCEAN_INSTANCE_IP>:8888/lab?token=<YOUR_JUPYTER_NOTEBOOK_TOKEN>
```