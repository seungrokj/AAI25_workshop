CONTAINER_NAME=billaai0605
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
