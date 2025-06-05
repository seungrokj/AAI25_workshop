 - TODOs: 
 * 1. Change host server IP
 * 2. Change the vLLM docker image if needed
 * 3. Change model downloaded location

# Table of content

0. [BEFORE WE BEGIN](#before-we-begin)
    1. [Case 1 vLLM v0 benchmarks](#case-1-vllm-v0-benchmarks)
    2. [Case 2 vLLM v1 benchmarks](#case-2-vllm-v1-benchmarks)
    3. [Case 3 vLLM v1 with Prefix-caching benchmarks](#case-3-vllm-v1-with-prefix-caching-benchmarks)

# BEFORE WE BEGIN

In this workshop, we will run three vLLM servers and compare the caracteristics of each server. 

![WORKSHOP_DESC](./assets/LLM_ws_201.jpg)

Each person will receive 1) a host machine ssh IP, 2) GPU Device ID in the host machine (HIP_VISIBLE_DEVICES), 3) vLLM server PORT, 4) Jupyter notebook PORT

-----------------------------
📌 For example, 

| Person ID    | Host machine ssh IP | GPU Device ID         | vLLM server PORT | Jupyter NB PORT |
| ------------ | --------------------|-----------------------|------------------|-----------------|
| 0            | amd@64.139.222.215  | HIP_VISIBLE_DEVICES=0 | 8100             | 7100            |
| 1            | amd@64.139.222.215  | HIP_VISIBLE_DEVICES=1 | 8101             | 7101            |
| 2            | amd@64.139.222.215  | HIP_VISIBLE_DEVICES=2 | 8102             | 7102            |
| 8            | amd@64.139.222.216  | HIP_VISIBLE_DEVICES=0 | 8108             | 7108            |
| 9            | amd@64.139.222.216  | HIP_VISIBLE_DEVICES=1 | 8109             | 7109            |

## Connecting to AMD host machines through SSH using any terminal applications

 - IMPORTANT: Replace a specific vLLM PORT `<81xx>`, JUPYTER NOTEBOOK PORT `<71xx>`, GPU ID <x>, and HOST MACHINE IP <host machine IP> according to your Personal ID. 


### 📌 Please change according to your assigned ports, gpus, and hosts machine
```
ssh -L <71xx>:localhost:<71xx> -L <81xx>:localhost:<81xx> <host machine IP>

# Inside the host machine
export PORT_VLLM=<81xx>
export PORT_JUPYTER=<71xx>
export GPU_ID=<x>
export model=/models/Llama-3.1-8B-Instruct
```

### For example, if you are using vLLM PORT `<8100>`, JUPYTER NOTEBOOK PORT `<7100>`, GPU ID <0>, and HOST MACHINE IP <amd@64.139.222.215> according to your Personal ID. 

```
ssh -L 7100:localhost:7100 -L 8100:localhost:8100 amd@64.139.222.215

# Inside the host machine
export PORT_VLLM=8100
export PORT_JUPYTER=7100
export GPU_ID=0
export model=/models/Llama-3.1-8B-Instruct
```

## ✨ 
## Case 1 vLLM v0 benchmarks
-----------------------------

Check the GPU availability by `rocm-smi`

```
================================================== ROCm System Management Interface ==================================================
============================================================ Concise Info ============================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf              PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
======================================================================================================================================
0       2     0x74a1,   28851  36.0°C      144.0W    NPS1, SPX, 0        122Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
1       3     0x74a1,   51499  35.0°C      141.0W    NPS1, SPX, 0        121Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
2       4     0x74a1,   57603  37.0°C      146.0W    NPS1, SPX, 0        121Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
3       5     0x74a1,   22683  33.0°C      140.0W    NPS1, SPX, 0        122Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
4       6     0x74a1,   53458  35.0°C      141.0W    NPS1, SPX, 0        121Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
5       7     0x74a1,   26954  34.0°C      141.0W    NPS1, SPX, 0        121Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
6       8     0x74a1,   16738  35.0°C      142.0W    NPS1, SPX, 0        125Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
7       9     0x74a1,   63738  32.0°C      143.0W    NPS1, SPX, 0        147Mhz  900Mhz  0%   perf_determinism  750.0W  0%     0%
======================================================================================================================================
======================================================== End of ROCm SMI Log =========================================================
```

(We downloaded the model, but in case the model is not found) Download target model [RedHatAI/Llama-3.1-8B-Instruct](https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct) at /home/amd/models

```
cd /home/amd/models
git-lfs clone https://huggingface.co/RedHatAI/Llama-3.1-8B-Instruct
```

### SERVER) vLLM v0 default option

```
docker run -it --rm --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e VLLM_USE_V1=0 \
    -e HIP_VISIBLE_DEVICES=$GPU_ID \
    -e VLLM_USE_TRITON_FLASH_ATTN=0 \
    -v /home/amd/models:/models  -v /home/amd/datasets:/datasets \
    rocm/vllm-dev:nightly_610_rc1_6.4.1_6_10_rc1_20250529 \
    vllm serve $model \
            --disable-log-requests \
            --trust-remote-code -tp 1 \
            --cuda-graph-sizes 64 \
            --chat-template /app/vllm/examples/tool_chat_template_llama3.1_json.jinja \
            --port $PORT_VLLM
```

Once servers are ready, you can see these logs in the terminal

```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### CLIENT) Launch Jupyternotebook servers on the AMD host machine

Server. Jupyter notebook, running at a port: <71xx>, for example, 7100

In an other terminal, now ssh again into the AMD host machine 

#### Once again, please change according to your assigned ports, gpus, and hosts machine
```
ssh -L <71xx>:localhost:<71xx> -L <81xx>:localhost:<81xx> <host machine IP>

# Inside the host machine
export PORT_VLLM=<81xx>
export PORT_JUPYTER=<71xx>
export GPU_ID=<x>
export model=/models/Llama-3.1-8B-Instruct
```

#### For example, if you are using vLLM PORT `<8100>`, JUPYTER NOTEBOOK PORT `<7100>`, GPU ID <0>, and HOST MACHINE IP <amd@64.139.222.215> according to your Personal ID. 

```
ssh -L 7100:localhost:7100 -L 8100:localhost:8100 amd@64.139.222.215

# Inside the host machine
export PORT_VLLM=8100
export GPU_ID=0
export model=/models/Llama-3.1-8B-Instruct
export PORT_JUPYTER=7100
```

#### Launch Jupyter notebook container and access it via a web browser

Launch this Jupyter notebook container

```
docker run -it --rm -u root --entrypoint /bin/bash --net host \
    -v $(pwd):/workspace -v /home/amd/models:/models  -v /home/amd/datasets:/datasets \
    -e PORT_JUPYTER=$PORT_JUPYTER \
    jupyter/base-notebook

```

Inside the container, please clone this workshop repo
```
apt update
apt install git -y
cd /workspace
git clone https://github.com/ROCm/aai25_workshop.git
cd aai25_workshop/ws_201_Optimized_Model_Serving_with_vLLM
```

Launch the Jupyter notebook

```
jupyter-notebook --allow-root --port $PORT_JUPYTER
```

You can access the Jupyter notebook server that starts with http:/127.0.0.1:<71xx> below

```
[I 2025-06-02 15:00:13.584 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 2025-06-02 15:00:13.586 ServerApp]

    To access the server, open this file in a browser:
        file:///home/jovyan/.local/share/jupyter/runtime/jpserver-23-open.html
    Or copy and paste one of these URLs:
        http://tw015:<71xx>/tree?token=41748c2ad340ea87a568dd545986075082cf7d785f16787e
        http://127.0.0.1:<71xx>/tree?token=41748c2ad340ea87a568dd545986075082cf7d785f16787e

```

Now follow steps in the `AAI25_workshop_ws_201.ipynb`

## ✨ 
## Case 2 vLLM v1 benchmarks
-----------------------------

Now close the previous vLLM server by `ctrl+C` and launch a new vLLM server with v1 enabled
Chekc out `VLLM_USE_V1=1` env var to grigger v1. 
And follow instructions at `AAI25_workshop_ws_201.ipynb`

### SERVER) vLLM v1 without prefix-caching

```
docker run -it --rm --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e VLLM_USE_V1=1 \
    -e HIP_VISIBLE_DEVICES=$GPU_ID \
    -e VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
    -v /home/amd/models:/models  -v /home/amd/datasets:/datasets \
    rocm/vllm-dev:nightly_610_rc1_6.4.1_6_10_rc1_20250529 \
    vllm serve $model \
            --disable-log-requests \
            --trust-remote-code -tp 1 \
            --cuda-graph-sizes 64 \
            --no-enable-prefix-caching \
            --chat-template /app/vllm/examples/tool_chat_template_llama3.1_json.jinja \
            --port $PORT_VLLM
```

### CLIENT) Keep following instructions at `AAI25_workshop_ws_201.ipynb`

## ✨ 
## Case 3 vLLM v1 with Prefix-caching benchmarks
-----------------------------

Now close the previous vLLM server by `ctrl+C` and launch a new vLLM server with v1 and prefix cache are enabled
Chekc out `VLLM_USE_V1=1` env var to grigger v1 and `--enable-prefix-caching` vllm arg to enabled prefix caching. 

### SERVER) vLLM v1 with prefix-caching

```
docker run -it --rm --network=host \
    --device=/dev/kfd --device=/dev/dri \
    --ipc=host --shm-size 16G \
    --group-add video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -e VLLM_USE_V1=1 \
    -e HIP_VISIBLE_DEVICES=$GPU_ID \
    -e VLLM_V1_USE_PREFILL_DECODE_ATTENTION=1 \
    -v /home/amd/models:/models  -v /home/amd/datasets:/datasets \
    rocm/vllm-dev:nightly_610_rc1_6.4.1_6_10_rc1_20250529 \
    vllm serve $model \
            --disable-log-requests \
            --trust-remote-code -tp 1 \
            --cuda-graph-sizes 64 \
            --enable-prefix-caching \
            --chat-template /app/vllm/examples/tool_chat_template_llama3.1_json.jinja \
            --port $PORT_VLLM
```

### CLIENT) Keep following instructions at `AAI25_workshop_ws_201.ipynb`
