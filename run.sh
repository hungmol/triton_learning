docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    --gpus all \
    -v $(pwd)/models:/models \
    nvcr.io/nvidia/tritonserver:23.09-py3 \
    tritonserver --model-repository=/models