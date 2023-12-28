FROM nvcr.io/nvidia/tensorrt:23.06-py3

ENV TZ=America/Santiago
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && \
    apt-get install -y curl ca-certificates && \
    apt-get install -y cmake && \
    rm -rf /var/lib/apt/lists/*

# Fix common locale problem
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN apt-get update  && apt-get install -y locales && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8

WORKDIR /opt/app
COPY requirements.txt /opt/app
COPY deps/tide /opt/app/tide
COPY deps/ultralytics /opt/app/ultralytics
WORKDIR /opt/app/tide

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .

RUN apt-get update  && apt-get install -y git && \
    apt-get install -y wget && \
    apt-get install -y vim  && \
    apt-get install -y ffmpeg && \
    apt-get install -y libsm6 && \
    apt-get install -y libxext6 && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && apt-get install git-lfs

WORKDIR /opt/app/    
RUN python3 -m pip install --upgrade pip && \
python3 -m pip install -r requirements.txt

WORKDIR /opt/app/ultralytics
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .

RUN python3 -m pip uninstall -y torch torchvision numpy && \
    python3 -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118 && \
    python3 -m pip install numpy==1.23.1 && \
    python3 -m pip install onnx_graphsurgeon --index-url https://pypi.ngc.nvidia.com
    
WORKDIR /workspace

ENV PYTHONPATH /workspace/detectron2
