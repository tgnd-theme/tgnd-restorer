FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# PyTorch (CPU-friendly install, CUDA handled at runtime)
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Numpy first (required by torch, realesrgan, gfpgan)
RUN pip install --no-cache-dir numpy==1.26.4

# App deps
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    realesrgan \
    gfpgan \
    opencv-python-headless>=4.7.0 \
    Pillow

# Download model weights (~480MB total)
RUN mkdir -p /app/weights /app/gfpgan/weights

RUN wget -q -O /app/weights/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

RUN wget -q -O /app/weights/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

RUN wget -q -O /app/weights/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

RUN wget -q -O /app/gfpgan/weights/detection_Resnet50_Final.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth

RUN wget -q -O /app/gfpgan/weights/parsing_parsenet.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

COPY handler.py .

CMD ["python", "-u", "handler.py"]
