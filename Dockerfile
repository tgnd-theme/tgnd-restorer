FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python deps — Real-ESRGAN includes GFPGAN
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    realesrgan \
    opencv-python-headless>=4.7.0

# Download model weights
RUN mkdir -p /app/weights

# Real-ESRGAN x2plus (~64MB)
RUN wget -q -O /app/weights/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Real-ESRGAN x4plus (~64MB)
RUN wget -q -O /app/weights/RealESRGAN_x4plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

# GFPGAN v1.4 (~348MB)
RUN wget -q -O /app/weights/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# Face detection models needed by GFPGAN
RUN mkdir -p /app/gfpgan/weights
RUN wget -q -O /app/gfpgan/weights/detection_Resnet50_Final.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth
RUN wget -q -O /app/gfpgan/weights/parsing_parsenet.pth \
    https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth

COPY handler.py .

CMD ["python", "-u", "handler.py"]
