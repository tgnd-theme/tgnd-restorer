FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# PyTorch (CUDA 11.8)
RUN pip install --no-cache-dir \
    torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# App deps
RUN pip install --no-cache-dir \
    runpod==1.7.0 \
    realesrgan \
    gfpgan \
    opencv-python-headless>=4.7.0 \
    Pillow

# Force-reinstall numpy last to fix any version conflicts
RUN pip install --no-cache-dir --force-reinstall numpy==1.26.4

# Verify numpy works
RUN python -c "import numpy; print(f'numpy {numpy.__version__} OK')"

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

# Verify all imports work
RUN python -c "import numpy, cv2, runpod; from PIL import Image; print('All imports OK')"

CMD ["python", "-u", "handler.py"]
