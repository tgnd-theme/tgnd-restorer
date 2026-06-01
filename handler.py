"""
RunPod Serverless handler — GFPGAN face restoration + Real-ESRGAN upscale.

Input:  {"source_image": "<base64>", "scale": 2, "face_enhance": true}
Output: {"status": "ok", "image": "<base64>"}
"""

import base64
import io
import os

import numpy as np
import runpod
from PIL import Image

WEIGHTS_DIR = "/app/weights"

# Lazy-loaded
_upsampler_2x = None
_upsampler_4x = None


def get_upsampler(scale=2):
    """Load Real-ESRGAN upsampler (cached)."""
    global _upsampler_2x, _upsampler_4x

    if scale == 2 and _upsampler_2x is not None:
        return _upsampler_2x
    if scale == 4 and _upsampler_4x is not None:
        return _upsampler_4x

    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet

    if scale == 2:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x2plus.pth")
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True,
    )

    if scale == 2:
        _upsampler_2x = upsampler
    else:
        _upsampler_4x = upsampler

    return upsampler


def handler(job):
    """RunPod handler."""
    import cv2

    try:
        job_input = job["input"]

        # Decode input
        img_data = base64.b64decode(job_input["source_image"])
        img = Image.open(io.BytesIO(img_data)).convert("RGB")
        img_np = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        scale = int(job_input.get("scale", 2))
        if scale not in (2, 4):
            scale = 2
        face_enhance = bool(job_input.get("face_enhance", False))

        upsampler = get_upsampler(scale)

        if face_enhance:
            from gfpgan import GFPGANer

            face_enhancer = GFPGANer(
                model_path=os.path.join(WEIGHTS_DIR, "GFPGANv1.4.pth"),
                upscale=scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )

            _, _, result = face_enhancer.enhance(
                img_np, has_aligned=False, only_center_face=False, paste_back=True
            )
        else:
            result, _ = upsampler.enhance(img_np, outscale=scale)

        # Encode output as PNG
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        result_pil = Image.fromarray(result_rgb)
        buffer = io.BytesIO()
        result_pil.save(buffer, format="PNG", optimize=True)
        output_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "status": "ok",
            "image": output_b64,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})
