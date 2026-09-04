import os
import sys

import numpy as np
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# GeoSR - Visual Enhancement Inference
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    ROOT,
    "models",
    "edsr_visual_best.pth"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SCALE = 4

FEATURES = 64
NUM_BLOCKS = 8

RES_SCALE = 0.1


# ============================================================
# MODEL
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(
        self,
        channels=64,
        res_scale=0.1
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.res_scale = res_scale

    def forward(self, x):

        y = F.relu(
            self.conv1(x),
            inplace=True
        )

        y = self.conv2(y)

        return x + y * self.res_scale


class DetailEDSR(nn.Module):

    def __init__(
        self,
        features=64,
        num_blocks=8
    ):
        super().__init__()

        self.head = nn.Conv2d(
            3,
            features,
            3,
            padding=1
        )

        self.body = nn.Sequential(
            *[
                ResidualBlock(
                    features,
                    RES_SCALE
                )
                for _ in range(num_blocks)
            ]
        )

        self.body_conv = nn.Conv2d(
            features,
            features,
            3,
            padding=1
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(
                features,
                features * 4,
                3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(
                features,
                features * 4,
                3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            features,
            3,
            3,
            padding=1
        )

    def forward(self, x):

        x = self.head(x)

        residual = x

        x = self.body(x)

        x = self.body_conv(x)

        x = x + residual

        x = self.up1(x)

        x = self.up2(x)

        return self.tail(x)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n\n"
            "Train train_edsr_detail.py first."
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    features = checkpoint.get(
        "features",
        FEATURES
    )

    blocks = checkpoint.get(
        "num_blocks",
        NUM_BLOCKS
    )

    model = DetailEDSR(
        features,
        blocks
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = Image.open(
        path
    ).convert("RGB")

    array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        array
    ).permute(
        2,
        0,
        1
    ).unsqueeze(0)

    return tensor


# ============================================================
# VISUAL DETAIL ENHANCEMENT
# ============================================================

def enhance_detail(
    image,
    strength=0.50
):

    """
    Controlled unsharp/detail enhancement.

    This is deliberately used for the visual-enhancement
    prototype. It is NOT claiming to recover real pixels.
    """

    image = np.clip(
        image,
        0,
        1
    )

    uint8 = (
        image * 255.0
    ).astype(
        np.uint8
    )

    pil = Image.fromarray(
        uint8
    )

    blur = pil.filter(
        ImageFilter.GaussianBlur(
            radius=1.0
        )
    )

    original = np.asarray(
        pil,
        dtype=np.float32
    ) / 255.0

    blurred = np.asarray(
        blur,
        dtype=np.float32
    ) / 255.0

    high_frequency = (
        original - blurred
    )

    enhanced = (
        original
        + strength * high_frequency
    )

    return np.clip(
        enhanced,
        0,
        1
    )


# ============================================================
# UPSCALE
# ============================================================

def upscale(
    model,
    image,
    detail_strength=0.0
):

    image = image.to(
        DEVICE
    )

    with torch.no_grad():

        base = F.interpolate(
            image,
            scale_factor=SCALE,
            mode="bicubic",
            align_corners=False
        )

        detail = model(
            image
        )

        # The deployed checkpoint predicts a learned detail residual, which
        # is added once to the bicubic reconstruction.
        sr = base + detail

        sr = torch.clamp(
            sr,
            0,
            1
        )

    output = (
        sr.squeeze(0)
        .cpu()
        .numpy()
        .transpose(1, 2, 0)
    )

    # Cosmetic sharpening is optional and disabled by default.  It is not
    # part of the model's reconstruction.
    if detail_strength > 0:
        output = enhance_detail(output, strength=detail_strength)

    return output


# ============================================================
# SAVE
# ============================================================

def save_image(
    image,
    path
):

    image = (
        np.clip(
            image,
            0,
            1
        )
        * 255
    ).astype(
        np.uint8
    )

    Image.fromarray(
        image
    ).save(path)


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) < 3:

        print(
            "Usage:\n"
            "python inference.py "
            "input.png output.png"
        )

        raise SystemExit(1)

    input_path = sys.argv[1]

    output_path = sys.argv[2]

    print("=" * 60)
    print(
        "GeoSR - Visual Enhancement Inference"
    )
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Input: {input_path}"
    )

    print(
        f"Output: {output_path}"
    )

    model = load_model()

    image = load_image(
        input_path
    )

    output = upscale(
        model,
        image,
        detail_strength=0.35
    )

    save_image(
        output,
        output_path
    )

    print()
    print(
        "Enhancement complete."
    )

    print(
        f"Saved: {output_path}"
    )


if __name__ == "__main__":
    main()
