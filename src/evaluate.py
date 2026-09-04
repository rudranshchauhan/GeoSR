import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "edsr_best.pth")
VAL_LR = os.path.join(ROOT, "data", "processed", "val", "LR")
VAL_HR = os.path.join(ROOT, "data", "processed", "val", "HR")
OUTPUT_DIR = os.path.join(ROOT, "outputs", "evaluation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SAMPLES = 100
NUM_TRAIN_SAMPLES = 8000
NUM_VAL_SAMPLES = 1000
SEED = 42


class ResidualBlock(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, 3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self, features, scale):
        super().__init__()
        layers = []

        if scale == 4:
            layers.extend([
                nn.Conv2d(features, features * 4, 3, padding=1),
                nn.PixelShuffle(2),
                nn.Conv2d(features, features * 4, 3, padding=1),
                nn.PixelShuffle(2)
            ])
        elif scale == 2:
            layers.extend([
                nn.Conv2d(features, features * 4, 3, padding=1),
                nn.PixelShuffle(2)
            ])
        else:
            raise ValueError("Only 2x and 4x scaling are supported.")

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class EDSR(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        features=64,
        num_blocks=8,
        scale=4
    ):
        super().__init__()

        self.head = nn.Conv2d(in_channels, features, 3, padding=1)

        self.body = nn.Sequential(
            *[ResidualBlock(features) for _ in range(num_blocks)]
        )

        self.body_conv = nn.Conv2d(features, features, 3, padding=1)
        self.upsample = UpsampleBlock(features, scale)
        self.tail = nn.Conv2d(features, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.head(x)
        residual = x
        x = self.body(x)
        x = self.body_conv(x)
        x = x + residual
        x = self.upsample(x)
        return self.tail(x)


def load_image(path):
    image = Image.open(path).convert("RGB")
    image = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(image).permute(2, 0, 1)


def calculate_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)

    if mse.item() == 0:
        return 100.0

    return 10 * torch.log10(1.0 / mse).item()


def get_files(directory):
    return sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"))
    ])


def save_visuals(lr, bicubic, sr, hr, index):
    lr = lr.permute(1, 2, 0).cpu().numpy()
    bicubic = bicubic.permute(1, 2, 0).cpu().numpy()
    sr = sr.permute(1, 2, 0).cpu().numpy()
    hr = hr.permute(1, 2, 0).cpu().numpy()

    lr = Image.fromarray((np.clip(lr, 0, 1) * 255).astype(np.uint8))
    bicubic = Image.fromarray((np.clip(bicubic, 0, 1) * 255).astype(np.uint8))
    sr = Image.fromarray((np.clip(sr, 0, 1) * 255).astype(np.uint8))
    hr = Image.fromarray((np.clip(hr, 0, 1) * 255).astype(np.uint8))

    width, height = hr.size
    lr = lr.resize((width, height), Image.Resampling.BICUBIC)

    canvas = Image.new("RGB", (width * 4, height))

    canvas.paste(lr, (0, 0))
    canvas.paste(bicubic, (width, 0))
    canvas.paste(sr, (width * 2, 0))
    canvas.paste(hr, (width * 3, 0))

    canvas.save(
        os.path.join(
            OUTPUT_DIR,
            f"comparison_{index:02d}.png"
        )
    )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("GeoSR Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Model: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = EDSR(
        in_channels=checkpoint.get("in_channels", 3),
        out_channels=checkpoint.get("out_channels", 3),
        features=checkpoint.get("features", 64),
        num_blocks=checkpoint.get("num_blocks", 8),
        scale=checkpoint.get("scale", 4)
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    lr_files = get_files(VAL_LR)
    hr_files = get_files(VAL_HR)

    if len(lr_files) != len(hr_files):
        raise RuntimeError(
            f"LR/HR count mismatch: {len(lr_files)} vs {len(hr_files)}"
        )

    total = min(
        len(lr_files),
        NUM_TRAIN_SAMPLES + NUM_VAL_SAMPLES
    )

    lr_files = lr_files[:total]
    hr_files = hr_files[:total]

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(total, generator=generator).tolist()

    val_size = min(
        NUM_VAL_SAMPLES,
        max(1, int(total * 0.1))
    )

    val_indices = indices[total - val_size:]

    count = min(MAX_SAMPLES, len(val_indices))

    print(f"Training/validation pool: {total}")
    print(f"Reproduced validation set: {len(val_indices)}")
    print(f"Evaluating: {count}")
    print("")

    edsr_scores = []
    bicubic_scores = []

    with torch.no_grad():
        for i, index in enumerate(val_indices[:count]):
            lr = load_image(lr_files[index]).unsqueeze(0).to(DEVICE)
            hr = load_image(hr_files[index]).unsqueeze(0).to(DEVICE)

            sr = torch.clamp(model(lr), 0, 1)

            bicubic = F.interpolate(
                lr,
                size=hr.shape[-2:],
                mode="bicubic",
                align_corners=False
            )

            edsr_scores.append(
                calculate_psnr(sr, hr)
            )

            bicubic_scores.append(
                calculate_psnr(bicubic, hr)
            )

            if i < 5:
                save_visuals(
                    lr.squeeze(0),
                    bicubic.squeeze(0),
                    sr.squeeze(0),
                    hr.squeeze(0),
                    i + 1
                )

            if (i + 1) % 10 == 0 or i + 1 == count:
                print(f"Progress: {i + 1}/{count}")

    edsr_avg = sum(edsr_scores) / len(edsr_scores)
    bicubic_avg = sum(bicubic_scores) / len(bicubic_scores)

    print("")
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Bicubic PSNR: {bicubic_avg:.2f} dB")
    print(f"EDSR PSNR:    {edsr_avg:.2f} dB")
    print(f"Improvement:  {edsr_avg - bicubic_avg:+.2f} dB")
    print("=" * 60)
    print("")
    print(f"Visual comparisons:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
