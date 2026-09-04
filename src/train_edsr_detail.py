import os
import math
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# GeoSR - Residual Detail Enhancement EDSR
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

LR_DIR = os.path.join(
    ROOT,
    "data",
    "processed",
    "train",
    "LR"
)

HR_DIR = os.path.join(
    ROOT,
    "data",
    "processed",
    "train",
    "HR"
)

MODEL_DIR = os.path.join(
    ROOT,
    "models"
)

MODEL_OUT = os.path.join(
    MODEL_DIR,
    "edsr_visual_best.pth"
)


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SCALE = 4

# IMPORTANT:
# Your dataset contains some 16x16 LR tiles.
# Therefore the training crop cannot be 32x32.
#
# We use 8x8 -> 32x32 so BOTH 16x16 and larger tiles
# can participate in training.
LR_CROP = 8

BATCH_SIZE = 8

EPOCHS = 5

LEARNING_RATE = 1e-4

NUM_WORKERS = 0

MAX_SAMPLES = 12000

FEATURES = 64

NUM_BLOCKS = 8

RES_SCALE = 0.1


# ============================================================
# RESIDUAL BLOCK
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
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1
        )

        self.res_scale = res_scale

    def forward(self, x):

        residual = self.conv1(x)

        residual = F.relu(
            residual,
            inplace=True
        )

        residual = self.conv2(
            residual
        )

        return x + residual * self.res_scale


# ============================================================
# DETAIL EDSR
# ============================================================

class DetailEDSR(nn.Module):

    """
    EDSR-style network that predicts the high-frequency
    residual/detail rather than recreating the entire image.

    Final output:

        SR = Bicubic(LR) + predicted_detail
    """

    def __init__(
        self,
        features=64,
        num_blocks=8
    ):
        super().__init__()

        self.head = nn.Conv2d(
            3,
            features,
            kernel_size=3,
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
            kernel_size=3,
            padding=1
        )

        # 2x
        self.up1 = nn.Sequential(
            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        # another 2x
        # Total = 4x
        self.up2 = nn.Sequential(
            nn.Conv2d(
                features,
                features * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )

        self.tail = nn.Conv2d(
            features,
            3,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        x = self.head(x)

        skip = x

        x = self.body(x)

        x = self.body_conv(x)

        x = x + skip

        x = self.up1(x)

        x = self.up2(x)

        detail = self.tail(x)

        return detail


# ============================================================
# DATASET
# ============================================================

class SRDataset(Dataset):

    def __init__(
        self,
        lr_dir,
        hr_dir,
        crop_size=8,
        scale=4,
        max_samples=None
    ):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale = scale

        if not os.path.isdir(self.lr_dir):

            raise RuntimeError(
                "LR directory does not exist:\n"
                f"{self.lr_dir}"
            )

        if not os.path.isdir(self.hr_dir):

            raise RuntimeError(
                "HR directory does not exist:\n"
                f"{self.hr_dir}"
            )

        self.pairs = []

        skipped_small = 0
        skipped_missing = 0
        skipped_invalid = 0

        valid_extensions = (
            ".png",
            ".jpg",
            ".jpeg",
            ".tif",
            ".tiff"
        )

        for filename in sorted(
            os.listdir(self.lr_dir)
        ):

            if not filename.lower().endswith(
                valid_extensions
            ):
                continue

            lr_path = os.path.join(
                self.lr_dir,
                filename
            )

            hr_path = os.path.join(
                self.hr_dir,
                filename
            )

            if not os.path.exists(hr_path):

                skipped_missing += 1

                continue

            try:

                with Image.open(lr_path) as lr_img:

                    lr_w, lr_h = lr_img.size

                with Image.open(hr_path) as hr_img:

                    hr_w, hr_h = hr_img.size

                required_hr = (
                    crop_size * scale
                )

                if (
                    lr_w < crop_size
                    or lr_h < crop_size
                ):

                    skipped_small += 1

                    continue

                if (
                    hr_w < required_hr
                    or hr_h < required_hr
                ):

                    skipped_small += 1

                    continue

                self.pairs.append(
                    (
                        lr_path,
                        hr_path
                    )
                )

            except Exception:

                skipped_invalid += 1

                continue

        if max_samples is not None:

            self.pairs = self.pairs[
                :max_samples
            ]

        if len(self.pairs) == 0:

            raise RuntimeError(
                "No valid LR/HR pairs found.\n\n"
                f"LR: {self.lr_dir}\n"
                f"HR: {self.hr_dir}\n\n"
                f"Required minimum LR size: "
                f"{crop_size}x{crop_size}\n"
                f"Required minimum HR size: "
                f"{crop_size * scale}x"
                f"{crop_size * scale}"
            )

        print(
            f"Valid pairs: {len(self.pairs)}"
        )

        print(
            f"Skipped undersized pairs: "
            f"{skipped_small}"
        )

        print(
            f"Skipped missing HR pairs: "
            f"{skipped_missing}"
        )

        print(
            f"Skipped invalid pairs: "
            f"{skipped_invalid}"
        )

    def __len__(self):

        return len(self.pairs)

    def __getitem__(self, index):

        lr_path, hr_path = self.pairs[
            index
        ]

        lr = np.asarray(
            Image.open(
                lr_path
            ).convert("RGB"),
            dtype=np.float32
        ) / 255.0

        hr = np.asarray(
            Image.open(
                hr_path
            ).convert("RGB"),
            dtype=np.float32
        ) / 255.0

        lr = torch.from_numpy(
            lr
        ).permute(
            2,
            0,
            1
        )

        hr = torch.from_numpy(
            hr
        ).permute(
            2,
            0,
            1
        )

        lr_h, lr_w = lr.shape[-2:]

        crop = self.crop_size

        # ----------------------------------------------------
        # Random LR crop
        # ----------------------------------------------------

        if lr_h == crop:

            y = 0

        else:

            y = random.randint(
                0,
                lr_h - crop
            )

        if lr_w == crop:

            x = 0

        else:

            x = random.randint(
                0,
                lr_w - crop
            )

        lr = lr[
            :,
            y:y + crop,
            x:x + crop
        ]

        # ----------------------------------------------------
        # Corresponding HR crop
        # ----------------------------------------------------

        hr_crop = crop * self.scale

        hr_y = y * self.scale
        hr_x = x * self.scale

        # In case dimensions differ slightly because of
        # preprocessing, make sure the requested HR crop
        # remains inside the image.
        hr_h, hr_w = hr.shape[-2:]

        if (
            hr_y + hr_crop > hr_h
            or hr_x + hr_crop > hr_w
        ):

            hr_y = max(
                0,
                min(
                    hr_y,
                    hr_h - hr_crop
                )
            )

            hr_x = max(
                0,
                min(
                    hr_x,
                    hr_w - hr_crop
                )
            )

        hr = hr[
            :,
            hr_y:hr_y + hr_crop,
            hr_x:hr_x + hr_crop
        ]

        # ----------------------------------------------------
        # Make absolutely sure dimensions match.
        # ----------------------------------------------------

        if (
            lr.shape[-2:] !=
            (
                crop,
                crop
            )
        ):

            raise RuntimeError(
                f"Invalid LR crop from "
                f"{os.path.basename(lr_path)}: "
                f"{tuple(lr.shape[-2:])}"
            )

        if (
            hr.shape[-2:] !=
            (
                hr_crop,
                hr_crop
            )
        ):

            raise RuntimeError(
                f"Invalid HR crop from "
                f"{os.path.basename(hr_path)}: "
                f"{tuple(hr.shape[-2:])}"
            )

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if random.random() < 0.5:

            lr = torch.flip(
                lr,
                dims=[2]
            )

            hr = torch.flip(
                hr,
                dims=[2]
            )

        if random.random() < 0.5:

            lr = torch.flip(
                lr,
                dims=[1]
            )

            hr = torch.flip(
                hr,
                dims=[1]
            )

        if random.random() < 0.25:

            lr = lr.transpose(
                1,
                2
            )

            hr = hr.transpose(
                1,
                2
            )

        return lr.contiguous(), hr.contiguous()


# ============================================================
# GRADIENT
# ============================================================

def gradient_map(x):

    gx = (
        x[:, :, :, 1:]
        -
        x[:, :, :, :-1]
    )

    gy = (
        x[:, :, 1:, :]
        -
        x[:, :, :-1, :]
    )

    return gx, gy


# ============================================================
# LAPLACIAN
# ============================================================

def laplacian(x):

    kernel = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ],
        dtype=x.dtype,
        device=x.device
    )

    kernel = kernel.view(
        1,
        1,
        3,
        3
    )

    kernel = kernel.repeat(
        x.shape[1],
        1,
        1,
        1
    )

    return F.conv2d(
        x,
        kernel,
        padding=1,
        groups=x.shape[1]
    )


# ============================================================
# HIGH FREQUENCY
# ============================================================

def high_frequency(x):

    blur = F.avg_pool2d(
        x,
        kernel_size=5,
        stride=1,
        padding=2
    )

    return x - blur


# ============================================================
# LOSS
# ============================================================

def detail_loss(
    prediction,
    target
):

    # --------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------

    l1 = F.l1_loss(
        prediction,
        target
    )

    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    pred_gx, pred_gy = gradient_map(
        prediction
    )

    target_gx, target_gy = gradient_map(
        target
    )

    gradient_loss = (
        F.l1_loss(
            pred_gx,
            target_gx
        )
        +
        F.l1_loss(
            pred_gy,
            target_gy
        )
    )

    # --------------------------------------------------------
    # Laplacian detail
    # --------------------------------------------------------

    pred_lap = laplacian(
        prediction
    )

    target_lap = laplacian(
        target
    )

    lap_loss = F.l1_loss(
        pred_lap,
        target_lap
    )

    # --------------------------------------------------------
    # High-frequency detail
    # --------------------------------------------------------

    pred_hf = high_frequency(
        prediction
    )

    target_hf = high_frequency(
        target
    )

    hf_loss = F.l1_loss(
        pred_hf,
        target_hf
    )

    # --------------------------------------------------------
    # Combined loss
    # --------------------------------------------------------

    loss = (
        0.55 * l1
        +
        0.20 * gradient_loss
        +
        0.15 * lap_loss
        +
        0.10 * hf_loss
    )

    return (
        loss,
        l1,
        gradient_loss,
        lap_loss,
        hf_loss
    )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    prediction,
    target
):

    mse = F.mse_loss(
        prediction,
        target
    ).item()

    if mse <= 1e-12:

        return 100.0

    return (
        10.0
        *
        math.log10(
            1.0 / mse
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "GeoSR - Residual Detail Enhancement EDSR"
    )
    print("=" * 60)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"LR directory: {LR_DIR}"
    )

    print(
        f"HR directory: {HR_DIR}"
    )

    print()

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = SRDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        crop_size=LR_CROP,
        scale=SCALE,
        max_samples=MAX_SAMPLES
    )

    print()

    print(
        f"Samples: {len(dataset)}"
    )

    print(
        f"Crop: "
        f"{LR_CROP}x{LR_CROP} LR "
        f"-> "
        f"{LR_CROP * SCALE}x"
        f"{LR_CROP * SCALE} HR"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )

    print(
        f"Features: {FEATURES}"
    )

    print(
        f"Residual blocks: {NUM_BLOCKS}"
    )

    print()

    print("Architecture:")
    print(
        "  LR"
    )
    print(
        "   ↓"
    )
    print(
        "  EDSR"
    )
    print(
        "   ↓"
    )
    print(
        "  learned detail"
    )
    print(
        "   +"
    )
    print(
        "  bicubic baseline"
    )
    print(
        "   ↓"
    )
    print(
        "  enhanced SR"
    )

    print()

    print("Loss:")
    print(
        "  55% L1 reconstruction"
    )

    print(
        "  20% gradient"
    )

    print(
        "  15% Laplacian detail"
    )

    print(
        "  10% high-frequency"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = DetailEDSR(
        features=FEATURES,
        num_blocks=NUM_BLOCKS
    ).to(DEVICE)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999)
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[3, 4],
        gamma=0.5
    )

    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_psnr = -float("inf")

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # TRAINING
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        total_l1 = 0.0
        total_gradient = 0.0
        total_lap = 0.0
        total_hf = 0.0
        total_psnr = 0.0

        progress = tqdm(
            loader,
            desc=(
                f"Epoch "
                f"{epoch}/{EPOCHS}"
            ),
            unit="batch"
        )

        for lr_img, hr_img in progress:

            lr_img = lr_img.to(
                DEVICE,
                non_blocking=True
            )

            hr_img = hr_img.to(
                DEVICE,
                non_blocking=True
            )

            # ------------------------------------------------
            # Bicubic baseline
            # ------------------------------------------------

            base = F.interpolate(
                lr_img,
                size=hr_img.shape[-2:],
                mode="bicubic",
                align_corners=False
            )

            # ------------------------------------------------
            # Network predicts detail
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            detail = model(
                lr_img
            )

            # ------------------------------------------------
            # Add learned detail to bicubic
            # ------------------------------------------------

            prediction = (
                base + detail
            )

            prediction = torch.clamp(
                prediction,
                0.0,
                1.0
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            (
                loss,
                l1,
                gradient,
                lap,
                hf
            ) = detail_loss(
                prediction,
                hr_img
            )

            # ------------------------------------------------
            # Backprop
            # ------------------------------------------------

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            total_loss += loss.item()

            total_l1 += l1.item()

            total_gradient += (
                gradient.item()
            )

            total_lap += lap.item()

            total_hf += hf.item()

            total_psnr += calculate_psnr(
                prediction.detach(),
                hr_img
            )

            progress.set_postfix(
                loss=f"{loss.item():.5f}"
            )

        scheduler.step()

        n = len(loader)

        avg_loss = (
            total_loss / n
        )

        avg_l1 = (
            total_l1 / n
        )

        avg_gradient = (
            total_gradient / n
        )

        avg_lap = (
            total_lap / n
        )

        avg_hf = (
            total_hf / n
        )

        avg_psnr = (
            total_psnr / n
        )

        # ====================================================
        # EPOCH RESULTS
        # ====================================================

        print()

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )

        print(
            f"Loss:       {avg_loss:.6f}"
        )

        print(
            f"L1:         {avg_l1:.6f}"
        )

        print(
            f"Gradient:   {avg_gradient:.6f}"
        )

        print(
            f"Laplacian:  {avg_lap:.6f}"
        )

        print(
            f"HighFreq:   {avg_hf:.6f}"
        )

        print(
            f"PSNR:       {avg_psnr:.2f} dB"
        )

        print(
            f"LR:         "
            f"{optimizer.param_groups[0]['lr']:.2e}"
        )

        # ====================================================
        # SAVE BEST
        # ====================================================

        if avg_psnr > best_psnr:

            best_psnr = avg_psnr

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "psnr":
                        best_psnr,

                    "epoch":
                        epoch,

                    "scale":
                        SCALE,

                    "features":
                        FEATURES,

                    "num_blocks":
                        NUM_BLOCKS,

                    "crop_size":
                        LR_CROP,

                    "architecture":
                        "residual_detail_edsr"
                },
                MODEL_OUT
            )

            print(
                f"Saved best model: "
                f"{best_psnr:.2f} dB"
            )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()

    print("=" * 60)
    print(
        "TRAINING COMPLETE"
    )
    print("=" * 60)

    print(
        f"Best PSNR: "
        f"{best_psnr:.2f} dB"
    )

    print(
        f"Model: {MODEL_OUT}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()