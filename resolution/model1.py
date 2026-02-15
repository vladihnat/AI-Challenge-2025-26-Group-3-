"""
Grain variety classification model

What this model does
--------------------
We classify RGB images of grains into 8 classes (labels 1..8). The core model is a
standard ResNet18 architecture (BasicBlock, layers [2,2,2,2]) implemented *from scratch*,
i.e., without importing torchvision.

Why ResNet18 (and why "no torchvision")
---------------------------------------
- We tested a classical ML baseline (hand-crafted features + tree model). It was easy to run,
  but it plateaued lower because fixed features (histograms/statistics/downsampled pixels)
  are limited for fine-grained visual differences.
- A CNN can learn discriminative texture/shape features directly from pixels, which is what we need here.
- ResNet18 is a good trade-off: strong enough for fine-grained classification, still fast enough
  for the time limits of the competition.
- On Codabench, torchvision was not available in the container ("torchvision is required but not available").
  To keep the same architecture and be compatible, we re-implemented ResNet18 directly in this file.

Features / preprocessing
------------------------
Input images are (252,252,3). We:
1) Convert uint8 -> float in [0,1] (divide by 255).
2) Convert to PyTorch tensor (C,H,W).
3) Crop to 224x224:
   - train: random crop (adds translation/zoom robustness)
   - val/test: center crop (stable evaluation)
4) Normalize with ImageNet mean/std (helps optimization even without pretraining).

Data augmentation (train only)
------------------------------
- Random horizontal flip
- Random vertical flip
- Random rotation by {0,90,180,270}
- Random crop (224)
These augmentations are simple, cheap, and improve robustness to orientation/position.

Training choices
----------------
- Loss: CrossEntropyLoss
- Optimizer: AdamW (stable and commonly used for CNNs)
- Regularization: weight_decay (L2)
- LR schedule: ReduceLROnPlateau on validation accuracy
  (if val acc stops improving, we reduce lr by factor 0.5)
- Early stopping: stop if val acc does not improve for several epochs

Important implementation details
-------------------------------------------------------
1) Label mapping:
   The dataset labels are 1..8, but CrossEntropyLoss expects 0..(C-1).
   So during training we use y_train = y_raw - 1, and during prediction we output argmax + 1.
   This avoids CUDA "device-side assert" errors.
2) DataLoader shared memory issue:
   On Codabench, using multiple workers can crash with "bus error" due to limited /dev/shm.
   We set num_workers=0 and pin_memory=False to be safe.

How to read the logs
--------------------
Each epoch prints:
- train_loss: average training loss
- train_acc: accuracy on training batches
- val_acc: accuracy on the held-out validation split
- lr: current learning rate (useful to see when ReduceLROnPlateau triggers)

This file is self-contained: only requires numpy + torch.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# =========================================================
# Reproducibility helpers
# =========================================================
def seed_everything(seed: int = 42) -> None:
    """Fix random seeds for more stable experiments."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior is slower but makes debugging easier.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def stratified_split_indices(
    y: np.ndarray, val_ratio: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Stratified train/val split indices, to keep label distribution similar.
    This avoids "val set missing some classes" by accident.
    """
    rng = np.random.RandomState(seed)
    y = np.asarray(y)

    train_idx, val_idx = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_ratio)))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


# =========================================================
# Dataset: numpy -> torch tensor + preprocessing/augmentation
# =========================================================
class NumpyImageDataset(Dataset):
    """
    X: numpy array (N,H,W,C), dtype uint8 or float
    y: numpy array (N,), labels in 0..7 (for training) or None (for test)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        augment: bool = False,
        crop_size: int = 224,
    ):
        self.X = X
        self.y = y
        self.augment = bool(augment)
        self.crop_size = int(crop_size)

        # ImageNet normalization (works well for many CNNs)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _rand_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Random 224x224 crop for training."""
        _, H, W = x.shape
        s = self.crop_size
        if H < s or W < s:
            return x
        top = int(torch.randint(0, H - s + 1, (1,)).item())
        left = int(torch.randint(0, W - s + 1, (1,)).item())
        return x[:, top : top + s, left : left + s]

    def _center_crop(self, x: torch.Tensor) -> torch.Tensor:
        """Center crop for validation/test (deterministic)."""
        _, H, W = x.shape
        s = self.crop_size
        if H < s or W < s:
            return x
        top = (H - s) // 2
        left = (W - s) // 2
        return x[:, top : top + s, left : left + s]

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Cheap augmentations that help generalization."""
        # flips
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[2])  # horizontal
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[1])  # vertical

        # rotate 0/90/180/270 (fast, no interpolation)
        k = int(torch.randint(0, 4, (1,)).item())
        if k:
            x = torch.rot90(x, k, dims=[1, 2])

        # random crop
        x = self._rand_crop(x)
        return x

    def __getitem__(self, idx: int):
        img = self.X[idx]

        # 1) uint8 -> float32
        if img.dtype != np.float32:
            img = img.astype(np.float32, copy=False)

        # 2) scale to [0,1] if needed
        if img.max() > 1.0:
            img = img / 255.0

        # 3) (H,W,C) -> (C,H,W)
        x = torch.from_numpy(img).permute(2, 0, 1).contiguous()

        # 4) augmentation / crop
        if self.augment:
            x = self._augment(x)
        else:
            x = self._center_crop(x)

        # 5) normalize
        x = (x - self.mean) / self.std

        if self.y is None:
            return x
        return x, int(self.y[idx])


# =========================================================
# ResNet18 implementation (no torchvision)
# =========================================================
def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding."""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution (for downsampling shortcut)."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock (used in ResNet18/34)."""

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or inplanes != planes:
            # match dimensions between identity and residual
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride), nn.BatchNorm2d(planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


class ResNet(nn.Module):
    """ResNet backbone with configurable blocks per layer."""

    def __init__(self, block, layers: list[int], num_classes: int = 8):
        super().__init__()
        self.inplanes = 64

        # ResNet stem: 224 -> 112 -> 56
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )  # 224->112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 112->56

        # 4 stages: 56 -> 28 -> 14 -> 7
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)  # 56
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 28
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 14
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Kaiming init for conv, and nice defaults for BN
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(
        self, block, planes: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [block(self.inplanes, planes, stride)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def resnet18_no_torchvision(num_classes: int = 8) -> ResNet:
    """Factory for standard ResNet18: layers [2,2,2,2]."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


# =========================================================
# Codabench Model wrapper (required API: __init__, fit, predict)
# =========================================================
class Model:
    def __init__(self):
        print("[*] Initializing ResNet18 (no torchvision, from scratch)")

        # Reproducibility
        self.seed = 42
        seed_everything(self.seed)

        # Labels are 1..8 in the dataset, but CE loss needs 0..7
        self.num_classes = 8
        self.label_offset = 1

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[*] Device:", self.device)

        # Codabench stability: avoid multiprocessing DataLoader shared memory issues
        self.num_workers = 0
        self.pin_memory = False

        # Hyperparameters (final tuned version)
        self.batch_size = 48 if self.device.type == "cuda" else 24
        self.max_epochs = 25
        self.lr = 3e-4
        self.weight_decay = 1e-4
        self.patience = 6
        self.val_ratio = 0.15

        # Build network
        self.net = resnet18_no_torchvision(num_classes=self.num_classes).to(self.device)

        # Mixed precision: disabled for stability (also avoids some AMP issues on certain setups)
        self.use_amp = False

        # We keep a scaler object for code simplicity, but disabled
        self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.is_fitted = False

    def fit(self, train_data: dict) -> None:
        print("[*] Fitting model...")

        X = train_data["X"]
        y_raw = np.asarray(train_data["y"]).astype(np.int64)

        # Map labels 1..8 -> 0..7
        y = y_raw - self.label_offset
        y_min, y_max = int(y.min()), int(y.max())
        if y_min < 0 or y_max >= self.num_classes:
            raise ValueError(
                f"Label out of range after mapping: min={y_min}, max={y_max}"
            )

        # Train/val split
        tr_idx, va_idx = stratified_split_indices(y, self.val_ratio, self.seed)
        print(f"[*] Split: train={len(tr_idx)} val={len(va_idx)}")

        train_ds = NumpyImageDataset(X[tr_idx], y[tr_idx], augment=True, crop_size=224)
        val_ds = NumpyImageDataset(X[va_idx], y[va_idx], augment=False, crop_size=224)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Plateau scheduler: reduce lr when val_acc stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, min_lr=1e-6
        )

        best_val_acc = -1.0
        best_state = None
        no_improve = 0

        for epoch in range(1, self.max_epochs + 1):
            # ---- train ----
            self.net.train()
            total, correct = 0, 0
            running_loss = 0.0

            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                # AMP disabled (enabled=self.use_amp => False)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.net(xb)
                    loss = criterion(logits, yb)

                # scaler is disabled, so this behaves like normal backward/step
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                running_loss += float(loss.item()) * int(xb.size(0))
                preds = logits.argmax(dim=1)
                correct += int((preds == yb).sum().item())
                total += int(xb.size(0))

            train_loss = running_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            # ---- validation ----
            self.net.eval()
            v_total, v_correct = 0, 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    with torch.cuda.amp.autocast(enabled=self.use_amp):
                        logits = self.net(xb)
                    preds = logits.argmax(dim=1)
                    v_correct += int((preds == yb).sum().item())
                    v_total += int(xb.size(0))

            val_acc = v_correct / max(v_total, 1)

            print(
                f"[*] Epoch {epoch:02d}/{self.max_epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
            )

            # Scheduler step uses validation metric
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"[*] lr={current_lr:.2e}")

            # Early stopping bookkeeping
            if val_acc > best_val_acc + 1e-4:
                best_val_acc = val_acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in self.net.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print("[*] Early stopping")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self.is_fitted = True
        print(f"[*] Done. Best val_acc={best_val_acc:.4f}")

    def predict(self, test_data: dict) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        X = test_data["X"]
        test_ds = NumpyImageDataset(X, y=None, augment=False, crop_size=224)
        test_loader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        self.net.eval()
        preds_all = []
        with torch.no_grad():
            for xb in test_loader:
                xb = xb.to(self.device)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.net(xb)
                preds_all.append(logits.argmax(dim=1).detach().cpu().numpy())

        y0 = np.concatenate(preds_all, axis=0).astype(np.int64)
        return y0 + self.label_offset  # back to 1..8
