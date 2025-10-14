import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm


from models.mobile_sam_base import MobileSAMBase
from models.mobile_sam_adapter import MobileSAMAdapter
from models.mobile_sam_lora import MobileSAMLoRA

from losses import (
    CamouflagedObjectLoss,
    ShadowDetectionLoss,
    StructureSegmentationLoss
)

from PIL import Image
from torchvision import transforms
import glob

MOBILESAM_CHECPOINT_PATH = "./models/mobileSam/weights/mobile_sam.pt"

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, task_name, split="train", target_size=1024, transform=None, keep_original_size=False):
        self.img_paths = sorted(glob.glob(f"./data/{task_name}/{split}/images/*"))
        self.mask_paths = sorted(glob.glob(f"./data/{task_name}/{split}/masks/*"))
        assert len(self.img_paths) == len(self.mask_paths), "Images and masks mismatch!"
        
        self.target_size = target_size
        self.transform = transform
        self.keep_original_size = keep_original_size

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def _resize_and_pad(self, img):
        """Resize image to fit inside target_size, pad remaining areas to get exact target_size."""
        w, h = img.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)

        # Create new padded image
        new_img = Image.new("RGB", (self.target_size, self.target_size))
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        new_img.paste(img_resized, (pad_left, pad_top))
        return new_img

    def _resize_and_pad_mask(self, mask):
        w, h = mask.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        mask_resized = mask.resize((new_w, new_h), resample=Image.NEAREST)

        new_mask = Image.new("L", (self.target_size, self.target_size))
        pad_left = (self.target_size - new_w) // 2
        pad_top = (self.target_size - new_h) // 2
        new_mask.paste(mask_resized, (pad_left, pad_top))
        return new_mask

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")
        orig_size = (img.height, img.width)

        if not self.keep_original_size:
            img = self._resize_and_pad(img)
            mask = self._resize_and_pad_mask(mask)

        img = self.to_tensor(img)
        mask = self.to_tensor(mask)
        mask = (mask > 0.5).float()  # binarize

        if self.keep_original_size:
            return {"image": img, "mask": mask, "original_size": orig_size}
        else:
            return img, mask

def get_loss_function(task_name, **kwargs):
    if task_name == "cod":
        return CamouflagedObjectLoss(**kwargs)
    elif task_name == "shadow":
        return ShadowDetectionLoss(**kwargs)
    elif task_name == "structure":
        return StructureSegmentationLoss(**kwargs)
    else:
        raise ValueError("Unknown task name")

# -----------------------------
# Utility functions
# -----------------------------
# def count_trainable_params(model):
#     total = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total, trainable


def save_checkpoint(model, optimizer, epoch, val_loss, save_dir, best=False):
    ckpt_name = f"best_model.pth" if best else f"checkpoint_epoch{epoch}.pth"
    path = os.path.join(save_dir, ckpt_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }, path)
    print(f"Saved checkpoint: {path}")


def log_metrics(log_path, epoch, train_loss, val_loss, lr):
    header = ["epoch", "train_loss", "val_loss", "lr"]
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([epoch, train_loss, val_loss, lr])


# -----------------------------
# Training and validation
# -----------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0

    for images, masks in tqdm(dataloader, desc="Training", leave=False):
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()  # ensure float

        optimizer.zero_grad()

        batched_input = [
            {
                "image": img, 
                "original_size": (mask.shape[1], mask.shape[2])
            }
            for img, mask in zip(images, masks)
        ]

        with torch.amp.autocast(device_type=device.type):
            outputs = model(batched_input, multimask_output=False)

            # Stack low-res logits: [B, 1, 1, H, W] -> [B, 1, H, W]
            preds = torch.stack([out["low_res_logits"] for out in outputs], dim=0)  # stack along batch
            preds = preds.float()  # safe, won't detach            
            preds = preds.squeeze(2)  # remove singleton channel dim

            # Resize to match mask if needed
            if preds.shape[-2:] != masks.shape[-2:]:
                preds = F.interpolate(preds, size=masks.shape[-2:], mode="bilinear", align_corners=False)

            preds = preds.float()  # ensure float32

            loss = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    return running_loss / len(dataloader)



@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0

    for sample in tqdm(dataloader, desc="Validation", leave=False):
        # sample is a dict: {"image": img, "mask": mask, "original_size": (H,W)}
        img = sample["image"][0].to(device)    # [C,H,W]
        mask = sample["mask"][0].to(device)    # [1,H,W]

        orig_size = sample["original_size"]

        # Wrap image in a list for model input
        batched_input = [{"image": img, "original_size": orig_size}]
        outputs = model(batched_input, multimask_output=False)

        output_mask = outputs[0]["masks"].float()

        # Resize to match mask
        output_mask_resized = torch.nn.functional.interpolate(
            output_mask, size=mask.shape[-2:], mode="bilinear", align_corners=False
        )

        # squeeze batch & channel dims to match mask
        val_loss += criterion(output_mask_resized, mask.unsqueeze(0)).item()  # mask[0] -> [H,W]
        
    return val_loss / len(dataloader)

# -----------------------------
# Main training loop
# -----------------------------
def train_model(
    variant="adapter",
    num_epochs=10,
    lr=1e-4,
    batch_size=2,
    scheduler_type="cosine",
    task_name="cod",
    save_root="checkpoints"
):
    save_root = save_root + "/" + variant + "/" + task_name
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Model: {variant}")
    print(f"[INFO] Schdeuler: {scheduler_type}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Number of epochs: {num_epochs}\n")
    
    # Timestamped save dir
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.csv")

    # Initialize model
    if variant == "base":
        model = MobileSAMBase(pretrained_checkpoint=MOBILESAM_CHECPOINT_PATH)
    elif variant == "adapter":
        model = MobileSAMAdapter(pretrained_checkpoint=MOBILESAM_CHECPOINT_PATH)
    elif variant == "lora":
        model = MobileSAMLoRA(rank=8, alpha=32, pretrained_checkpoint=MOBILESAM_CHECPOINT_PATH)
    else:
        raise ValueError("Invalid variant")

    model.to(device)
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)
    
    # total, trainable = count_trainable_params(model)
    # print(f"\n[INFO] Params: {trainable/1e6:.3f}M trainable / {total/1e6:.3f}M total\n")

    # Training
    train_dataset = SegmentationDataset(task_name="cod", split="train", target_size=1024, keep_original_size=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Validation
    val_dataset = SegmentationDataset(task_name="cod", split="val", keep_original_size=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Loss function
    criterion = get_loss_function(task_name=task_name)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    scaler = torch.amp.GradScaler()
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        

        val_loss = validate(model, val_loader, criterion, device)
        
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {lr_now:.6f}")

        # Log to CSV
        log_metrics(log_path, epoch+1, train_loss, val_loss, lr_now)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, epoch + 1, val_loss, save_dir, best=True)

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, val_loss, save_dir)

    print(f"\nTraining complete! Logs saved at: {log_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_model(
        variant="lora",          # "base", "adapter", or "lora"
        num_epochs=10,
        lr=1e-3,
        scheduler_type="cosine",    # or "step"
        task_name="cod",            # "cod", "shadow", "structure"
        batch_size=8,
        save_root="checkpoints",
    )