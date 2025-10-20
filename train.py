import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append("/content/mobileSam_adapter")

import yaml
import argparse
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


def batch_iou(preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) for a batch of predictions and targets.

    IoU is a common evaluation metric for semantic segmentation tasks which measures the overlap
    between two boundaries. It's defined as the area of the intersection divided by the area of the union.

    Args:
    preds (torch.Tensor): The predicted segmentations with shape [batch_size, 1, height, width].
    targets (torch.Tensor): The ground truth segmentations with the same shape as predictions.
    threshold (float): A threshold value to convert the predicted probabilities into a binary mask.

    Returns:
    torch.Tensor: A tensor of IoU values for each pair in the batch with shape [batch_size, 1].
    """
    
    # Binarize predictions based on the threshold. Convert to float for subsequent calculations.
    preds = (preds >= threshold).float()
    
    # Compute the intersection by element-wise multiplication of predictions and targets,
    # followed by summing over the spatial dimensions (height and width).
    intersection = (preds * targets).sum((2, 3))
    
    # Compute the union by summing predictions and targets separately, followed by summing over
    # spatial dimensions and then subtracting the intersection to avoid double counting.
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - intersection
    
    # Calculate IoU by dividing intersection by union. Adding a small epsilon (1e-6) to avoid
    # division by zero errors.
    iou = intersection / (union + 1e-6)
    
    # Return the computed IoU values for the batch.
    return iou

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


def save_checkpoint(model, optimizer, epoch, val_loss, val_iou, save_dir, best=False):
    ckpt_name = f"best_model.pth" if best else f"checkpoint_epoch{epoch}.pth"
    path = os.path.join(save_dir, ckpt_name)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "val_iou": val_iou
    }, path)
    print(f"Saved checkpoint: {path}")


def log_metrics(log_path, epoch, train_loss, val_loss, val_iou, lr):
    header = ["epoch", "train_loss", "val_loss", "val_iou", "lr"]
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([epoch, train_loss, val_loss, val_iou, lr])


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
    val_iou = 0.0

    for images, masks in tqdm(dataloader, desc="Validation", leave=False):
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()

        batched_input = [
            {
                "image": img,
                "original_size": (m.shape[1], m.shape[2])  # same as training
            }
            for img, m in zip(images, masks)
        ]

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
        val_loss += loss.item()
        
        # Compute IoU
        val_i = batch_iou(torch.sigmoid(preds), masks).mean().item()
        val_iou += val_i

    return val_loss / len(dataloader), val_iou / len(dataloader)

# -----------------------------
# Main training loop
# -----------------------------
def train_model(
    variants=["adapter"],
    num_epochs=10,
    lr=1e-4,
    batch_size=2,
    scheduler_type="cosine",
    task_name="cod",
    save_root="checkpoints",
    device_str="mps"
):
    save_root = f"{save_root}/{'_'.join(variants)}/{task_name}"
    
    device = torch.device(device_str)
    MOBILESAM_CHECPOINT_PATH = "./models/mobileSam/weights/mobile_sam.pt"
        

    
    print(f"[INFO] {torch.cuda.is_available() = }")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Adapters: {', '.join(variants)}")
    print(f"[INFO] Schdeuler: {scheduler_type}")
    print(f"[INFO] Learning Rate: {lr}")
    print(f"[INFO] Batch size: {batch_size}")
    print(f"[INFO] Number of epochs: {num_epochs}\n")
    start_time = time.time()
    print(f"[INFO] Start Time: {start_time}\n")
    
    # Timestamped save dir
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(save_root, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, "training_log.csv")

    # Initialize model
    model = MobileSAMBase(pretrained_checkpoint=MOBILESAM_CHECPOINT_PATH)
    
    for variant in variants: 
        if variant == "base":
            break
        elif variant == "adapter":
            model.add_adapter()
            # model = MobileSAMAdapter()
        elif variant == "lora":
            model.add_lora(r=8, alpha=32)
            # model = MobileSAMLoRA(rank=8, alpha=32, pretrained_checkpoint=MOBILESAM_CHECPOINT_PATH)
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
    val_dataset = SegmentationDataset(task_name="cod", split="val", keep_original_size=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Loss function
    criterion = get_loss_function(task_name=task_name)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-2)
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    try:
        scaler = torch.amp.GradScaler()
    except AttributeError:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()    
    best_val_loss = torch.inf
    epochs_no_improve = 0
    patience = 30  # stop if val loss does not improve for 5 epochs

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val IoU: {val_iou:.4f}")
        
        # Log to CSV
        log_metrics(log_path, epoch+1, train_loss, val_loss, val_iou, lr_now)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch + 1, val_loss, val_iou, save_dir, best=True)
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered! Validation Loss has not improved for {patience} epochs.")
            break
        
    save_checkpoint(model, optimizer, epoch + 1, val_loss, val_iou, save_dir)

    print(f"\nTraining complete! Logs saved at: {log_path}")
    print(f"Best validation Loss: {best_val_loss:.4f}")
    end_time = time.time()

    print(f"[INFO] End Time: {end_time}\n")
    print(f"[INFO] Total Time: {(end_time - start_time)/(60*60):.1f}h")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', required=True, choices=["cuda", "mps", "cpu"])
    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config : dict= yaml.load(f, Loader=yaml.FullLoader)

    train_model(**config, device_str=args.device)
    
    
    # train_model(
    #     variant="lora",          # "base", "adapter", or "lora"
    #     num_epochs=10,
    #     lr=1e-3,
    #     scheduler_type="cosine",    # or "step"
    #     task_name="cod",            # "cod", "shadow", "structure"
    #     batch_size=8,
    #     save_root="checkpoints",
    # )