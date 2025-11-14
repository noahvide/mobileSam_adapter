import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import csv
import os

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import argparse
from PIL import Image

from models.mobileSam.mobile_sam import sam_model_registry
from models.mobile_sam_base import MobileSAMBase
from train import SegmentationDataset, batch_iou, get_loss_function





def load_adapter_checkpoint(model, checkpoint_path, map_location="cpu", strict=True):
    """
    Loads a fine-tuned MobileSAM model that includes adapter layers.

    Args:
        model (torch.nn.Module): An *instantiated* MobileSAMBase model 
                                 (e.g., MobileSAMBase(use_adapter=True)).
        checkpoint_path (str): Path to the checkpoint file (.pth / .pt).
        map_location (str): Device to map tensors to (default: 'cpu').
        strict (bool): Whether to strictly enforce matching keys.
    """
    print(f"[INFO] Loading fine-tuned MobileSAM adapter checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    # Handle DDP or Lightning wrapping
    if "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    # Remove "module." prefix if model was saved under DDP
    state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            k = k[len("module."):]
        state_dict[k] = v

    # Try loading
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print(f"[INFO] State dict loaded (strict={strict})")
    if missing_keys:
        raise Exception(f"The following keys are missing in checkpoint: {missing_keys}")
    if unexpected_keys:
        raise Exception(f"The following unexpected keys are present in checkpoint: {unexpected_keys}")

    print("[INFO] Adapter checkpoint successfully loaded.")

def compute_pad_info(orig_w, orig_h, target_size=1024):
    """Return scale, new_w, new_h, pad_left, pad_top used when padding orig->target_size."""
    scale = target_size / max(orig_w, orig_h)
    new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
    pad_left = (target_size - new_w) // 2
    pad_top  = (target_size - new_h) // 2
    return scale, new_w, new_h, pad_left, pad_top

def unpad_and_resize_mask_tensor(mask_tensor, orig_size, target_size=1024):
    """
    Convert a mask tensor in padded target_size coords -> numpy 2D in original image coords.
    mask_tensor: torch tensor with shape (H, W) or (1, H, W) or (1,1,H,W) in [0,1].
    orig_size: (orig_w, orig_h)
    Returns: numpy 2D mask in {0,1} shaped (orig_h, orig_w)
    """
    orig_w, orig_h = orig_size
    scale, new_w, new_h, pad_left, pad_top = compute_pad_info(orig_w, orig_h, target_size=target_size)

    # Convert to 2D numpy float [0,1]
    mask_np = mask_tensor.cpu().numpy()
    mask_np = np.squeeze(mask_np)
    mask_np = np.clip(mask_np, 0.0, 1.0)

    # Sanity: if mask not the expected target_size, try to resize it to target_size first
    if mask_np.shape != (target_size, target_size):
        # convert to PIL and resize to target_size (nearest)
        tmp = Image.fromarray((mask_np * 255).astype(np.uint8))
        tmp = tmp.resize((target_size, target_size), resample=Image.NEAREST)
        mask_np = np.array(tmp) / 255.0

    # Crop the padded region that corresponds to the resized original image
    crop = mask_np[pad_top: pad_top + new_h, pad_left: pad_left + new_w]
    # Resize crop back to original image size (orig_w, orig_h) using nearest to preserve binary masks
    crop_pil = Image.fromarray((crop * 255).astype(np.uint8))
    crop_resized_pil = crop_pil.resize((orig_w, orig_h), resample=Image.NEAREST)
    crop_resized = np.array(crop_resized_pil) / 255.0
    # Binarize to 0/1
    crop_resized = (crop_resized > 0.5).astype(np.uint8)
    return crop_resized




@torch.no_grad()
def validate(model, dataloader, dataset, device, desc):
    model.eval()
    ious = []
    records = []
    
    for idx, data_dict in enumerate(tqdm(dataloader, desc=desc, leave=True)):
        images = data_dict["image"]
        masks = data_dict["mask"]
        orig_sizes = data_dict["original_size"]
        use_prompt = "point_coords" in data_dict
        if use_prompt:
            point_coords = data_dict["point_coords"].to(device)
            point_labels = data_dict["point_labels"].to(device)
        
        images, masks = images.to(device), masks.to(device)
        orig_sizes = [o for o in zip(orig_sizes[0], orig_sizes[1])]
        masks = masks.float()  # ensure float
        
        orig_w, orig_h = orig_sizes[0]
        orig_w = int(orig_w)
        orig_h = int(orig_h)
        image_name = os.path.basename(dataset.img_paths[idx])

        if use_prompt:
            batched_input = [
                {
                    "image": img, 
                    "original_size": (int(orig_size[0]), int(orig_size[1])),
                    "point_coords": point_coord,
                    "point_labels": point_label            
                }
                for img, orig_size, point_coord, point_label in zip(images, orig_sizes, point_coords, point_labels)
            ]
        else:
            batched_input = [
                {
                    "image": img, 
                    "original_size": (int(orig_size[0]), int(orig_size[1]))            
                }
                for img, orig_size in zip(images, orig_sizes)
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
        
        # Compute IoU
        iou = batch_iou(torch.sigmoid(preds), masks).mean().item()
        ious.append(iou)
        records.append({
                "model_name": model_str,
                "task": task,
                "split": split,
                "image_name": image_name,
                "width": orig_w,
                "height": orig_h,
                "iou": iou
            })



    return ious, records

def main(task, split, model, model_str: str, save_csv=True, csv_path=None, is_legacy=False, use_prompt=False):
    print(model_str.lower() in ["mobilesam", "sam"] or not is_legacy)
    test_dataset = SegmentationDataset(task, split=split, target_size=1024, use_orig_normalization=True, use_prompt=use_prompt)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model.to(device)
    ious, records = validate(model, testloader, test_dataset, device, desc = f"Evaluating {model_str} on {task} ({split}-split) ({use_prompt=})")
    
    mean_iou = np.nanmean(ious)
    print(f"[RESULT] Mean IoU over {len(test_dataset)} images: {mean_iou:.4f}")

    # Save CSV
    if save_csv:
        if csv_path is None:
            csv_path = f"{model_str}_{task}{'_prompt_' if use_prompt else '_'}{split}_ious.csv"
        keys = ["model_name", "task", "split", "image_name", "width", "height", "iou"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec)
        print(f"[INFO] Per-image IoU saved to {csv_path}")

    return mean_iou

# def main1(task, split, model, model_str: str, save_csv=True, csv_path=None, is_legacy=False, use_prompt=False):
#     print(model_str.lower() in ["mobilesam", "sam"] or not is_legacy)
#     test_dataset = SegmentationDataset(task, split=split, target_size=1024, use_orig_normalization=model_str.lower() in ["mobilesam", "sam"] or not is_legacy, use_prompt=use_prompt)
#     testloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     device = torch.device("cuda" if torch.cuda.is_available() else "mps")
#     model.to(device)
#     model.eval()

#     ious = []
#     records = []

#     with torch.no_grad():
#         for idx, data_dict in enumerate(tqdm(testloader, desc=f"Evaluating {model_str} on {task} ({split}-split)")):
#             img = data_dict["image"]
#             gt_mask = data_dict["mask"]
#             orig_size = data_dict["original_size"]
            
#             use_prompt = "point_coords" in data_dict
#             if use_prompt:
#                 point_coords = data_dict["point_coords"].to(device)
#                 point_labels = data_dict["point_labels"].to(device)
                
            
#             img, gt_mask = img.to(device), gt_mask.to(device)

#             orig_w, orig_h = orig_size
#             orig_w = int(orig_w)
#             orig_h = int(orig_h)
#             image_name = os.path.basename(test_dataset.img_paths[idx])

#             batched_input = {"image": img[0], "original_size": (1024, 1024)}
            
#             if use_prompt:
#                 batched_input["point_coords"] = point_coords[0]
#                 batched_input["point_labels"] = point_labels[0]
                
#             outputs = model([batched_input], multimask_output=False)

#             pred_mask = outputs[0]["masks"][0]   # shape may be (H,W) or (1,H,W)
#             if pred_mask.ndim == 3:
#                 pred_mask = pred_mask[0]
#             # ensure float and padded-target_size
#             pred_mask_padded = F.interpolate(pred_mask.unsqueeze(0).unsqueeze(0).float(),
#                                             size=(1024, 1024), mode="nearest").squeeze()
#             # pred_masks_padded.append((image_name, pred_mask_padded))

#             # compute IoU on ORIGINAL image coordinates (more intuitive):
#             if gt_mask is not None:
#                 # convert both to original coords
#                 pred_orig = unpad_and_resize_mask_tensor(pred_mask_padded, (orig_w, orig_h), target_size=1024)
#                 gt_orig   = unpad_and_resize_mask_tensor(gt_mask, (orig_w, orig_h), target_size=1024)
#                 intersection = np.logical_and(pred_orig, gt_orig).sum()
#                 union = np.logical_or(pred_orig, gt_orig).sum()
#                 iou = float(intersection / union) if union > 0 else float("nan")
#                 ious.append(iou)

#             # Record for CSV
#             records.append({
#                 "model_name": model_str,
#                 "task": task,
#                 "split": split,
#                 "image_name": image_name,
#                 "width": orig_w,
#                 "height": orig_h,
#                 "iou": iou
#             })

#     mean_iou = np.nanmean(ious)
#     print(f"[RESULT] Mean IoU over {len(test_dataset)} images: {mean_iou:.4f}")

#     # Save CSV
#     if save_csv:
#         if csv_path is None:
#             csv_path = f"{model_str}_{task}{'_prompt_' if use_prompt else '_'}{split}_ious.csv"
#         keys = ["model_name", "task", "split", "image_name", "width", "height", "iou"]
#         with open(csv_path, "w", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=keys)
#             writer.writeheader()
#             for rec in records:
#                 writer.writerow(rec)
#         print(f"[INFO] Per-image IoU saved to {csv_path}")

#     return mean_iou



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True, choices=["cod", "shadow"])
    parser.add_argument('--split', required=True, choices=["train", "val", "test"])
    parser.add_argument('--chkp', required=True)
    parser.add_argument('--use_prompt', required=True)
    args = parser.parse_args()

    model_str = args.model
    task = args.task
    split = args.split
    checkpoint = args.chkp
    use_prompt = args.use_prompt.lower() in ["true", "1", "yes"]
        
    print(f"[INFO] Split: {split}")
    print(f"[INFO] Task: {task}") 
    print(f"[INFO] Use point prompt: {use_prompt}")
    
    if model_str.lower() not in ["sam", "mobilesam"]:
        model_name = checkpoint.split("/")[-1].split(".")[0]
        csv_path = "/".join(checkpoint.split("/")[:-1]) + f"/{model_str}_{model_name}{'_prompt_' if use_prompt else "_"}{task}_{split}_ious.csv"
    else:
        csv_path = None
    
    image_folder = f"./data/{task}/{split}/images"
    mask_folder = f"./data/{task}/{split}/masks"
    
    mobileSam_checkpoint = "./models/mobileSam/weights/mobile_sam.pt"
    
    if model_str == "mobileSam":
        model = sam_model_registry["vit_t"](checkpoint=mobileSam_checkpoint)
    elif model_str == "sam":
        raise NotImplementedError("SAM Baseline not implemented")
    else:
        model = MobileSAMBase(pretrained_checkpoint=mobileSam_checkpoint)
        if model_str == "adapter":
            model.add_adapter()
            load_adapter_checkpoint(model, checkpoint)
        elif model_str == "lora":
            model.add_lora(r=8, alpha=32)
            load_adapter_checkpoint(model, checkpoint)
        elif model_str == "adapter_lora":
            model.add_adapter()
            model.add_lora()
            load_adapter_checkpoint(model, checkpoint)

    
    main(task=task, split=split, model=model, model_str=model_str, save_csv=True, csv_path=csv_path, use_prompt=use_prompt)