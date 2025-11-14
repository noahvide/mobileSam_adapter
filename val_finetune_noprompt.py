## Code from https://github.com/mazurowski-lab/finetune-SAM.git
#from segment_anything import SamPredictor, sam_model_registry
import argparse
from models.mobileSam.mobile_sam import SamPredictor, sam_model_registry
from models.mobileSam.mobile_sam.modeling import Sam as mobileSam
from models.mobileSam.mobile_sam.modeling import Sam
# from models.mobileSam.mobile_sam.utils.transforms import ResizeLongestSide
# from skimage.measure import label
# from models.sam_LoRa import LoRA_Sam

#Scientific computing 
# import numpy as np
import os
#Pytorch packages
import torch
# from torch import nn
# import torch.optim as optim
import torchvision
# from torchvision import datasets
#Visulization
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
#Others
from torch.utils.data import DataLoader, Subset
# from torch.autograd import Variable
# import matplotlib.pyplot as plt
# import copy
from models.mobile_sam_base import MobileSAMBase
# from utils.dataset import Public_dataset
# import torch.nn.functional as F
# from torch.nn.functional import one_hot
from pathlib import Path
from tqdm import tqdm
# from utils.losses import DiceLoss
from utils.dsc import dice_coeff
# import cv2
# import monai
# from utils.utils import vis_image
# import cfg
# from argparse import Namespace
# import json
from train import SegmentationDataset


def main(task : str, split : str, model : Sam):
    # change to 'combine_all' if you want to combine all targets into 1 cls
    # test_dataset = Public_dataset(args,args.img_folder, args.mask_folder, test_img_list,phase='val',targets=[args.targets],if_prompt=False)
    test_dataset = SegmentationDataset(task, split=split)
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # if args.finetune_type == 'adapter' or args.finetune_type == 'vanilla':
    #     sam_fine_tune = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.dir_checkpoint,'checkpoint_best.pth'),num_classes=args.num_cls)
    # elif args.finetune_type == 'lora':
    #     sam = sam_model_registry[args.arch](args,checkpoint=os.path.join(args.sam_ckpt),num_classes=args.num_cls)
    #     sam_fine_tune = LoRA_Sam(args,sam,r=4).to('cuda').sam
    #     sam_fine_tune.load_state_dict(torch.load(args.dir_checkpoint + '/checkpoint_best.pth'), strict = False)
        
    # sam_fine_tune = sam_fine_tune.to('cuda').eval()
    model.to(device).eval()
    class_iou = torch.zeros(1,dtype=torch.float)
    cls_dsc = torch.zeros(1,dtype=torch.float)
    eps = 1e-9
    img_name_list = []
    pred_msk = []
    test_img = []
    test_gt = []

    for i,data in enumerate(tqdm(testloader)):
        imgs, msks = data
        
        imgs = imgs.to(device)
        msks = torchvision.transforms.Resize((1024,1024))(msks)
        msks = msks.to(device)
        # img_name_list.append(data['img_name'][0])

        with torch.no_grad():
            img_emb= model.image_encoder(imgs)

            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred_fine, _ = model.mask_decoder(
                            image_embeddings=img_emb,
                            image_pe=model.prompt_encoder.get_dense_pe(), 
                            sparse_prompt_embeddings=sparse_emb,
                            dense_prompt_embeddings=dense_emb, 
                            multimask_output=True,
                          )
           
        pred_fine = pred_fine.argmax(dim=1)

        
        pred_msk.append(pred_fine.cpu())
        test_img.append(imgs.cpu())
        test_gt.append(msks.cpu())
        yhat = (pred_fine).cpu().long().flatten()
        y = msks.cpu().flatten()

        for j in range(1):
            y_bi = y==j
            yhat_bi = yhat==j
            I = ((y_bi*yhat_bi).sum()).item()
            U = (torch.logical_or(y_bi,yhat_bi).sum()).item()
            class_iou[j] += I/(U+eps)

        for cls in range(1):
            mask_pred_cls = ((pred_fine).cpu()==cls).float()
            mask_gt_cls = (msks.cpu()==cls).float()
            cls_dsc[cls] += dice_coeff(mask_pred_cls,mask_gt_cls).item()
        #print(i)

    class_iou /=(i+1)
    cls_dsc /=(i+1)

    save_folder = os.path.join('test_results')
    Path(save_folder).mkdir(parents=True,exist_ok = True)
    #np.save(os.path.join(save_folder,'test_masks.npy'),np.concatenate(pred_msk,axis=0))
    #np.save(os.path.join(save_folder,'test_name.npy'),np.concatenate(np.expand_dims(img_name_list,0),axis=0))


    print('class dsc:',cls_dsc)      
    print('class iou:',class_iou)



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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--task', required=True, choices=["cod", "shadow"])
    parser.add_argument('--split', required=True, choices=["train", "val", "test"])
    parser.add_argument('--chkp', required=True)
    args = parser.parse_args()

    model_str = args.model
    task = args.task
    split = args.split
    checkpoint = args.chkp
    
    image_folder = f"./data/{task}/{split}/images"
    mask_folder = f"./data/{task}/{split}/masks"
    
    mobileSam_checkpoint = "./models/mobileSam/weights/mobile_sam.pt"
    
    if model_str == "mobileSam":
        model = sam_model_registry["vit_t"](checkpoint=mobileSam_checkpoint)
    elif model_str == "sam":
        pass
    else:
        model = MobileSAMBase(pretrained_checkpoint=mobileSam_checkpoint)
        if model_str == "adapter":
            model.add_adapter()
            load_adapter_checkpoint(model, checkpoint)
        elif model_str == "lora":
            model.add_lora()
            load_adapter_checkpoint(model, checkpoint)
        elif model_str == "adapter_lora":
            model.add_adapter()
            model.add_lora()
            load_adapter_checkpoint(model, checkpoint)

    
    main(task=task, split=split, model=model)
    

    # if 1: # if you want to load args from taining setting or you want to identify your own setting
    #     args_path = f"{args.dir_checkpoint}/args.json"

    #     # Reading the args from the json file
    #     with open(args_path, 'r') as f:
    #         args_dict = json.load(f)
        
    #     # Converting dictionary to Namespace
    #     args = Namespace(**args_dict)
        
    # dataset_name = args.dataset_name
    # print('train dataset: {}'.format(dataset_name)) 
    # test_img_list =  args.img_folder + '/train_slices_info_sampled_1000.txt'
    # main(args,test_img_list)