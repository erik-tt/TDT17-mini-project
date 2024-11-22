import os
import torch
import monai
from monai.utils import set_determinism
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import Dataset, DataLoader, CacheDataset, SmartCacheDataset, ThreadDataLoader, DistributedSampler
from monai.inferers import sliding_window_inference
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.multiprocessing import spawn
import time
from tqdm import tqdm 
import gc
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
    EnsureTyped,
    EnsureChannelFirstd,
    ToTensord,
    Resized,
    AsDiscreted
)

def example(rank, world_size): #change name
    setup_ddp(rank, world_size)
    
    train(rank, world_size)
    
    cleanup()

    
def cleanup():
    dist.destroy_process_group()

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def load_data(data_path):
    data_preRT = []
    for patient_num in os.listdir(data_path):
        patient = f"{data_path}{patient_num}"
        image = f"{patient}/preRT/{patient_num}_preRT_T2.nii.gz"
        mask = f"{patient}/preRT/{patient_num}_preRT_mask.nii.gz"

        data_preRT.append({"image": image, "label": mask})
    return data_preRT

def train(rank, world_size):
    set_determinism(seed=42)
    
    train_transforms = Compose(
     [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 64)),
        AsDiscreted(keys=["label"], to_onehot=3),
        ToTensord(keys=["image", "label"])
    ])
    
    train_ds = Dataset(data=load_data("/cluster/projects/vc/data/mic/open/HNTS-MRG/train/"), 
                  transform=train_transforms)    
    
    train_sampler = DistributedSampler(dataset=train_ds, shuffle=True, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_ds, batch_size=1, sampler=train_sampler)
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=3,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(rank)
    
    model = DDP(model, device_ids=[rank])
    
    loss_function = DiceLoss(softmax=True)
    optimizer = torch.optim.Adam(model.parameters())
    dice_metric = DiceMetric(include_background=True, reduction="mean")
    max_epochs = 20

    for epoch in range(max_epochs):
            torch.cuda.empty_cache()
            gc.collect()
            epoch_start = time.time()
            epoch_loss = []
            correct = 0
            total = 0
            print("-" * 10)
            print(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            train_sampler.set_epoch(epoch)
            training_losses = []
            try:
                for batch_data in tqdm(train_loader):
                    images, labels = batch_data["image"].to(rank), batch_data["label"].to(rank)
                    optimizer.zero_grad()
                    outputs = model(images)
                    #l = np.argmax(outputs[0], axis=0)
                    #print(np.unique(l))
                    loss = loss_function(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                    training_losses.append(loss.item())
                    total += 1
                print(f"Training mean loss: {np.mean(training_losses)}")
            except Exception as e:
                print(e)
def main():
    world_size = torch.cuda.device_count()
    spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)


    
if __name__=="__main__":
    
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    main()

