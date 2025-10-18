import os 
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.io import read_video
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

batch_size = 4 
stride = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class VideoDataset(Dataset): 
    def __init__(self, video_path, stride): 
        self.stride = stride

        video_subfolders = os.listdir(video_path)
        video_subfolders = [
            folder for folder in video_subfolders if folder.startswith("video")
        ]
        
        video_subfolders.sort(
            key=lambda x: int(x.split('_')[1].split('-')[0])
        )

        video_paths = []
        for subfolder in video_subfolders:
            video_list = os.listdir(f'{video_path}/{subfolder}')
            video_list.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
            video_paths.extend([video_path + '/' + subfolder + '/' + f for f in video_list if f.endswith('.mp4')])
              
        self.video_paths = video_paths


    def __len__(self): 
        return len(self.video_paths)
    
    def __getitem__(self, index): 
        video_path = self.video_paths[index]
        video_id = video_path.split('.')[0].split('_')[-1]

        frames, _, _ = read_video(str(video_path), pts_unit='sec')
        frames = frames.permute(0, 3, 1, 2) 
        
        # Attention: If you apply stride, you must apply it to motion trajectories, collisions, etc. as well
        frames = frames[::self.stride]

        return frames, video_id
    

# Preprocessing applied once at module scope
preprocess = T.Compose([
    T.ConvertImageDtype(torch.float32),
    T.Resize(size=(160, 240), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
    T.Normalize(mean=0.5, std=0.5),
])

def compute_optical_flows(frames, raft_model):
    # frames: [B,F,C,H,W] in [-1,1], dtype float32
    B, F, C, H, W = frames.shape
    flows_bt = []
    with torch.no_grad():
        for j in range(F - 1):
            f1 = frames[:, j]     # [B,C,H,W]
            f2 = frames[:, j+1]   # [B,C,H,W]
            list_of_flows = raft_model(f1, f2)   
            flow = list_of_flows[-1].cpu() # last entry: [B,2,H,W]
            flows_bt.append(flow) # F - 1 x [B, 2, H, W]
    return torch.stack(flows_bt, dim=1)  # [B, F-1, 2, H, W]

def save_flows_npz_single(flow_tensor, video_id, output_dir):
    video_num = int(video_id)
    start = (video_num // 1000) * 1000
    end = start + 1000
    subfolder = f"flow_{start:05d}-{end:05d}"

    output_path = os.path.join(output_dir, subfolder)
    os.makedirs(output_path, exist_ok=True)

    # flow_tensor: [F-1, 2, H, W] on CPU
    flow_np = flow_tensor.numpy()
    filename = os.path.join(output_path, f"flow_{video_id}.npz")
    # Save compressed numpy archive
    np.savez_compressed(filename, flow=flow_np)
    return filename

def main():
    print("Loading RAFT model...")
    weights = Raft_Large_Weights.DEFAULT
    raft_model = raft_large(weights=weights, progress=True).to(device).eval()
    print("RAFT model loaded successfully!")

    os.makedirs("flows_train", exist_ok=True)
    os.makedirs("flows_validation", exist_ok=True)

    train_data = VideoDataset(video_path="video_train", stride=stride)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=0)

    for frames, video_id in tqdm(train_loader, desc="Training videos"):
        # frames = 32 x 128 x 3 x 224 x 224 [B,F,C,H,W]
        B, F, C, H, W = frames.shape
        ft = frames.reshape(B*F, C, H, W)
        ft = preprocess(ft)  
        ft = ft.reshape(B, F, C, ft.shape[-2], ft.shape[-1])
        ft = ft.to(device)
        
        flows = compute_optical_flows(ft, raft_model)
        # flows: [B, F-1, 2, H, W] on CPU
        for b in range(B):
            save_flows_npz_single(flows[b], video_id[b], "flows_train")

    validation_data = VideoDataset(video_path="video_validation", stride=stride)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False, num_workers=0)

    for frames, video_id in tqdm(validation_loader, desc="Validation videos"):
        B, F, C, H, W = frames.shape
        ft = frames.reshape(B*F, C, H, W)
        ft = preprocess(ft)  
        ft = ft.reshape(B, F, C, ft.shape[-2], ft.shape[-1])
        ft = ft.to(device)
        
        flows = compute_optical_flows(ft, raft_model)
        for b in range(B):
            save_flows_npz_single(flows[b], video_id[b], "flows_validation")

    print("\n Optical flow pre-computation completed!")  

if __name__ == "__main__": 
    main()