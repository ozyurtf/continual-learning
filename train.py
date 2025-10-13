import argparse
import torch
import torchvision
from torchvision.models.optical_flow import raft_large as raft
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import flow_to_image
from torchvision.io import read_video
import torchvision.transforms as T
from torchvision.models import vgg16
from tqdm import tqdm
import os
import shutil
from clip.model import CLIP
from models import *
from utils import *
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

parser = argparse.ArgumentParser(description="Model parameters")
parser.add_argument('--num_predictions', type=int, default=3, help='Number of predictions to make in each step')
parser.add_argument('--embed_dim', type=int, default=512, help='Embedding dimension for the model')
parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size for the hidden state')
parser.add_argument('--stride', type=int, default=1, help='Stride for video frame sampling')
parser.add_argument('--num_frames', type=int, default=127, help='Total number of frames used for training for each video')
parser.add_argument('--resize_img', type=int, default=224, help='Size to resize images for processing')
parser.add_argument('--patch_size', type=int, default=32, help='Patch size for CLIP image encoder')
args = parser.parse_args()

num_predictions = args.num_predictions
embed_dim = args.embed_dim
hidden_size = args.hidden_size
stride = args.stride
num_frames = args.num_frames
resize_img = args.resize_img
patch_size = args.patch_size

class VideoDataset(Dataset): 
    def __init__(self, resize_img, patch_size, video_path, annotation_path,stride): 
        video_subfolders = os.listdir(video_path)
        annotation_subfolders = os.listdir(annotation_path)

        video_subfolders = [
            folder for folder in video_subfolders if folder.startswith("video")
        ]

        annotation_subfolders = [
            folder for folder in annotation_subfolders if folder.startswith("annotation")
        ]
        
        video_subfolders.sort(
            key=lambda x: int(x.split('_')[1].split('-')[0])
        )

        annotation_subfolders.sort(
            key=lambda x: int(x.split('_')[1].split('-')[0])
        )
        
        video_paths = []
        for subfolder in video_subfolders:
            video_paths.extend([video_path + '/' + subfolder + '/' + f for f in os.listdir(f'{video_path}/{subfolder}') if f.endswith('.mp4')])
        
        annotation_paths = []
        for subfolder in annotation_subfolders:
            annotation_paths.extend([annotation_path + '/' + subfolder + '/' + f for f in os.listdir(f'{annotation_path}/{subfolder}') if f.endswith('.json')])
        
        self.video_paths = video_paths
        self.annotation_paths = annotation_paths

        self.preprocess = T.Compose(
            [
                T.ConvertImageDtype(torch.float32),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073), 
                    std=(0.26862954, 0.26130258, 0.27577711))
            ]
        )

        self.resize = T.Compose([
            T.Resize(size=resize_img, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(resize_img, resize_img))]
        )

        self.norm = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073), 
            std=(0.26862954, 0.26130258, 0.27577711)
        )

        self.inv_norm = T.Compose(
            [
                T.Normalize(mean = [ 0., 0., 0. ],
                std = [1/0.26862954, 1/0.26130258, 1/0.27577711]),
                T.Normalize(
                    mean = [-0.48145466, -0.4578275, -0.40821073],
                    std = [ 1., 1., 1. ]
                )
            ]
        )        

        self.stride = stride

    def __len__(self): 
        return len(self.video_paths)
    
    def __getitem__(self, index): 
        video_path = self.video_paths[index]
        annotation_path = self.annotation_paths[index]

        frames, _, _ = read_video(str(video_path), pts_unit='sec')
        frames = frames.permute(0, 3, 1, 2)
        frames = frames[::self.stride]
        frames_norm = self.preprocess(frames)
        frames_resized = self.resize(frames_norm)
        frames_processed = self.preprocess(frames_resized)        

        with open(annotation_path, 'r') as f:
            annotations = json.load(f)

        object_properties = annotations['object_property']
        motion_trajectories = annotations['motion_trajectory']
        collisions = annotations['collision']
        return frames, frames_processed,object_properties, motion_trajectories, collisions

def custom_collate(batch):
    frames = torch.stack([item[0] for item in batch])
    frames_processed = torch.stack([item[1] for item in batch])
    object_properties = [item[2] for item in batch]  
    motion_trajectories = [item[3] for item in batch]
    collisions = [item[4] for item in batch] 
    return frames, frames_processed, object_properties, motion_trajectories, collisions 

raft_model = raft(weights=Raft_Large_Weights.DEFAULT, progress = True).to(device).eval()

image_feature_extraction = CLIP(
    embed_dim=embed_dim,
    image_resolution=resize_img,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=patch_size,
    context_length=77,
    vocab_size=49408,
    transformer_width=1024,
    transformer_heads=8,
    transformer_layers=12).to(device).train()

horizontal_flow_reconstruction = Reconstruction(
    input_size = hidden_size, 
    output_channels=1, 
    num_predictions = num_predictions,
    device = device).train()

vertical_flow_reconstruction = Reconstruction(
    input_size = hidden_size, 
    output_channels=1, 
    num_predictions = num_predictions,
    device = device).train()

image_reconstruction = Reconstruction(
    input_size = hidden_size,
    output_channels=3, 
    num_predictions = num_predictions,
    device = device).train()

state_prediction = StatePrediction(
    input_size = hidden_size, 
    num_predictions = num_predictions,
    device = device).train()

rnn_cell = nn.LSTMCell(
    input_size = embed_dim, 
    hidden_size = hidden_size).to(device).train()

faster_rcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

optimizer_image = torch.optim.Adam(image_reconstruction.parameters(), lr=0.001)
optimizer_flow_h = torch.optim.Adam(horizontal_flow_reconstruction.parameters(), lr=0.001)
optimizer_flow_v = torch.optim.Adam(vertical_flow_reconstruction.parameters(), lr=0.001)
optimizer_state = torch.optim.Adam(state_prediction.parameters(), lr=0.001)

for item in os.listdir('frames'):
    item_path = os.path.join('frames', item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path) 

for item in os.listdir('flows'):
    item_path = os.path.join('flows', item)
    if os.path.isdir(item_path):
        shutil.rmtree(item_path)       
        
models_folder = "models"
if not os.path.exists(models_folder):
    os.makedirs(models_folder)          
                
num_frames = min(num_frames, 127 - num_predictions)

train_data = VideoDataset(resize_img = 224, patch_size = 32, video_path = "video_train", annotation_path = "annotation_train", stride = 1)
train_loader = DataLoader(train_data, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate)

validation_data = VideoDataset(resize_img = 224, patch_size = 32, video_path = "video_validation", annotation_path = "annotation_validation", stride = 1)
validation_loader = DataLoader(validation_data, batch_size=2, shuffle=True, num_workers=0, collate_fn=custom_collate)       

for frames, frames_processed, object_properties, motion_trajectories, collisions in train_loader:     
    # frames = torch.Size([batch_size, 128, 3, 320, 480])
    accumulated_img_loss = 0.0
    accumulated_flow_h_loss = 0.0
    accumulated_flow_v_loss = 0.0
    accumulated_state_loss = 0.0
    
    # Initialize hidden and cell state before the sequence
    hidden_state = torch.zeros(1, hidden_size, device=device)
    cell_state = torch.zeros(1, hidden_size, device=device)

    i = 0 
    while (i <= num_frames):    
        ########### Extracting Actual Data ###########
        actual_imgs_norm = frames_norm[:, i: i + num_predictions + 1]    
        
        actual_states = []
        actual_states_flat = []
        actual_flows = []    
        actual_flows_rgb = []        
                
        for j in range(num_predictions + 1):
            num_objects, state = return_state(frames[:, j], faster_rcnn, device)
            batch_size = frames.shape[0]
            state_flat = state.view(batch_size, -1) # 1 x 128
            actual_states_flat.append(state_flat) 

            actual_states.append(state[:num_objects]) # 2 x 128

            if (j <= num_predictions-1):
                flow = raft_model(frames_norm[:, j], frames_norm[:, j+1])[-1]
                flow_rgb = flow_to_image(flow)    
                            
                actual_flows.append(flow)
                actual_flows_rgb.append(flow_rgb)    
            
        actual_states_flat = torch.stack(actual_states_flat, dim = 1)
        actual_flows = torch.stack(actual_flows, dim = 1)          
        actual_flows_rgb = torch.stack(actual_flows_rgb, dim = 1)  
        ########### Extracting Actual Data ###########
        
        ############ Extracting Current Image Features ###########
        current_img = frames[:, 0]
        current_img_norm = frames_norm[:, 0]
        current_img_resized = frames_resized[:, 0] # resize(current_img_norm)
        current_img_processed = frames_processed[:, 0] # preprocess(current_img_resized)                             
        current_img_features = image_feature_extraction.encode_image(current_img_processed) 
        ############ Extracting Current Image Features ###########
        
        ############ Predicting the Next Frames, Optical Flows, and States  ###########
        predicted_next_imgs_norm = []
        predicted_next_vertical_flows = [] 
        predicted_next_horizontal_flows = []
        predicted_next_states = []
        current_state = actual_states_flat[:,0]
        
        for k in range(num_predictions): 
            
            hidden_state, cell_state = rnn_cell(current_img_features, (hidden_state, cell_state))
            
            predicted_img_norm = image_reconstruction(hidden_state) 
            predicted_vertical_flow =  vertical_flow_reconstruction(hidden_state) 
            predicted_horizontal_flow = horizontal_flow_reconstruction(hidden_state) 
            predicted_state = state_prediction(hidden_state)
                    
            predicted_next_imgs_norm.append(predicted_img_norm)
            predicted_next_vertical_flows.append(predicted_vertical_flow)
            predicted_next_horizontal_flows.append(predicted_horizontal_flow)
            predicted_next_states.append(predicted_state)
            
            predicted_img_resized = frames_resized[:, k+1] # resize(predicted_img_norm)
            predicted_img_processed = frames_processed[:, k+1] # preprocess(predicted_img_resized)
            current_img_features = image_feature_extraction.encode_image(predicted_img_processed)
            
        predicted_next_imgs_norm = torch.stack(predicted_next_imgs_norm,dim=1)
        predicted_next_vertical_flows = torch.stack(predicted_next_vertical_flows,dim=1)
        predicted_next_horizontal_flows = torch.stack(predicted_next_horizontal_flows,dim=1)
        predicted_next_states = torch.stack(predicted_next_states,dim=1)
        predicted_next_imgs_inv_norm = frames_norm[:, 1:num_predictions+1] # inv_norm(predicted_next_imgs_norm)
        predicted_next_flows = torch.cat((predicted_next_horizontal_flows, predicted_next_vertical_flows), axis = 2)
        
        predicted_next_flows_rgb = []
        for batch in range(batch_size):
            predicted_next_flows_rgb.append(
                flow_to_image(
                    predicted_next_flows[batch]
                )
            )

        predicted_next_flows_rgb = torch.stack(predicted_next_flows_rgb, dim = 0)                                    
        ############ Predicting the Next Frames, Optical Flows, and States  ###########           
                        
        ############ Visualizing Predictions of Next Frames and Optical Flows ###########
        visualize_comparisons(predicted_next_flows_rgb, actual_flows_rgb, subfolder, file, "flows", i)        
        visualize_comparisons(predicted_next_imgs_inv_norm, frames_norm[:, 1:num_predictions+1], subfolder, file, "frames", i)
        ############ Visualizing Predictions of Next Frames and Optical Flows ###########
                
        ############ Loss Computation ###########
        img_loss   = nn.functional.mse_loss(predicted_next_imgs_norm[:, -1:], frames_norm[:, -1:])
        flow_h_loss  = nn.functional.mse_loss(predicted_next_horizontal_flows[:, -1:], actual_flows[:, -1:, 0:1])
        flow_v_loss  = nn.functional.mse_loss(predicted_next_vertical_flows[:, -1:], actual_flows[:, -1:, 1:2])
        state_loss = nn.functional.mse_loss(predicted_next_states[:, -1:],    actual_states_flat[:, -1:])
        # motion_loss is not used for optimizer, but you can still print it if desired
        motion_loss = motion_error(frames_norm, predicted_next_imgs_norm, actual_states, num_predictions)
        ############ Loss Computation ###########
    
        ############ Print Loss ###########
        print(f"Image Loss: {img_loss}")
        print(f"Flow H Loss: {flow_h_loss}")
        print(f"Flow V Loss: {flow_v_loss}")
        print(f"State Loss: {state_loss}")
        print(f"Motion Loss: {motion_loss}")
        print()
        ############ Print Loss ###########
        
        # Accumulate each loss
        accumulated_img_loss += img_loss
        accumulated_flow_h_loss += flow_h_loss
        accumulated_flow_v_loss += flow_v_loss
        accumulated_state_loss += state_loss
        
        # Gradient accumulation step
        if ((i % 8 == 0 and i > 0) or (i == num_frames)):
            # Image model
            optimizer_image.zero_grad()
            accumulated_img_loss.backward(retain_graph=True)
            optimizer_image.step()
            accumulated_img_loss = 0.0

            # Horizontal flow model
            optimizer_flow_h.zero_grad()
            accumulated_flow_h_loss.backward(retain_graph=True)
            optimizer_flow_h.step()
            accumulated_flow_h_loss = 0.0

            # Vertical flow model
            optimizer_flow_v.zero_grad()
            accumulated_flow_v_loss.backward(retain_graph=True)
            optimizer_flow_v.step()
            accumulated_flow_v_loss = 0.0

            # State model (last backward, no retain_graph)
            optimizer_state.zero_grad()
            accumulated_state_loss.backward()
            optimizer_state.step()
            accumulated_state_loss = 0.0

            # Reset hidden and cell state
            hidden_state = torch.zeros(1, hidden_size, device=device)
            cell_state = torch.zeros(1, hidden_size, device=device)
        
        ############ Preparing for the Next Step ############
        i += 1   
        ############ Preparing for the Next Step ############                