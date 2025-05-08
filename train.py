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

device = torch.device("cpu")
batch_size = 1

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

preprocess = T.Compose([T.ConvertImageDtype(torch.float32),
                        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                    std=(0.26862954, 0.26130258, 0.27577711))])

resize = T.Compose([T.Resize(size=resize_img, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
                    T.CenterCrop(size=(resize_img, resize_img))])

norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                   std=(0.26862954, 0.26130258, 0.27577711))

inv_norm = T.Compose([T.Normalize(mean = [ 0., 0., 0. ],
                                  std = [ 1/0.26862954, 1/0.26130258, 1/0.27577711]),
                                  
                      T.Normalize(mean = [-0.48145466, -0.4578275, -0.40821073],
                                  std = [ 1., 1., 1. ])])

raft_model = raft(weights=Raft_Large_Weights.DEFAULT, progress = True).to(device).eval()
image_feature_extraction = CLIP(embed_dim=embed_dim,
                                image_resolution=resize_img,
                                vision_layers=12,
                                vision_width=768,
                                vision_patch_size=patch_size,
                                context_length=77,
                                vocab_size=49408,
                                transformer_width=1024,
                                transformer_heads=8,
                                transformer_layers=12).to(device).train()

horizontal_flow_reconstruction = Reconstruction(input_size = hidden_size, 
                                                output_channels=1, 
                                                num_predictions = num_predictions,
                                                device = device).train()

vertical_flow_reconstruction = Reconstruction(input_size = hidden_size, 
                                              output_channels=1, 
                                              num_predictions = num_predictions,
                                              device = device).train()

image_reconstruction = Reconstruction(input_size = hidden_size,
                                      output_channels=3, 
                                      num_predictions = num_predictions,
                                      device = device).train()

state_prediction = StatePrediction(input_size = hidden_size, 
                                   num_predictions = num_predictions,
                                   device = device).train()

rnn_cell = nn.LSTMCell(input_size = embed_dim, 
                       hidden_size = hidden_size).to(device).train()

object_detection = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

hidden_state_prediction = nn.Sequential(nn.Linear(512,512),
                                        nn.GELU(),
                                        nn.Linear(512,512)).to(device)

cell_state_prediction = nn.Sequential(nn.Linear(512,512),
                                        nn.GELU(),
                                        nn.Linear(512,512)).to(device)

parameters = (list(image_feature_extraction.parameters()) +
              list(image_reconstruction.parameters()) +
              list(horizontal_flow_reconstruction.parameters()) +
              list(vertical_flow_reconstruction.parameters()) +
              list(rnn_cell.parameters()) +
              list(hidden_state_prediction.parameters()) +
              list(cell_state_prediction.parameters()) + 
              list(state_prediction.parameters()))


optimizer = torch.optim.Adam(parameters, lr = 0.001)

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

hidden_state = torch.rand(1, hidden_size, dtype = torch.float32, requires_grad=True)
cell_state = torch.rand(1, hidden_size, dtype = torch.float32, requires_grad=True)

video_train_subfolders = os.listdir("video_train")
video_train_subfolders = [folder for folder in video_train_subfolders if folder.startswith("video")]
video_train_subfolders.sort(key=lambda x: int(x.split('_')[1].split('-')[0]))

for subfolder in video_train_subfolders: 
    print(f"{subfolder} is being processed")
    video_files = [f for f in os.listdir(f'video_train/{subfolder}') if f.endswith('.mp4')]
    
    for file in video_files: 
        video_path = f'video_train/{subfolder}/{file}'
        frames, _, _ = read_video(str(video_path), pts_unit='sec')
        frames = frames.permute(0, 3, 1, 2).unsqueeze(0)
        frames = frames[:, ::stride]
        i = 0    
        total_loss = 0.0
        total_img_loss = 0.0 
        total_flow_loss = 0.0
            
        while (i <= num_frames):    
            ########### Extracting Actual Data ###########
            actual_imgs_inv_norm = frames[:, i: i + num_predictions + 1] 
            actual_imgs_norm = preprocess(actual_imgs_inv_norm).to(device)       
            
            actual_states = []
            actual_states_flat = []
            actual_flows = []    
            actual_flows_rgb = []        
                    
            for j in range(num_predictions + 1):
                num_objects, state = return_state(actual_imgs_inv_norm[0, j], object_detection, device)
                state_flat = state.flatten()[None]
                actual_states_flat.append(state_flat) 
                actual_states.append(state[:num_objects])  
                if (j <= num_predictions-1):
                    flow = raft_model(actual_imgs_norm[:, j], actual_imgs_norm[:, j+1])[-1]
                    flow_rgb = flow_to_image(flow)                
                    actual_flows.append(flow)
                    actual_flows_rgb.append(flow_rgb)    
                
            actual_states_flat = torch.stack(actual_states_flat, dim = 1)
            actual_flows = torch.stack(actual_flows, dim = 1)          
            actual_flows_rgb = torch.stack(actual_flows_rgb, dim = 1)  
            ########### Extracting Actual Data ###########
            
            ############ Extracting Current Image Features ###########
            current_img_inv_norm = actual_imgs_inv_norm[:, 0]
            current_img_norm = actual_imgs_norm[:, 0]
            current_img_resized = resize(current_img_norm)
            current_img_processed = preprocess(current_img_resized)                             
            current_img_features = image_feature_extraction.encode_image(current_img_processed) 
            ############ Extracting Current Image Features ###########
            
            ############ Predicting the Next Frames, Optical Flows, and States  ###########
            predicted_next_imgs_norm = []
            predicted_next_vertical_flows = [] 
            predicted_next_horizontal_flows = []
            predicted_next_states = []
            current_state = actual_states_flat[:,0]
                    
            for k in range(num_predictions): 
                hidden_state = hidden_state_prediction(current_img_features)
                cell_state = cell_state_prediction(current_img_features)
                
                hidden_state, cell_state = rnn_cell(current_img_features, (hidden_state, cell_state))
                
                predicted_img_norm = image_reconstruction(hidden_state) 
                predicted_vertical_flow =  vertical_flow_reconstruction(hidden_state) 
                predicted_horizontal_flow = horizontal_flow_reconstruction(hidden_state) 
                predicted_state = state_prediction(hidden_state)
                        
                predicted_next_imgs_norm.append(predicted_img_norm)
                predicted_next_vertical_flows.append(predicted_vertical_flow)
                predicted_next_horizontal_flows.append(predicted_horizontal_flow)
                predicted_next_states.append(predicted_state)
                
                predicted_img_resized = resize(predicted_img_norm)
                predicted_img_processed = preprocess(predicted_img_resized)
                current_img_features = image_feature_extraction.encode_image(predicted_img_processed)
                
            predicted_next_imgs_norm = torch.stack(predicted_next_imgs_norm,dim=1)
            predicted_next_vertical_flows = torch.stack(predicted_next_vertical_flows,dim=1)
            predicted_next_horizontal_flows = torch.stack(predicted_next_horizontal_flows,dim=1)
            predicted_next_states = torch.stack(predicted_next_states,dim=1)
            predicted_next_imgs_inv_norm = inv_norm(predicted_next_imgs_norm)
            predicted_next_flows = torch.cat((predicted_next_horizontal_flows, predicted_next_vertical_flows), axis = 2)
            
            predicted_next_flows_rgb = []
            for batch in range(batch_size):
                predicted_next_flows_rgb.append(flow_to_image(predicted_next_flows[batch]))
            predicted_next_flows_rgb = torch.stack(predicted_next_flows_rgb, dim = 0)                                    
            ############ Predicting the Next Frames, Optical Flows, and States  ###########           
                            
            ############ Visualizing Predictions of Next Frames and Optical Flows ###########
            visualize_comparisons(predicted_next_flows_rgb, actual_flows_rgb, subfolder, file, "flows", i)        
            visualize_comparisons(predicted_next_imgs_inv_norm, actual_imgs_inv_norm[:, 1:], subfolder, file, "frames", i)
            ############ Visualizing Predictions of Next Frames and Optical Flows ###########
                    
            ############ Loss Computation ###########
            img_loss   = nn.functional.mse_loss(predicted_next_imgs_norm, actual_imgs_norm[:,1:])
            flow_loss  = nn.functional.mse_loss(predicted_next_flows,     actual_flows) 
            state_loss = nn.functional.mse_loss(predicted_next_states,    actual_states_flat[:, 1:])
            motion_loss = motion_error(actual_imgs_norm, predicted_next_imgs_norm, actual_states, num_predictions)
            total_loss = img_loss + flow_loss + torch.log(state_loss) + motion_loss
            ############ Loss Computation ###########
        
            ############ Print Loss ###########
            print(f"Flow Loss: {flow_loss}")
            print(f"Image Loss: {img_loss}")
            print(f"Motion Loss: {motion_loss}")
            print(f"State Loss: {state_loss}")
            print()
            ############ Print Loss ###########
                         
            ############ Optimization ###########
            optimizer.zero_grad()            
            total_loss.backward()
            optimizer.step()
            ############ Optimization ###########
            
            if (i % 20 == -1): 
                torch.save(image_feature_extraction.state_dict(), os.path.join(models_folder, "image_feature_extraction.pth"))
                torch.save(horizontal_flow_reconstruction.state_dict(), os.path.join(models_folder, "horizontal_flow_reconstruction.pth"))
                torch.save(vertical_flow_reconstruction.state_dict(), os.path.join(models_folder, "vertical_flow_reconstruction.pth"))
                torch.save(image_reconstruction.state_dict(), os.path.join(models_folder, "image_reconstruction.pth"))
                torch.save(state_prediction.state_dict(), os.path.join(models_folder, "state_prediction.pth"))
                torch.save(rnn_cell.state_dict(), os.path.join(models_folder, "rnn_cell.pth"))
                
            ############ Preparing for the Next Step ############
            i += 1   
            ############ Preparing for the Next Step ############                