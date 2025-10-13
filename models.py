from torch import nn
import torch

class Reconstruction(nn.Module):
    def __init__(self, input_size, num_predictions, device, output_channels=3, output_height=320, output_width=480):
        super(Reconstruction, self).__init__()
        intermediate_channels = 64
        intermediate_height = output_height // 4
        intermediate_width = output_width // 4
        intermediate_dim = intermediate_channels * intermediate_height * intermediate_width

        self.reconstruction = nn.Sequential(nn.Linear(input_size, intermediate_dim),
                                            Reshape(-1, intermediate_channels, intermediate_height, intermediate_width),                   
                                            nn.ConvTranspose2d(intermediate_channels, 32,
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1,
                                                               output_padding=1),
                                            nn.ReLU(),
                                            nn.ConvTranspose2d(32, output_channels,
                                                               kernel_size=3,
                                                               stride=2,
                                                               padding=1,
                                                               output_padding=1)).to(device)

    def forward(self, x):
        predictions = self.reconstruction(x)
        return predictions

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
    
class StatePrediction(nn.Module):
    def __init__(self, input_size, num_predictions, device):
        super(StatePrediction, self).__init__()
        self.state_prediction  = nn.Sequential(nn.Linear(input_size, 128), 
                                               nn.ReLU()).to(device)


    def forward(self, x):
        predictions = self.state_prediction(x)
        return predictions
    
    
def motion_error(actual_imgs_norm, predicted_next_imgs_norm, actual_states, num_predictions):
    # Only compare the last prediction with the last actual value
    m = num_predictions - 1  # Use the last prediction
    actual = actual_imgs_norm[0, m+1]
    predicted = predicted_next_imgs_norm[0, m]
    state = actual_states[m+1]
    
    num_detected_objects = state.shape[0]
    
    motion_loss = 0.0
    
    for o in range(num_detected_objects): 
        width_start, width_len, height_start, height_len = state[o]
        
        window1 = actual[:, int(height_start): int(height_start + height_len), int(width_start): int(width_start + width_len)]
        window2 = predicted[:, int(height_start): int(height_start + height_len), int(width_start): int(width_start + width_len)]
        
        motion_loss += torch.nn.functional.mse_loss(window1, window2)
    
    # Average by number of objects only (not by num_predictions since we're only using one)
    if num_detected_objects > 0:
        avg_motion_loss = motion_loss / num_detected_objects
    else:
        avg_motion_loss = torch.tensor(0.0)
    
    return avg_motion_loss        