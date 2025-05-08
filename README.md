# Continual Learning 

**Note**: Currently, the models are cheating. They memorize the past frame(s) and optical flow(s) and showing those as the prediction of the next video frame(s) and optical flow(s). I am currently working on to fix that issue.

## Next Frame(s) Prediction 

![Next Frame(s) Prediction Gif](https://imgur.com/a/dAEFZEt)

## Optical Flow(s) Prediction

![Optical Flows(s) Prediction Gif](gifs/video_00000_flow_pred.gif)

## Data

The full dataset can be downloaded from here: [http://clevrer.csail.mit.edu](http://clevrer.csail.mit.edu).  

## Training

After installing the libraries listed in `requirements.txt`, the training process can be started using the following code:  

```python 
python train.py --num_predictions 3 --embed_dim 512 --hidden_size 512 --stride 1 --num_frames 127 --resize_img 224 --patch_size 32
```  

- `num_predictions` specifies the number of predictions made in each step. For example, if set to 4, the next 4 frames, optical flows, and states are predicted in the current step. The visualizations of the frame predictions and optical flow predictions are saved into the `flows` and `frames` folders for each video separately.  
- `embed_dim` specifies the embedding dimension for CLIP's image encoder.  
- `hidden_size` specifies the size of the hidden state for the LSTM cell.  
- `stride` specifies the intervals between predictions. For instance, if the stride is set to 4 and the number of predictions to 3, the 5th, 9th, and 13th frames and the optical flows between the 1st-5th frames, 5th-9th frames, and 9th-13th frames are predicted in the first step. In the next step, the 9th, 13th, and 17th frames and optical flows between the 5th-9th frames, 9th-13th frames, and 13th-17th frames are predicted, and so on.  
- `num_frames` specifies the number of frames used to train the model. Each video contains 128 frames.  
- `resize_img` specifies the target dimensions of the images before extracting features with CLIP's image encoder.  
- `patch_size` specifies the size of the patches used to process images in CLIP's image encoder.  

These are all optional parameters, and the code can also run with the simpler command:  

```python 
python train.py
```  




