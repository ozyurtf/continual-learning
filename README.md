## Dataset

The full dataset can be downloaded from here: [http://clevrer.csail.mit.edu](http://clevrer.csail.mit.edu).  

## Different Training Architectures

1) In this approach, we can use an image encoder (e.g., CLIP, ViT), an LSTM cell, and separate decoder heads for frames, optical flows, object states, and collisions. Instead of decoding frames directly from the hidden state, we can use the predicted optical flow to warp the current frame into an initial estimate of the next frame, and a small residual network to correct the artifacts introduced by the warping process.                                    

At each time step, the LSTM takes the CLIP features of the current frame as input and updates its hidden state. From this hidden state, the flow head predicts the optical flow between the current and next frame. This predicted flow is then used to warp the current frame via backward warping with bilinear interpolation. The residual head then takes the warped frame and produces a small correction, and the final predicted frame is the sum of the warped frame and this correction. The state head and collision head also read from the same hidden state and predict the bounding box state and whether a collision occurs at the next time step respectively.

During inference, the first x real frames are fed sequentially through the LSTM to build up the hidden state. Then for each of the y prediction steps, the flow head predicts the next flow, the current frame is warped, and the residual head refines the result. To advance the hidden state for subsequent prediction steps, the CLIP features of the previously predicted frame can be fed back into the LSTM.

Pros:
- Warping preserves the majority of pixel content from the previous frame, making temporal consistency easier to maintain than full-frame decoding.
- Frame and flow predictions are consistent with each other by construction — if the flow is accurate, the warped frame is accurate.
- The residual head has a much easier task than a full decoder since it only needs to correct occlusion boundaries and small misalignments rather than generate an entire frame.
- No separate VAE pre-training stage required — the full pipeline is trained end-to-end.
- Collision annotations can directly supervise the collision head, providing a structured signal that the other approaches lack.

Cons:
- The quality of frame predictions is tightly coupled to the quality of flow predictions. If the LSTM predicts inaccurate flow, the warped frame will be misaligned and the residual head may not be able to fully compensate.
- Artifacts from warping compound over multiple prediction steps during inference because each predicted frame becomes the source for the next warp.
- CLIP was trained on general internet images rather than synthetic physics simulations, so its frame representations may not be optimally suited for capturing the motion dynamics of CLEVRER objects.

1) We can build and train a VAE model to represent the video frames in a lower dimension. Once the VAE is trained, we can encode all the videos into lower-dimensional latents. In the next step, we can train another model (e.g., a transformer-based or diffusion model) to predict the latent variable at t + 1 based on the latent variable at t. And during inference, we can give the latents to this model, predict the next latents, use the VAE decoder to decode these latents, and predict/reconstruct the next frame. During the training of the VAE model, we can use reconstruction loss + KL divergence as our loss function. For the second model, L1 or L2 loss can be a good option.

Pros:
- Very simple method. 
- Clear separation of tasks: VAE handles compression, and the second model handles capturing the temporal relationship. 

Cons: 
- The temporal model has to figure out motion, occlusion, and the changes of the appearances of the objects during the video.
- It has to learn from scratch that pixels should stay coherent in the generated videos.

2) Similar to the first approach, we can train a VAE model to convert the video frames into lower-dimension latents. And for the temporal model, instead of predicting the next latent, we can use two heads and predict the next latent and optical flow field between the current step and the next step. 

We can give the current latent as input to the head of the temporal model to predict the latent in the next step. And we can give the current latent and next latent as input to the second head to predict the optical flow between them. 

During inference, we can discard the second head since its main purpose was to ensure that the model parameters get a sense of how the motion might look between steps. And we can only use the first head to predict the next latent, and use the VAE decoder to reconstruct the next frame from the next latent.

To obtain ground truth optical flows, we can use a pre-trained optical model and generate the optical flow field between each consecutive frame in the videos. 

Pros:
- Flow supervision might help the model to take the motion into account and predict the next frames more accurately.

Cons: 
- Need to compute optical flow.
- The quality of the flow supervision depends on whichever model we use to compute the flow.

3) As before, we can train a VAE model to represent the video frames in a lower dimension. And in the temporal model, we can use 3 heads: 

- Flow predictor: For taking the latent at time step t and predicting the optical flow between t and t + 1. 
- Occlusion mask predictor: For taking the latent at step t and predicting which regions in the frame are trustworthy versus occluded. This runs in parallel with the flow predictor. 
- Residual predictor: For taking the warped latent and correcting the errors in it. 

And we can use the predicted optical flow to transform the latent at time step t into an initial estimate of the latent at time step t + 1. This is called warping, and we can use bilinear interpolation, which is differentiable, to do that.

Note that after warping, new regions appear (disocclusion), and objects move and hide pixels (occlusion). As a result of this, we might see that some areas in the image become empty, some areas are misaligned, and other areas are duplicated. 

Instead of letting the residual predictor figure out how to handle all of these distinct scenarios, we can identify these problematic regions with an occlusion mask, and then combine the warped frame with newly generated residual content using this occlusion mask (Final prediction = Warped latent x Occlusion mask + Residual). The residual predictor still looks at the entire warped latent and produces corrections for all pixels, but the mask controls how much each component contributes in the final combination. It ensures the system does not force motion to explain pixels that cannot be obtained by moving previous-frame content, which is essential for realistic video prediction and generation.

And one way to obtain an occlusion mask is to take the optical flows and apply forward-backward consistency on them. In this matrix, a score between 0 and 1 is given to each pixel in the warped frame, and these scores tell us which parts of the warped frame come from a visible pixel in the previous frame (1) and which parts were hidden before (0).

And during the training process, we can use an additional head to predict these occlusion masks.

Pros: 
- Warping process preserves the majority of the content from the previous frame.
- Strong temporal consistency because of the warping process.
- Warping + residual is easier to learn compared to full-frame prediction

Cons: 
- We rely on optical flow too much 
- If optical flow is not good, the residual predictor has to close the gap, which may not be possible.
- May not perform well if there are sudden, chaotic or highly unpredictable motions.

4) In this approach, we train two VAEs: one for encoding and decoding video frames, and another for encoding and decoding optical flows. And we concatenate the frame latents and optical flow latents in time step t, and use this as input to the temporal model to predict the next frame latents and optical flow latents. During the training process of the temporal model, we supervise each prediction against the corresponding ground truth latent. 

One note is that the system can try to represent both appearance features and optical flow features in the same latent space because the goal is to reconstruct the frames and optical flows accurately, and there is no way to prevent the optical flow information and appearance information to be mixed in the latent space without special constraints.

If our only goal is to predict the next frames accurately, this may not be a problem. But if we want independent control over appearance and motion at inference time (e.g., animating a still image, motion transfer, restyling a video while keeping motion), this shouldn't happen. 

One way to prevent the appearance features and optical flow features from being represented in the same latent space is to decode the frames using the appearance latent during the training process and using an additional loss that measures the error between the optical flow decoded from the latent and ground truth optical flow. But this is not 100% guaranteed solution. 

Pros: 
- If disentanglement works well, we have powerful controllability. We can fix the appearance latent and vary the optical flow latent and generate videos with different motions. We can also do the opposite of this. 

Cons: 
- Disentanglement is hard to achieve because two latent streams tend to leak information into each other. 
- Training complexity is higher than the other methods.

## Optical Flow Model

To obtain ground truth optical flows for supervising the flow head, we can use a pre-trained optical flow model such as RAFT[1], FlowFormer[2], SEA-RAFT[3] or WAFT[4] and precompute the flow fields between each consecutive frame offline. 

In this system, I am using RAFT.

## Test-Time Training (TTT)

Assuming we choose approach 3, we can utilize the flow predictor head during test-time training. After training the 3 heads jointly, we can continue updating the flow predictor for each video during inference and reset it back to its pre-trained version for each new video.

## Loss Functions

### VAE 
- Reconstruction loss: Pixel-wise L1 loss between the frames decoded from the latents and the actual frames. 
- KL divergence: Forces the latents to follow a standard gaussian distribution to ensure standardized latent space and make things easier for the temporal model. 

### Temporal Model 
- Latent prediction loss: L1 loss between the predicted next latent and actual next latent.
- Flow loss: The euclidian distance between the predicted optical flow and the actual optical flow.
- Occlusion mask loss: Binary cross entropy-loss between the predicted occlusion mask and forward-backward consistency occlusion mask that is computed from optical flows.
- Warped latent loss: L1 loss between the frame decoded from the warped latent (before residual correction) and the actual next frame.
- Residual loss: L1 loss between the predicted residual and the actual residual
- Collision loss: Focal loss between the predicted collision probability and ground truth binary value that tells us whether collision happened in the specific time frame. The reason why focal loss is used instead of binary-cross entropy is because collisions are rare.

### Test-Time Training

- Frame reconstruction loss: At each step k, warp frame t using the flow predictor's output and compare the warped result against the actual frame t+1 with L1 loss.

## Evaluation Metrics

- PSNR
- SSIM
- LPIPS

