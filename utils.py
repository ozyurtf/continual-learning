import torch 
import json 
import matplotlib.pyplot as plt 
import torchvision.transforms as T
import matplotlib.gridspec as gridspec
import os

norm = T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                   std=(0.26862954, 0.26130258, 0.27577711))

def return_objects(img, object_detection, device): 
    input_image = norm(img/255).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = object_detection(input_image)

    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']

    threshold = 0.1
    filtered_boxes = boxes[scores > threshold]
    filtered_labels = labels[scores > threshold]
    filtered_scores = scores[scores > threshold]
    return filtered_boxes

def return_state(img, object_detection, device): 
    num_objects_batch = []
    state_batch = []
    
    for i in range(img.shape[0]):
        print(img[i].shape)
        boxes = return_objects(img[i], object_detection, device)
        num_objects = len(boxes)
        state = torch.zeros(size = (32,4), device=device)
        state[:num_objects] = boxes      
        state_stacked = torch.stack((
            state[:,0], 
            state[:,2] - state[:,0], 
            state[:,1], 
            state[:,3] - state[:,1])).T

        num_objects_batch.append(num_objects)
        state_batch.append(state_stacked)
        
    num_objects_batch = torch.tensor(num_objects_batch, device=device)
    state_batch = torch.stack(state_batch, dim=0)
    
    return num_objects_batch, state_batch

def extract_questions_answers(subfolder, file): 
    video_num = file.replace(".mp4", "").replace("video_", "")
    annotation_path = f'annotation_train/{subfolder}/annotation_' + video_num + '.json'
    with open(annotation_path, 'r') as file:
        annotation = json.load(file)

    with open('train.json', 'r') as file:
        data = json.load(file)[0]
        
    question_answer_list = data['questions']
    questions = []
    answers = []

    for question_answer in question_answer_list:
        question = question_answer['question']
        if 'answer' in question_answer.keys():
            answer = question_answer['answer']
            questions.append(question)
            answers.append(answer)
                        
        elif 'choices' in question_answer.keys():
            choices = question_answer['choices']

            for choice in choices:
                choice_id = choice['choice_id']
                choice_text = choice['choice']
                answer = choice['answer']
                
                questions.append(question + " " + choice_text)
                answers.append(answer)       
    return questions, answers                   


def visualize_comparisons(predicted, actual, subfolder, file, type, i):
    batch_size = predicted.shape[0]
    num_predictions = predicted.shape[1]
    rows = batch_size * num_predictions
    cols = 2
    
    fig2 = plt.figure(figsize=(20, rows * 6))  
    
    gs = gridspec.GridSpec(rows, cols, figure=fig2, 
                           width_ratios=[1, 1],
                           wspace=0.001,  
                           hspace=0.05)   
    
    row_idx = 0
    for batch in range(batch_size):
        predicted_batch = predicted[batch]
        actual_batch = actual[batch]
        
        for n in range(num_predictions):
            predicted_next_np = predicted_batch[n].permute(1, 2, 0).detach().cpu().numpy()
            actual_next_np = actual_batch[n].permute(1, 2, 0).detach().cpu().numpy()
            
            ax_pred = fig2.add_subplot(gs[row_idx, 0])
            ax_actual = fig2.add_subplot(gs[row_idx, 1])
            
            ax_pred.imshow(predicted_next_np, interpolation='nearest')
            ax_actual.imshow(actual_next_np, interpolation='nearest')
            
            ax_pred.axis('off')
            ax_actual.axis('off')
            
            if (type == "frames"):
                ax_pred.set_title(f'Predicted Frame {i+1+n}', 
                                fontsize=12, 
                                fontweight='semibold',
                                pad=2) 
                
                ax_actual.set_title(f'Actual Frame {i+1+n}', 
                                fontsize=12, 
                                fontweight='semibold',
                                pad=2)  
            elif (type == "flows"): 
                ax_pred.set_title(f'Predicted Flow {i+n} - {i+n+1}', 
                                fontsize=12, 
                                fontweight='semibold',
                                pad=2) 
                
                ax_actual.set_title(f'Actual Flow {i+n} - {i+n+1}', 
                                fontsize=12, 
                                fontweight='semibold',
                                pad=2)                  
            row_idx += 1
    
    if not os.path.exists(f'{type}/{subfolder}/{file}/'):
        os.makedirs(f'{type}/{subfolder}/{file}/')
    
    plt.savefig(f'{type}/{subfolder}/{file}/combined_{type}_{i}.png', 
                bbox_inches='tight', 
                dpi=300,
                pad_inches=0.1)  #
    plt.close()
