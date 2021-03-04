import random
import json
import torch
from model import NeuralModel
from nltk_func import bag_of_words, tokenize
import func

# check if GPU Compute Unified Device Architecture available
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)
    
# load trained data
data_file = 'data.pth'
data = torch.load(data_file)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

model = NeuralModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Maartenok'

def get_response(message):
    sentence = tokenize(message)
    bow = bag_of_words(sentence, all_words) # create numpy_array
    bow = bow.reshape(1, bow.shape[0]) # reshape to 1 row and 0 columns
    bow = torch.from_numpy(bow) # creates a Tensor from a numpy.ndarray
    
    output = model(bow) # get predictions
    # return max prediction value and index of tag
    _max_p, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()] # get tag name
    
    # get probabilities and probability of predicted item
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
        
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == 'current_time':
                return func.current_time()
            elif tag == 'day_today':
                return func.day_today()
            if tag == intent['tag']:
                return random.choice(intent["responses"])
                # if intent["responses"][0].startswith('funtion'):
                    # return func.current_time()
                # else:
    else:
       return 'I do not undestand, friend:('

# print("I am your friend! Speak with me!")

# while True:
    # quit_words = ['exit', 'quit', 'leave']
    # sentence = input("You: ")
    # if sentence.lower()in quit_words:
        # print(bot_name + ': Until next time!')
        # break
    # sentence = tokenize(sentence)
    # bow = bag_of_words(sentence, all_words) # create numpy_array
    # bow = bow.reshape(1, bow.shape[0]) # reshape to 1 row and 0 columns
    # bow = torch.from_numpy(bow) # creates a Tensor from a numpy.ndarray
    
    # output = model(bow) # get predictions
    # return max prediction value and index of tag
    # _max_p, predicted = torch.max(output, dim=1)
    # tag = tags[predicted.item()] # get tag name
    
    # get probabilities and probability of predicted item
    # probs = torch.softmax(output, dim=1)
    # prob = probs[0][predicted.item()]
    
    # if prob.item() > 0.75:
        # for intent in intents['intents']:
            # if tag == intent['tag']:
                # if intent["responses"][0].startswith('print'):
                    # exec(intent["responses"][0])
                # else:
                    # print(f'{bot_name}: {random.choice(intent["responses"])}')
    # else:
        # print(f'{bot_name}: I do not undestand, friend:(')