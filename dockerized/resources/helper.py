from string import punctuation
from spacy.lang.en import STOP_WORDS
from model import SentimentRNN
import pickle
import numpy as np
import torch



train_on_gpu=torch.cuda.is_available()

model_path = 'model.pt'
vocab_path = 'vocab_to_int.sav'

# load vocab
vocab_to_int = pickle.load(open(vocab_path, 'rb'))

def get_model():
    vocab_size = (len(vocab_to_int)) + 1
    output_size = 1
    embedding_dim = 300
    hidden_dim = 256
    n_layers = 2
    model = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return model
    


def tokenize_review(review):
    review = review.lower() # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in review if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int.get(word, 0) for word in test_words if word not in STOP_WORDS])

    return test_ints


def pad_features(reviews_ints, seq_length):
    
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    
    for i, review in enumerate(reviews_ints):
        features[i, -len(review):] = np.array(review)[:seq_length]
    
    return features


def predict(model, test_review, sequence_length=200):

    model.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length=sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = model.init_hidden(batch_size)

    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = model(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    # print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if(pred.item()==1):
        # print("Positive review detected!")
        confidence = output.item()
    else:
        # print("Negative review detected.")
        confidence = 1 - output.item()
    
    return pred.item(), confidence
