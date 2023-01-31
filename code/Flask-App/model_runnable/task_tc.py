import glob
import os.path
import codecs
import torch
import numpy as np

from torch import cuda, tensor
from sklearn import preprocessing
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

label_to_class_map = {0: 'Appeal_to_Authority',
                      1: 'Appeal_to_fear-prejudice',
                      2: 'Bandwagon,Reductio_ad_hitlerum',
                      3: 'Black-and-White_Fallacy',
                      4: 'Causal_Oversimplification',
                      5: 'Doubt',
                      6: 'Exaggeration,Minimisation',
                      7: 'Flag-Waving',
                      8: 'Loaded_Language',
                      9: 'Name_Calling,Labeling',
                      10: 'Repetition',
                      11: 'Slogans',
                      12: 'Thought-terminating_Cliches',
                      13: 'Whataboutism,Straw_Men,Red_Herring'}
class_to_label_map = {'Appeal_to_Authority': 0,
                      'Appeal_to_fear-prejudice': 1,
                      'Bandwagon,Reductio_ad_hitlerum': 2,
                      'Black-and-White_Fallacy': 3,
                      'Causal_Oversimplification': 4,
                      'Doubt': 5,
                      'Exaggeration,Minimisation': 6,
                      'Flag-Waving': 7,
                      'Loaded_Language': 8,
                      'Name_Calling,Labeling': 9,
                      'Repetition': 10,
                      'Slogans': 11,
                      'Thought-terminating_Cliches': 12,
                      'Whataboutism,Straw_Men,Red_Herring': 13}


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load("/content/drive/MyDrive/NLP/results/results/bert_no_wordNet_com_e5_iter1.pt", map_location=torch.device('cpu'))
model.eval()
sigmoid = torch.nn.Sigmoid()

def get_prediction(text):
    encoding = tokenizer([text], truncation=True, padding=True, max_length=512)
    result = model(tensor(encoding['input_ids']), attention_mask=tensor(['attention_mask']))

    pred_probs = sigmoid(result.logits[0]).detach().numpy()

    pred_probs_cutoff = np.zeros(pred_probs.shape)
    pred_probs_cutoff[np.where(pred_probs >= 0.90)] = 1

    if sum(pred_probs_cutoff) == 0:
        pred_probs_cutoff[np.argmax(pred_probs)] = 1

    predicted_classes = [label_to_class_map[idx] for idx, label in enumerate(pred_probs_cutoff) if label == 1.0]
    return str(predicted_classes)
