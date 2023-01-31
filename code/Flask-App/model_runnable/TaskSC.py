import torch
import numpy as np

from torch import tensor
from transformers import BertTokenizer

class TaskSC():
    def __init__(self, model):
        self.model = model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_to_class_map = {0: 'NOT Propaganda', 1: 'Propaganda'}

    def get_prediction(self, text):
        encoding = self.tokenizer([text], truncation=True, padding=True, max_length=256)
        result = self.model(tensor(encoding['input_ids']), attention_mask=tensor(encoding['attention_mask']))
        class_prediction = np.argmax(result[0][0].detach().numpy())
        return str(self.label_to_class_map[class_prediction])