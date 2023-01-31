import torch
import numpy as np

from torch import tensor
from transformers import RobertaTokenizer

class TaskSI:
    def __init__(self, model):
        self.model = model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.ids_to_labels = {0: 'I-Prop', 1: 'O'}
        self.labels_to_ids = {'I-Prop': 0, 'O': 1}

    def get_prediction(self, text):
        encoding = self.tokenizer(text,
                  padding='max_length',
                  truncation=True,
                  max_length=512)

        with torch.no_grad():
            result1 = self.model(tensor([encoding['input_ids']]), attention_mask=tensor([encoding['attention_mask']]))
        preds = torch.argmax(result1[0].view(-1, self.model.num_labels), axis=1).numpy()

        index_labels = [0] * len(text.split(' '))
        for index in range(len(index_labels)):
            if self.ids_to_labels[preds[index]] == 'I-Prop':
                index_labels[index] = 1

        span_started = False
        text_array = text.split(' ')
        word_spans = []
        words = ""
        for idx in range(len(index_labels)):
            if index_labels[idx] == 1 and not span_started:
                span_started = True
                words = words + text_array[idx] + ' '
            elif index_labels[idx] == 1 and idx == len(index_labels) - 1:
                words = words + text_array[idx] + ' '
                if len(words.strip()) > 0:
                    word_spans.append(str(words.strip()))
                    words = ""
            elif index_labels[idx] == 1 and span_started:
                span_started = True
                words = words + text_array[idx] + ' '

            elif index_labels[idx] == 0 or idx == len(index_labels) - 1:
                span_started = False
                if len(words.strip()) > 0:
                    print("Writing and resetting Span:", words)
                    word_spans.append(str(words.strip()))
                    words = ""

        return str(word_spans)