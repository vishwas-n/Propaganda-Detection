import torch
import numpy as np

from torch import tensor
from transformers import RobertaTokenizer

class TaskTC:
    def __init__(self, model):
        self.model = model
        self.sigmoid = torch.nn.Sigmoid()
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.label_to_class_map = {0: 'Appeal_to_Authority', 1: 'Appeal_to_fear-prejudice', 2: 'Bandwagon,Reductio_ad_hitlerum', 3: 'Black-and-White_Fallacy',
                              4: 'Causal_Oversimplification', 5: 'Doubt', 6: 'Exaggeration,Minimisation', 7: 'Flag-Waving', 8: 'Loaded_Language',
                              9: 'Name_Calling,Labeling', 10: 'Repetition', 11: 'Slogans', 12: 'Thought-terminating_Cliches', 13: 'Whataboutism,Straw_Men,Red_Herring'}
        self.class_to_label_map = {'Appeal_to_Authority': 0, 'Appeal_to_fear-prejudice': 1, 'Bandwagon,Reductio_ad_hitlerum': 2, 'Black-and-White_Fallacy': 3,
                              'Causal_Oversimplification': 4, 'Doubt': 5, 'Exaggeration,Minimisation': 6, 'Flag-Waving': 7, 'Loaded_Language': 8,
                              'Name_Calling,Labeling': 9, 'Repetition': 10, 'Slogans': 11, 'Thought-terminating_Cliches': 12, 'Whataboutism,Straw_Men,Red_Herring': 13}

    def get_prediction(self, text):
        encoding = self.tokenizer([text], truncation=True, padding=True, max_length=512)
        result = self.model(tensor(encoding['input_ids']), attention_mask=tensor(encoding['attention_mask']))

        pred_probs =self.sigmoid(result[0][0]).detach().numpy()

        pred_probs_cutoff = np.zeros(pred_probs.shape)
        pred_probs_cutoff[np.where(pred_probs >= 0.95)] = 1

        if sum(pred_probs_cutoff) == 0:
            pred_probs_cutoff[np.argmax(pred_probs)] = 1

        predicted_classes = [self.label_to_class_map[idx] for idx, label in enumerate(pred_probs_cutoff) if label == 1.0]
        return str(predicted_classes)