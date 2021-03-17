import json
import numpy as np
import pandas as pd
import pathlib

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from transformers import BertForSequenceClassification
from transformers import BertTokenizer

from IPython.display import clear_output
from sklearn.metrics import accuracy_score

from train_text_classifier import create_mini_batch


CALL_TEXT = 'call_text'


class CustomDataset(Dataset):
    """Convert data to tokens, segment and class tensor.
    """
    
    def __init__(self, table, tokenizer):
        self.tokenizer = tokenizer
        self.texts = table[CALL_TEXT].to_list()
        self.len = len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        class_tensor = None
        
        word_pieces = ['[CLS]']
        ori_tokens = self.tokenizer.tokenize(text)
        tokens = ori_tokens[-511:]
        word_pieces += tokens
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        if len(ids) != 512:
            ids += [0] * (512-len(ids))
        
        tokens_tensor = torch.tensor(ids)
        return tokens_tensor, class_tensor
    
    def __len__(self):
        return self.len


def get_predictions(model, data_loader, predictions=None):
    confidences = None
    hidden_states = None

    with torch.no_grad():
        for data in data_loader:
            if next(model.parameters()).is_cuda:
                data = [t.to('cuda:0') for t in data if t is not None]

            tokens_tensors, masks_tensors = data
            outputs = model(
                input_ids=tokens_tensors,
                attention_mask=masks_tensors)

            hidd = outputs['hidden_states'][0]
            logits = outputs['logits']
            softmax = torch.softmax(logits, dim=1)
            conf, pred = torch.max(softmax.data, 1)

            if predictions is None:
                predictions = pred.to('cpu')
                confidences = conf.to('cpu')
                hidden_states = hidd.to('cpu')
            else:
                predictions = torch.cat((predictions, pred.to('cpu')))
                confidences = torch.cat((confidences, conf.to('cpu')))
                hidden_states = torch.cat((hidden_states, hidd.to('cpu')), 0)
    return predictions, confidences, hidden_states


class ClassifyTextWithBERT:
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.batch_size = 32
    
    def load_model(self):
        with open(f'{self.model_dir}/config.json', 'r') as f:
            config = json.load(f)
        self.pretrained_model_name = config['_name_or_path']
        
        with open(f'{self.model_dir}/class_indices.json', 'r') as f:
            self.class_indices = json.load(f)
        
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_dir,
            num_labels=len(self.class_indices),
            output_hidden_states=True)
        self.model.eval()
        
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
    
    def predict(self, file_path, report_path=None, feature_dir=None):
        table = pd.read_excel(file_path, engine='openpyxl')
        
        if table[[CALL_TEXT]].isnull().values.any():
            raise ValueError(f'Oh... is Nan. >"< {file_path}')
        
        tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_model_name)
        data = CustomDataset(
            table,
            tokenizer)
        data_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            collate_fn=create_mini_batch)
        
        predictions, confidences, hidden_states = get_predictions(
            self.model, data_loader)
        
        assert len(table) == len(predictions)
        
        # Only left...
        table = table[[CALL_TEXT]]
        
        class_indices = {v: k for k, v in self.class_indices.items()}
        table['prediction'] = [
            class_indices[i] for i in predictions.numpy()]
        
        table['confidence'] = [i for i in confidences.numpy()]
        table['model'] = [self.model_dir]*len(predictions)
        
        if feature_dir is not None:
            pathlib.Path(feature_dir).mkdir(
                parents=True, exist_ok=True)
            
            features = [i.flatten() for i in hidden_states.numpy()]
            feature_path = [f'{feature_dir}/feature_{i:06}'
                            for i in range(len(features))]
            
            for i in range(len(features)):
                np.save(feature_path[i], features[i])
            
            table['feature'] = [f'{i}.npy' for i in feature_path]
        
        if report_path is not None:
            pathlib.Path(report_path).parent.mkdir(
                parents=True, exist_ok=True)
            table.to_excel(report_path, index=False)
        return table


if __name__ == '__main__':
    rosa = Rosa('./Model')
    rosa.load_model()
    table = rosa.predict(
        './call_text_1.xlsx',
        './Prediction1/call_text_1.xlsx',
        './Prediction1/Feature')
