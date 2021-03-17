import json
import pandas as pd
import pathlib

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from IPython.display import clear_output


CALL_TEXT = 'call_text'
CLASS_NAME = 'class'


class CustomDataset(Dataset):
    """Convert data to tokens, segment and class tensor.
    """
    
    def __init__(self, data_dir, tokenizer):
        self.tokenizer = tokenizer
        
        self.texts, self.classes = [], []
        for file_path in pathlib.Path(data_dir).glob('**/*.xlsx'):
            table = pd.read_excel(file_path, engine='openpyxl')
            
            if table[[CALL_TEXT, CLASS_NAME]].isnull().values.any():
                raise ValueError(f'Oh... is Nan. >"< {file_path}')
            
            self.texts += table[CALL_TEXT].to_list()
            self.classes += table[CLASS_NAME].to_list()
        
        self.class_indices = list(set(self.classes))
        self.class_indices.sort()
        self.class_indices = {
            val: ind for ind, val in enumerate(self.class_indices)}
        
        self.len = len(self.texts)
        assert self.len == len(self.classes)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        class_tensor = torch.tensor(
            self.class_indices[self.classes[idx]])
        
        word_pieces = ['[CLS]']
        ori_tokens = self.tokenizer.tokenize(text)
        tokens = ori_tokens[-511:]
        word_pieces += tokens
        
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        return tokens_tensor, class_tensor
    
    def __len__(self):
        return self.len


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    
    if samples[0][1] is not None:
        class_ids = torch.stack([s[1] for s in samples])
    else:
        class_ids = None
    
    tokens_tensors = pad_sequence(
        tokens_tensors, batch_first=True)
    
    masks_tensors = torch.zeros(
        tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    return tokens_tensors, masks_tensors, class_ids


def get_predictions(model, data_loader, calculate_accuracy=True, predictions=None):
    confidences = None
    total = 0
    correct = 0
    
    with torch.no_grad():
        for data in data_loader:
            if next(model.parameters()).is_cuda:
                data = [t.to('cuda:0') for t in data if t is not None]
            
            tokens_tensors, masks_tensors, class_ids = data
            outputs = model(
                input_ids=tokens_tensors,
                attention_mask=masks_tensors)
            logits = outputs[0]
            softmax = torch.softmax(logits, dim=1)
            conf, pred = torch.max(softmax.data, 1)
            
            if calculate_accuracy:
                total += class_ids.size(0)
                correct += (pred == class_ids).sum().item()
            
            if predictions is None:
                predictions = pred.to('cpu')
                confidences = conf.to('cpu')
            else:
                predictions = torch.cat((predictions, pred.to('cpu')))
                confidences = torch.cat((confidences, conf.to('cpu')))
    
    if calculate_accuracy:
        accuracy = correct / total
    else:
        accuracy = None
    return predictions, confidences, accuracy


class ClassifyTextWithBERT:
    
    def __init__(self, data_dir, model_dir):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.pretrained_model_name = 'bert-base-multilingual-cased'
        self.batch_size = 32
        self.epochs = 10
        self.learning_rate = 1e-5
    
    def fit(self):
        tokenizer = BertTokenizer.from_pretrained(
            self.pretrained_model_name)
        data = CustomDataset(
            self.data_dir,
            tokenizer)
        class_indices = data.class_indices
        
        data_loader = DataLoader(
            data,
            batch_size=self.batch_size,
            collate_fn=create_mini_batch)
        
        model = BertForSequenceClassification.from_pretrained(
            self.pretrained_model_name,
            num_labels=len(class_indices))
        clear_output()
        
        device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Finetune Model
        model.train()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate)
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            
            for data in data_loader:
                tokens_tensors, masks_tensors, classes = [
                    t.to(device) for t in data]
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=tokens_tensors,
                    attention_mask=masks_tensors,
                    labels=classes)
                
                loss = outputs[0]
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            _, _, accuracy = get_predictions(model, data_loader)
            print(f'[epoch {epoch+1}] loss: {running_loss:.3f}, accuracy: {accuracy:.3f}')
        
        # Save Model
        model.save_pretrained(self.model_dir)
        
        js_format = json.dumps(class_indices)
        with open(f'{self.model_dir}/class_indices.json', 'w') as f:
            f.write(js_format)
        return model, class_indices


if __name__ == '__main__':
    ct = ClassifyTextWithBERT('./CallText', './Model')
    model, class_indices = ct.fit()
