import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap
from torch.nn import functional as f
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch

def defining_bert_tokenizer(PRE_TRAINED_MODEL_NAME):
    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    return tokenizer

class ReviewDataset(Dataset):
    def __init__(self, doc, targets, tokenizer, max_len):
        self.doc = doc
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.doc)
  
    def __getitem__(self, item):
        doc = str(self.doc[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
                    doc,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    padding='max_length',
                    return_attention_mask=True,
                    return_tensors='pt',
                    truncation=True
                    )

        return {
            'doc_text': doc,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(target, dtype=torch.long)
            }

def create_data_loader(inp,label,tokenizer, max_len, batch_size):
    ds = ReviewDataset(
        doc=np.array(inp),
        targets=np.array(label),
        tokenizer=tokenizer,
        max_len=max_len
      )
    
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=2
        )

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=0.1)
        self.out1 = nn.Linear(self.bert.config.hidden_size , 256)
        self.out2 = nn.Linear(256 , 64)
        self.out3 = nn.Linear(64 , 128)
        self.out = nn.Linear(128, n_classes)
    
    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
          input_ids=input_ids,
          attention_mask=attention_mask,
          return_dict=False
        )
        output = self.drop(pooled_output)
        output= self.out1(output)
        output = F.relu(output)
        output= self.out2(output)
        output = F.relu(output)
        output= self.out3(output)
        output = F.relu(output)


        return F.softmax(self.out(output),dim=1)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
    ):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
          input_ids=input_ids,
          attention_mask=attention_mask
        )

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()

    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets)
            losses.append(loss.item())

    return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    review_texts = []
    predictions = []
    prediction_probs = []
    real_values = []

    with torch.no_grad():
        for d in data_loader:

            texts = d["doc_text"]
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
            )
            _, preds = torch.max(outputs, dim=1)

            probs = F.softmax(outputs, dim=1)

            review_texts.extend(texts)
            predictions.extend(preds)
            prediction_probs.extend(probs)
            real_values.extend(targets)

    predictions = torch.stack(predictions).cpu()
    prediction_probs = torch.stack(prediction_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    return review_texts, predictions, prediction_probs, real_values

PRE_TRAINED_MODEL_NAME = 'bert-base-cased'
tokenizer=defining_bert_tokenizer(PRE_TRAINED_MODEL_NAME)

sample_txt='Personal Health Record (Extract)\nCreated on October 24, 2019\nPatient\nSteven Fuerst\nBirthdate\nDecember 10, 1979\nRace\nInformation not\navailable'

tokens = tokenizer.tokenize(sample_txt)
token_ids = tokenizer.convert_tokens_to_ids(tokens)
encoding = tokenizer.encode(
                      sample_txt,
                      max_length=64,
                      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                      return_token_type_ids=False,
                      padding=True,
                      return_attention_mask=True,
                      return_tensors='pt',  # Return PyTorch tensors
                      truncation=True,
                    )

with open("/content/drive/MyDrive/viettel_train_input.txt","r",encoding='UTF-8') as f:
  X_train=f.read().splitlines()
with open("/content/drive/MyDrive/viettel_train_label.txt","r",encoding='UTF-8') as f:
  y_train=f.read().splitlines()
with open("/content/drive/MyDrive/viettel_test_input.txt","r",encoding='UTF-8') as f:
  X_test=f.read().splitlines()
with open("/content/drive/MyDrive/viettel_test_label.txt","r",encoding='UTF-8') as f:
  y_test=f.read().splitlines()

label=list(set(y_train))
label2id=dict([label,id] for id,label in enumerate(label))
y_train=[label2id[label] for label in y_train]
y_test=[label2id[label] for label in y_test]

MAX_LEN=500
BATCH_SIZE = 32
train_data_loader = create_data_loader(X_train,y_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(X_test, y_test, tokenizer, MAX_LEN, BATCH_SIZE)

data = next(iter(train_data_loader))

bert_model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME,return_dict=False)

model = SentimentClassifier(len(label))
model = model.to(device)
input_ids = data['input_ids'].to(device)
attention_mask = data['attention_mask'].to(device)

EPOCHS = 100

optimizer = AdamW(model.parameters(), lr=2e-4, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
    )

loss_fn = nn.CrossEntropyLoss().to(device)


history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):
  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)
  train_acc, train_loss = train_epoch(
        model,
        train_data_loader,    
       loss_fn, 
        optimizer, 
        device, 
        scheduler, 
        len(X_train) 
        )
  print(f'Train loss {train_loss} accuracy {train_acc}')
  val_acc, val_loss = eval_model(
        model,
        test_data_loader,
        loss_fn, 
        device, 
        len(X_test),
        )
  print(f'Val   loss {val_loss} accuracy {val_acc}')
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)
  history['val_acc'].append(val_acc)
  history['val_loss'].append(val_loss)
  if val_acc > best_accuracy:
    torch.save(model.state_dict(), '/content/drive/MyDrive/best_model_state1.bin')
    best_accuracy = val_acc
  test_acc, _ = eval_model(
          model,
          test_data_loader,
          loss_fn,
          device,
          len(X_test),
        )
  print('\nTest Accuracy:\n')
  print(test_acc.item())
  y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
    model,
    test_data_loader
    )