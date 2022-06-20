#Load environment
!pip install transformers
!pip install fastBPE
!pip install fairseq

!pip install vncorenlp
!mkdir -p vncorenlp/models/wordsegmenter
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab
!wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr
!mv VnCoreNLP-1.1.1.jar vncorenlp/ 
!mv vi-vocab vncorenlp/models/wordsegmenter/
!mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/


!wget https://public.vinai.io/PhoBERT_base_transformers.tar.gz
!tar -xzvf PhoBERT_base_transformers.tar.gz

import numpy as np
from sklearn.metrics import f1_score, accuracy_score

import random
import time
from tqdm import notebook
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
from sklearn.model_selection import train_test_split
import argparse
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--bpe-codes', 
    default="/content/PhoBERT_base_transformers/bpe.codes",
    required=False,
    type=str,
    help='path to fastBPE BPE'
)
args, unknown = parser.parse_known_args()
bpe = fastBPE(args)

# Load the dictionary
vocab = Dictionary()
vocab.add_from_file("/content/PhoBERT_base_transformers/dict.txt")

import pandas as pd
data=pd.read_excel("/content/drive/MyDrive/data.xlsx")
X=list(data['text'])
y=list(data['Cấp 2'])

import re
def standard_data(data):
    for id in range(len(data)):
        data[id] = re.sub(r"[\.,\?]+$-", "", data[id])
        data[id] = data[id].replace(",", " ").replace(".", " ") \
            .replace(";", " ").replace("“", " ") \
            .replace(":", " ").replace("”", " ") \
            .replace('"', " ").replace("'", " ") \
            .replace("!", " ").replace("?", " ") \
            .replace("-", " ").replace("?", " ") \
            .replace("|"," ")
        data[id] = data[id].strip().lower()
        data[id] = re.sub(r'\s\s+', ' ', data[id])
    return data
X=standard_data(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=44)

MAX_LEN = 400
train_ids = []
for sent in X_train:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    train_ids.append(encoded_sent)

test_ids = []
for sent in X_test:
    subwords = '<s> ' + bpe.encode(sent) + ' </s>'
    encoded_sent = vocab.encode_line(subwords, append_eos=True, add_if_not_exist=False).long().tolist()
    test_ids.append(encoded_sent)
    
train_ids = pad_sequences(train_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
test_ids = pad_sequences(test_ids, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")

train_masks = []
for sent in train_ids:
    mask = [int(token_id > 0) for token_id in sent]
    train_masks.append(mask)

test_masks = []
for sent in test_ids:
    mask = [int(token_id > 0) for token_id in sent]

    test_masks.append(mask)

labels=list(set(y))
label=list(set(y_train))
label2id=dict([label,id] for id,label in enumerate(label))
y_train=[label2id[label] for label in y_train]
y_test=[label2id[label] for label in y_test]

train_inputs = torch.tensor(train_ids)
test_inputs = torch.tensor(test_ids)
train_labels = torch.tensor(y_train)
test_labels = torch.tensor(y_test)
train_masks = torch.tensor(train_masks)
test_masks = torch.tensor(test_masks)

train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = SequentialSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=32)

test_data = TensorDataset(test_inputs, test_masks, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=32)

from transformers import RobertaForSequenceClassification, RobertaConfig, AdamW

config = RobertaConfig.from_pretrained(
    "/content/PhoBERT_base_transformers/config.json", from_tf=False, num_labels = len(labels), output_hidden_states=False,
)
BERT_SA = RobertaForSequenceClassification.from_pretrained(
    "/content/PhoBERT_base_transformers/model.bin",
    config=config
)

BERT_SA.cuda()
print('Done')

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
    return accuracy_score(pred_flat, labels_flat)

device = 'cuda'
epochs = 5
max = 0

param_optimizer = list(BERT_SA.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=5e-7, correct_bias=False)


for epoch_i in range(0, epochs):
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_loss = 0
    BERT_SA.train()
    train_accuracy = 0
    nb_train_steps = 0
    start_time=time.time()
    for step, batch in notebook.tqdm(enumerate(train_dataloader)):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        BERT_SA.zero_grad()
        outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask, 
            labels=b_labels)
        loss = outputs[0]
        total_loss += loss.item()
        
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        tmp_train_accuracy = flat_accuracy(logits, label_ids)
        train_accuracy += tmp_train_accuracy
        nb_train_steps += 1
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(BERT_SA.parameters(), 1.0)
        optimizer.step()
        
    avg_train_loss = total_loss / len(train_dataloader)
    print(" Accuracy: {0:.4f}".format(train_accuracy/nb_train_steps))
    print(" Average training loss: {0:.4f}".format(avg_train_loss))
    BERT_SA.save_pretrained('/content/drive/MyDrive/Deep/modelbert/{}'.format(epoch_i+1))
    print("Done training this epoch for {} seconds".format(time.time()-start_time))
    print("Running Validation...")
    BERT_SA.eval()
    start_time=time.time()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    for batch in notebook.tqdm(test_dataloader):

        batch = tuple(t.to(device) for t in batch)

        b_input_ids, b_input_mask, b_labels = batch

        with torch.no_grad():
            outputs = BERT_SA(b_input_ids, 
            token_type_ids=None, 
            attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    if max < eval_accuracy/nb_eval_steps:
      max = eval_accuracy/nb_eval_steps
      e = epoch_i
    print('logits:    {}'.format(np.argmax(logits, axis=1).flatten()))
    print('labels_id: {}'.format(label_ids))
    print(" Accuracy: {0:.4f}".format(eval_accuracy/nb_eval_steps))
    print("Evaluating in {} seconds".format(time.time()-start_time))
print("Training complete!")
print('Best Valid acc : {}, epoch: {}'.format(max, e+1))





