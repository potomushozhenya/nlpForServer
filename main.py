import torch
import numpy as np
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import f1_score
from tabulate import tabulate
from sklearn.datasets import fetch_20newsgroups

categories = ['rec.autos', 'comp.graphics', 'sci.space']
newsgroup = fetch_20newsgroups(subset='all', categories=categories, shuffle=True,
                               remove=('headers', 'footers', 'quotes'))
device = torch.device("cuda:0")
print(device)


def train(x_train, x_test, y_train, y_test, clf):
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    return f1_score(y_test, y_pred, average='weighted')


bertMap = ["base-uncased", "large-uncased", "roberta"]
clfMap = ["RndmForset", "GBM", "ADA"]
table = [[" "] + clfMap]

for i in range(len(bertMap)):
    tokenizer = 0
    model = 0
    if i == 0:
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
    if i == 1:
        model_name = 'bert-large-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
    if i == 2:
        model_name = 'roberta-base'
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to(device)

    max_length = 512  # Set the maximum sequence length
    tokenized_texts = []
    for text in newsgroup.data:
        tokens = tokenizer.tokenize(text)
        # Truncate tokens if they exceed max_length - 2 (to account for [CLS] and [SEP])
        tokens = tokens[:max_length - 2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        tokenized_texts.append(tokens)

    input_ids = []
    attention_masks = []  # Initialize attention masks list
    for tokens in tokenized_texts:
        # Convert tokens to ids
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        # Pad input_id with 0s if its length is less than max_length
        input_id += [0] * (max_length - len(input_id))
        input_ids.append(input_id)

        # Create attention mask
        att_mask = [1] * len(tokens) + [0] * (max_length - len(tokens))
        attention_masks.append(att_mask)

    input_ids = torch.tensor(input_ids).to(device)
    attention_masks = torch.tensor(attention_masks).to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_masks)  # Pass attention masks to model

    last_hidden_states = outputs.last_hidden_state
    embeddings = last_hidden_states[:, 0, :]

    if device == torch.device("cuda:0"):
        embeddings = embeddings.cpu()

    dataVec = embeddings.numpy()
    X_train, X_test, y_train, y_test = train_test_split(dataVec, newsgroup.target, test_size=0.33)
    smallArr = [bertMap[i]]
    for j in range(len(clfMap)):
        clf = 0
        if j == 0:
            clf = RandomForestClassifier()
        if j == 1:
            clf = GradientBoostingClassifier(n_estimators=125)
        if j == 2:
            clf = AdaBoostClassifier(n_estimators=200, algorithm='SAMME')
        f1 = train(X_train, X_test, y_train, y_test, clf)
        smallArr.append(f1)
    table.append(smallArr)

print(tabulate(table))