from typing import Iterable, Mapping, Sequence, Tuple, Generator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import warnings

class LogReg:
    def __init__(self, data : Iterable[tuple[Sequence[str], int]], load=False, path=None):
        warnings.filterwarnings("ignore", category=FutureWarning)
        # print("start")
        self.vocab = set()
        self.labels = set()
        if isinstance(data, Generator):
            data1 = []
        for tup in data:
            self.vocab = self.vocab.union(tup[0])
            self.labels = self.labels.union([tup[1]])
            if isinstance(data, Generator):
                data1.append(tup)
        if isinstance(data, Generator):
            data = data1
        # print("export done")

        self.w2idx = {w:i for (i, w) in enumerate(self.vocab)}
        train_data = []
        for tup in data:
            x = torch.zeros(len(self.vocab))
            if type(tup[0]) == str:
                words = tup[0].split()
            else:
                words = tup[0]
            for w in words:
                if w in self.w2idx:
                    x[self.w2idx[w]] += 1
            train_data.append((x, tup[1]))
        # print("training data done")

        if load:
            self.logisticSentiment = LogisticSentimentClassifierBatched(self.vocab, self.labels)
            self.logisticSentiment.load_state_dict(torch.load(path))
            self.logisticSentiment.eval()
            return

        self.logisticSentiment = LogisticSentimentClassifierBatched(self.vocab, self.labels)
        # print("classifier made")

        learning_rate = 0.001

        optimizer  = optim.SGD(self.logisticSentiment.parameters(), lr=learning_rate)
        # print("optimizer made")

        num_epochs = 100
        for epoch in range(num_epochs):
            # print("epoch: ", + epoch)
            random.shuffle(train_data)
            batched_train_data = batch(train_data, 20)

            for X, Y in batched_train_data:    
                # reset for the next iteration
                optimizer.zero_grad()
            
                # predict
                Y_hat = self.logisticSentiment(X)
            
                loss = 0
                
                # how bad did we predict (batched!)
                for y, y_hat in zip(Y, Y_hat):
                    loss += F.cross_entropy(y_hat, y.long())
                
                loss /= len(Y)
                
                # how should parameters be adjusted
                loss.backward()
            
                # adjust our parameters
                optimizer.step()
    
    def label(self, example: Sequence[str]) -> int:
        x = torch.zeros(len(self.vocab))
        if type(example) == str:
            words = example.split()
        else:
            words = example
        for w in words:
            if w in self.w2idx:
                x[self.w2idx[w]] += 1
        with torch.no_grad():
            out = self.logisticSentiment(x)
            best_label = None
            best_score = -np.inf
            for label in range(len(out)):
                if out[label].item() > best_score:
                    best_label = label
                    best_score = out[label].item()
            return best_label
        
    def save(self):
        return self.logisticSentiment.state_dict()

    

class LogisticSentimentClassifier(nn.Module):
    def __init__(self, vocab, labels):
        # Call the nn.Module constructor
        super(LogisticSentimentClassifier, self).__init__()

        # Run our constructor
        self.vocab = vocab
        self.labels = labels

        # Wrapping in nn.Parameter indicates that these are our
        # model's learned parameters!
        self.w = nn.Parameter(torch.empty([len(vocab), len(labels)]))
        self.b = nn.Parameter(torch.empty(len(labels)))

        # Randomly initialize the parameters Don't worry about this
        nn.init.uniform_(self.w, -1.0/np.sqrt(len(self.vocab)), 1.0/np.sqrt(len(self.vocab)))

    def forward(self, x):
        logit = torch.dot(self.w, x) + self.b
        return F.log_softmax(logit, dim=-1)
    
class LogisticSentimentClassifierBatched(LogisticSentimentClassifier):
    def __init__(self, vocab, labels):
        # Call the nn.Module constructor
        super(LogisticSentimentClassifier, self).__init__()

        # Run our constructor
        self.vocab = vocab
        self.labels = labels

        # Wrapping in nn.Parameter indicates that these are our
        # model's learned parameters!
        self.w = nn.Parameter(torch.empty([len(vocab), len(labels)]))
        self.b = nn.Parameter(torch.empty(len(labels)))

        # Randomly initialize the parameters
        nn.init.uniform_(self.w, -1.0/np.sqrt(len(self.vocab)), 1.0/np.sqrt(len(self.vocab)))

    
    def forward(self, x):
        return F.log_softmax(torch.matmul(x,self.w) + self.b, dim=-1)



def batch(data : Iterable[Tuple[torch.Tensor, int]], batch_size : int) -> Iterable[Tuple[torch.Tensor, torch.Tensor]]:
    if len(data) < batch_size:
        batch_size = len(data)
    
    results = []
    i = 0
    while i < len(data):
        vectors = []
        labels = []
        for j in range(batch_size):
            datum = data[i]
            vectors.append(datum[0])
            labels.append(datum[1])
            i = i + 1
        stacked = torch.from_numpy(np.array([torch.Tensor.cpu(ten).numpy() for ten in vectors])) # https://discuss.pytorch.org/t/speed-up-collating-very-large-tensors/84365/7
        results.append(tuple([stacked, torch.Tensor(labels)]))
    return results
