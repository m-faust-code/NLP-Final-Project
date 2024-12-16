from typing import Iterable, Mapping, Sequence
import numpy as np


class NBBaseline:
    def __init__(self, data : Iterable[tuple[Sequence[str], int]]):

        self.ulm = {}
        label_data = {}
        self.vocab = set()
        for tup in data:
            if tup[1] in label_data:
                label_data[tup[1]].extend(tup[0])
            else:
                label_data[tup[1]] = tup[0]
            self.vocab = self.vocab.union(tup[0])
        
        for label in label_data:
            self.ulm[label] = self.get_logfreqs(label_data[label])

    def get_logfreqs(self, data : Iterable[str]) -> Mapping[str, int]: 
        counts = {}
        total = 0
        for w in data:
            counts[w] = counts.get(w, 0) + 1
            total += 1

        for v in self.vocab:
            if v not in counts.keys():
                counts[v] = 1
            else:
                counts[v] = counts[v] + 1
            total += 1
        
        logprobs = {}
        for w, c in counts.items():
            logprobs[w] = np.log2(c) - np.log2(total)
        
        return logprobs

    def label(self, example : Sequence[str]) -> int:
        if type(example) == str:
            example = example.split()
        totalProbs = {}
        for w in example:
            if w in self.vocab:
                for label in self.ulm:
                    if w in self.ulm[label]:
                        totalProbs[label] = totalProbs.get(label, 0) + self.ulm[label][w]

        bestLabel = None
        bestProb = None
        for label in totalProbs:
            if bestProb == None or totalProbs[label] > bestProb:
                bestLabel = label
                bestProb = totalProbs[label]

        return bestLabel

        
