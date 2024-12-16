import util
from NBBaseline import NBBaseline
from LogReg import LogReg

categories = ["health", "history", "religion", "science", "technology", "thearts"]

def LoadData(num):
    train_data = []
    for i, cat in enumerate(categories):
        filename = "clean_" + cat + str(num) + ".txt"
        f = open(filename, "r", encoding="utf-8")
        text = f.read()
        words = text.split()
        train_data.append(tuple([words, i]))

    return train_data

def LoadDevSet():
    dev_set = []
    for i, cat in enumerate(categories):
        filename = "clean_" + cat + "20dev.txt"
        f = open(filename, "r", encoding="utf-8")
        text = f.read()
        tests = text.split("\n")
        for test in tests:
            words = test.split()
            dev_set.append(tuple([words, i]))

    return dev_set
        
        

train_data = LoadData(20) # Change this line to change the sample

dev_set = LoadDevSet()

model = NBBaseline(train_data) # Change this line to change the model

true_labels = []
predicted_labels = []

for tup in dev_set:
    true_labels.append(tup[1])
    predicted_labels.append(model.label(tup[0]))

acc = util.accuracy(true_labels, predicted_labels)

precision = util.precision(true_labels, predicted_labels, range(len(categories)))

recall = util.recall(true_labels, predicted_labels, range(len(categories)))

f1 = util.f1score(true_labels, predicted_labels, range(len(categories)))

print("acc: " + str(round(acc, 3)))

print("precision: " + str(round(precision, 3)))

print("recall: " + str(round(recall, 3)))

print("f1: " + str(round(f1, 3)))
