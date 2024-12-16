from typing import Sequence

def accuracy[T](true_labels : Sequence[T], predicted_labels : Sequence[T]) -> float: 
    """ Compute the accuracy of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    total_correct = 0
    for i in range(len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            total_correct = total_correct + 1
    return total_correct / len(true_labels)

def f1score[T](true_labels : Sequence[T], predicted_labels : Sequence[T], labels: Sequence[T]) -> float: 
    """ Compute the f1 of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    p = precision(true_labels, predicted_labels, labels)
    r = recall(true_labels, predicted_labels, labels)
    return (2 * p * r) / (p + r)

def precision[T](true_labels : Sequence[T], predicted_labels : Sequence[T], labels: Sequence[T]) -> float: 
    """ Compute the precision of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    if len(labels) == 2:
        matrix = confusionMatrix(true_labels, predicted_labels, labels[0])
        return matrix[0][0] / (matrix[0][0] + matrix[0][1])
    else:
        total_prec = 0
        for label in labels:
            matrix = confusionMatrix(true_labels, predicted_labels, label)
            denom = matrix[0][0] + matrix[0][1]
            if denom != 0:
                total_prec = total_prec + (matrix[0][0] / denom)
        return total_prec / len(labels)


def recall[T](true_labels : Sequence[T], predicted_labels : Sequence[T], labels: Sequence[T]) -> float: 
    """ Compute the precision of a model given the true labels and predicted labels
        Note that T is a generic type
    """
    if len(labels) == 2:
        matrix = confusionMatrix(true_labels, predicted_labels, labels[0])
        return matrix[0][0] / (matrix[0][0] + matrix[1][0])
    else:
        total_prec = 0
        for label in labels:
            matrix = confusionMatrix(true_labels, predicted_labels, label)
            denom = matrix[0][0] + matrix[1][0]
            if denom != 0:
                total_prec = total_prec + (matrix[0][0] / denom)
        return total_prec / len(labels)
    

def confusionMatrix[T](true_labels : Sequence[T], predicted_labels : Sequence[T], pos_label: T) -> Sequence[Sequence[int]]:
    matrix = [[0, 0], [0, 0]]
    for i in range(len(true_labels)):
        if true_labels[i] == pos_label and predicted_labels[i] == pos_label:
            matrix[0][0] = matrix[0][0] + 1
        elif true_labels[i] == pos_label and predicted_labels[i] != pos_label:
            matrix[1][0] = matrix[1][0] + 1
        elif true_labels[i] != pos_label and predicted_labels[i] == pos_label:
            matrix[0][1] = matrix[0][1] + 1
        else:
            matrix[1][1] = matrix[1][1] + 1
            
    return matrix


# TODO: Be sure to implement the above measures, as well as a way to construct a confusion matrix 
# for your report
