import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


def evaluateTop1(logits, labels):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        mask = torch.ne(labels, 255)
        pred = pred[mask]
        labels = labels[mask]
        return torch.eq(pred, labels).sum().float().item()/labels.size(0)


def evaluateTop5(logits, labels):
    with torch.no_grad():
        maxk = max((1, 5))
        labels_resize = labels.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        mask = torch.ne(labels, 255)
        pred = pred[mask]
        labels_resize = labels_resize[mask]
        return torch.eq(pred, labels_resize).sum().float().item()/labels_resize.size(0)


class MetricLog():
    def __init__(self):
        self.record = {"train": {"location": {"loss": [], "p_loss": [], "acc_1": [], "acc_5": []},
                                 "function": {"loss": [], "p_loss": [], "acc_1": [], "acc_5": []}},
                       "val": {"location": {"loss": [], "p_loss": [], "acc_1": [], "acc_5": []},
                               "function": {"loss": [], "p_loss": [], "acc_1": [], "acc_5": []}}}

    def print_metric(self):
        print(self.record)


def matrixs(pre, true, cat, args):
    l = len(cat)
    matrix = np.zeros((l, l), dtype="float")
    for i in range(len(pre)):
        matrix[int(true[i])][int(pre[i])] += 1
    MakeMatrix(matrix, "no_normalized_"+args.data_type, cat).draw()
    print("all class acc: ", round(np.sum(np.diagonal(matrix))/np.sum(matrix), 3))
    for j in range(l):
        print(j, np.sum(matrix[j]))
        matrix[j] = matrix[j]/np.sum(matrix[j])
    MakeMatrix(matrix, "normalized_"+args.data_type, cat).draw()


class MakeMatrix():
    def __init__(self, matrix, name, class_name):
        self.matrix = matrix
        self.classes = class_name
        self.classes2 = class_name
        self.name = name

    def draw(self):
        plt.figure(dpi=600, facecolor='#FFFFFF')
        self.plot_confusion_matrix(self.matrix, self.classes, normalize=False,
                                   title=self.name)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(type(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, self.classes2, rotation=90, size=3)
        plt.yticks(tick_marks, classes, size=3)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, None,
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", size=5)

        plt.tight_layout()
        plt.ylabel('True', size="26")
        plt.xlabel('Predict', size="26")
        plt.savefig("/home/wangbowen/PycharmProjects/minpaku/results/matrix_" + self.name + ".png", dpi=600,  bbox_inches='tight')
        # plt.show()
