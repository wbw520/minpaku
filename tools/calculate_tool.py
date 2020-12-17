import torch
import numpy as np
import matplotlib.pyplot as plt
import itertools


def evaluateTop1(logits, labels):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return torch.eq(pred, labels).sum().float().item()/labels.size(0)


def evaluateTop5(logits, labels):
    with torch.no_grad():
        maxk = max((1, 5))
        labels_resize = labels.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        return torch.eq(pred, labels_resize).sum().float().item()/labels.size(0)


class MetricLog():
    def __init__(self):
        self.record = {"train": {"loss": [], "p_loss": [], "acc_1": [], "acc_5": []},
                       "val": {"loss": [], "p_loss": [], "acc_1": [], "acc_5": []}}

    def print_metric(self):
        print("train p loss:", self.record["train"]["p_loss"])
        print("val p loss:", self.record["val"]["p_loss"])
        print("train loss:", self.record["train"]["loss"])
        print("val loss:", self.record["val"]["loss"])
        print("train acc_1:", self.record["train"]["acc_1"])
        print("val acc_1:", self.record["val"]["acc_1"])
        print("train acc_5:", self.record["train"]["acc_5"])
        print("val acc_5:", self.record["val"]["acc_5"])


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
