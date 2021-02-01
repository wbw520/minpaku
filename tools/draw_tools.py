import matplotlib.pyplot as plt
import numpy as np
import cv2
from .color_board import *


def make_distribution(data, args, phase):
    plt.figure(figsize=(25, 5))
    plt.bar(range(len(data)), [k for v, k in data.items()])
    plt.xticks(range(len(data)), [v for v, k in data.items()], rotation=60, fontsize=12)
    plt.title(f"data distribution of {args.data_type}_{args.use_index}_index in phase {phase}", fontsize='20')
    plt.savefig(f"/home/wangbowen/data_{args.data_type}_{args.use_index}_index_{phase}.png", bbox_inches='tight')
    plt.show()


def show_single(name, label, rank):
    # show single image
    image = cv2.imread(name, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.title(label + "_" + str(rank))
    plt.show()


def plot_embedding(data, label, cat, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure(dpi=600, facecolor='#FFFFFF')
    for i in range(len(data)):
        if i % 100 == 0:
            print(f"{i}/{len(data)}")
        main_color = main_cat[cat.index(label[i])]
        plt.scatter([data[i][0]], [data[i][1]], color=main_color, zorder=5, s=50)
    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=14)
    # plt.savefig(title + "_t-SNE.png")
    plt.show()