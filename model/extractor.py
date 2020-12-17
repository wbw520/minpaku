from timm.models import create_model
import torch
import torch.nn.functional as F


def load_backbone(args):
    bone = create_model(args.base_model, pretrained=args.pre_trained,
                        num_classes=args.num_classes)
    return bone


def get_metric(metric_type):
    METRICS = {
        'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
        'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).mean(2),
        'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
        'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
    }
    return METRICS[metric_type]


def pair_counting(args, label_names, device, batch):
    b = batch
    counter_1 = torch.ones((b, b)).to(device)
    counter_2 = torch.zeros((b, b)).to(device)
    for i in range(b):
        for j in range(b):
            if label_names[i] != label_names[j]:
                counter_1[i][j] = -1
            else:
                counter_2[i][j] = 0.5
    return counter_1, counter_2


def pair_loss(args, features, label_names):
    batch = features.size()[0]
    counter_1, counter_2 = pair_counting(args, label_names, features.device, batch)
    distance = get_metric("euclidean")(features, features)
    distance = distance * counter_1
    distance_loss = torch.sigmoid(distance) - counter_2
    return torch.mean(distance_loss)

