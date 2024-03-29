from timm.models import create_model
import torch
import torch.nn.functional as F
import torch.nn as nn
from timm.models.layers import SelectAdaptivePool2d


class Identical(nn.Module):
    def __init__(self):
        super(Identical, self).__init__()

    def forward(self, x):
        return x


def load_backbone(args, num_classes):
    bone = create_model(args.base_model, pretrained=args.pre_trained,
                        num_classes=num_classes)
    bone.global_pool = Identical()
    bone.fc = Identical()
    return bone


def fix_parameter(model, name_fix, mode="open"):
    """
    fix parameter for model training
    """
    for name, param in model.named_parameters():
        for i in range(len(name_fix)):
            if mode != "fix":
                if name_fix[i] not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    break
            else:
                if name_fix[i] in name:
                    param.requires_grad = False


class MainModel(nn.Module):
    def __init__(self, args, num_classes):
        super(MainModel, self).__init__()
        self.args = args
        self.num_features = 512
        self.drop_rate = 0.0
        self.back_bone = load_backbone(args, 1000)
        # self.back_bone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)
        if args.fix:
            fix_parameter(self.back_bone, [""], mode="fix")
        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        if not args.extract:
            if args.data_type == "union":
                self.fc_location = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes["location"])
                self.fc_function = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes["function"])
            else:
                self.fc = nn.Linear(self.num_features * self.global_pool.feat_mult(), num_classes[args.data_type])
        else:
            print("use extract only")

    def forward(self, x):
        b = x.size()[0]
        f = self.back_bone(x)
        f = f.view(b, 512, 7, 7)
        f_avg = self.global_pool(f).flatten(1)
        # if self.drop_rate:
        #     f_avg = F.dropout(f_avg, p=float(self.drop_rate), training=self.training)
        if self.args.data_type == "union":
            if self.args.extract:
                return None, f_avg
            else:
                x_location = self.fc_location(f_avg)
                x_function = self.fc_function(f_avg)
                return {"location": x_location, "function": x_function}, f_avg
        else:
            if self.args.extract:
                return None, f_avg
            else:
                x = self.fc(f_avg)
                return {self.args.data_type: x}, f_avg


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


class PairLoss(nn.Module):
    def __init__(self, args):
        super(PairLoss, self).__init__()
        self.args = args
        print("use pair loss")

    def forward(self, features, label_names):
        batch = features.size()[0]
        counter_1, counter_2 = pair_counting(self.args, label_names, features.device, batch)
        distance = get_metric("euclidean")(features, features)
        distance = distance * counter_1
        distance_loss = torch.sigmoid(distance) - counter_2
        return torch.mean(distance_loss)

