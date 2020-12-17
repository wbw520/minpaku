import argparse
from train import get_args_parser
from model.extractor import load_backbone
from loaders.load_data import prepare_dataloaders
from tqdm.auto import tqdm
import torch.nn.functional as F
import tools.prepare_things as prt
import tools.calculate_tool as cal
import numpy as np
import torch
import shutil
import os


@torch.no_grad()
def test(mm, test_loader, cat, use_save=False):
    running_corrects_1 = 0.0
    running_corrects_5 = 0.0
    all_pre = None
    all_true = None
    hehe = 0
    for i_batch, sample_batch in enumerate(tqdm(test_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        img_root = sample_batch["name"]
        outputs = mm(inputs)
        outputs = F.softmax(outputs, dim=1)
        running_corrects_1 += cal.evaluateTop1(outputs, labels)
        running_corrects_5 += cal.evaluateTop5(outputs, labels)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        if use_save:
            record_imgs(outputs, labels, img_root, cat)
        if hehe == 0:
            all_pre = outputs
            all_true = labels
            hehe = 1
        else:
            all_pre = np.concatenate((all_pre, outputs), axis=0)
            all_true = np.concatenate((all_true, labels), axis=0)
    print("acc1: ", round(running_corrects_1/len(test_loader), 3))
    print("acc5: ", round(running_corrects_5/len(test_loader), 3))
    preds = np.argmax(all_pre, axis=1)
    cal.matrixs(preds, all_true, cat, args)


def record_imgs(pred, label, name, class_name):
    pred_ = np.argmax(pred, axis=1)
    for i in range(len(name)):
        pred_class = class_name[pred_[i]]
        true_class = class_name[label[i]]
        confidence = round(pred[i][pred_[i]], 3)
        if pred_[i] == label[i]:
            copy_img(name[i], true_class, pred_class, confidence, "succeed")
        else:
            copy_img(name[i], true_class, pred_class, confidence, "failure")


def copy_img(root, true, pred, confidence, type):
    name_list = root.split("/")
    new_root = os.path.join("/home/wangbowen/minpaku_pred", args.data_type, type, true, "predicted_as_" + pred + "_" + str(confidence) + "_" + name_list[-2] + "_" + name_list[-1])
    if not os.path.exists(new_root):
        os.makedirs(new_root, exist_ok=True)
    shutil.copy(root, new_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model test script', parents=[get_args_parser()])
    args = parser.parse_args()
    prt.init_distributed_mode(args)
    model_name = "function + _checkpoint.pth"
    device = torch.device(args.device)
    model = load_backbone(args)
    model.to(device)
    loaders, category = prepare_dataloaders(args)
    checkpoint = torch.load("saved_model/" + model_name)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    test(model, loaders["test"], category)