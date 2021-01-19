import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from loaders.transform_func import make_transform
from tqdm.auto import tqdm
from torch.utils.data import DistributedSampler
from tools.draw_tools import make_distribution
import argparse
from args import get_args_parser
import tools.prepare_things as prt
import json
from loaders.pre_prosess import selected, Minpaku
import csv


class MinpakuGenetator(Dataset):
    def __init__(self, args, data, class_index, transform=None):
        super(MinpakuGenetator, self).__init__()
        self.args = args
        self.use_index = args.use_index
        self.use_label_num = args.use_label_num
        self.image_root = args.dataset_dir
        self.data, self.index_records = self.reconstruction(data)
        self.class_index = class_index
        self.transform = transform

    def deal_label(self, input, class_index):
        return class_index.index(input)

    def reconstruction(self, data):
        index_records = Minpaku(self.args).construction()
        all_data = []
        start = 0
        for ll in selected["location"]:
            for ff in selected["function"]:
                current_data = data[ll][ff]
                length = len(current_data)
                index_records[ll][ff].append([start, start+length-1])
                all_data += current_data
        return all_data, index_records

    def __getitem__(self, index):
        ID, img_dir = self.data[index]["id"], self.data[index]["path"]
        image_path = os.path.join(self.image_root, 'images', img_dir)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img, 2)
        if self.args.data_type != "union":
            type = self.data[index][self.args.data_type]
            label = self.deal_label(type[self.use_label_num][:self.use_index], self.class_index[self.args.data_type])
            label = torch.from_numpy(np.array(label))
            return {"image": img, "label_"+self.args.data_type: label, "path": image_path,
                    self.args.data_type+"_name": type[self.use_label_num][:self.use_index], "label_"+self.args.data_type+"_full": type[self.use_label_num]}
        else:
            type_location = self.data[index]["location"]
            type_function = self.data[index]["function"]
            label_location = self.deal_label(type_location[self.use_label_num][:self.use_index], self.class_index["location"])
            label_function = self.deal_label(type_function[self.use_label_num][:self.use_index], self.class_index["function"])
            return {"image": img, "label_location": label_location, "label_function": label_function, "path": image_path,
                    "location_name": type_location[self.use_label_num][:self.use_index], "label_location_full": type_location[self.use_label_num],
                    "function_name": type_function[self.use_label_num][:self.use_index], "label_function_full": type_function[self.use_label_num]}

    def __len__(self):
        return len(self.data)


def get_minpaku_data():
    with open(os.path.join("../config/train.json"), "r") as load_train:
        train_data = json.load(load_train)
        statistic(train_data)

    with open(os.path.join("../config/valid.json"), "r") as load_val:
        val_data = json.load(load_val)

    with open(os.path.join("../config/test.json"), "r") as load_test:
        test_data = json.load(load_test)

    with open(os.path.join("../config/category.json"), "r") as load_category:
        category = json.load(load_category)

    return {"train": train_data, "val": val_data, "test": test_data}, category


def prepare_dataloaders(args):
    print("start load data")
    total_data, index = get_minpaku_data()
    dataset_train = MinpakuGenetator(args, total_data["train"], index, transform=make_transform(args, "train"))
    dataset_val = MinpakuGenetator(args, total_data["val"], index, transform=make_transform(args, "val"))
    dataset_test = MinpakuGenetator(args, total_data["test"], index, transform=make_transform(args, "inference"))
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
        sampler_test = DistributedSampler(dataset_test, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test, num_workers=args.num_workers)
    print("load data over")
    return {"train": data_loader_train, "val": data_loader_val, "test": data_loader_test}, index, {"location": len(index["location"]), "function": len(index["function"])}


def statistic(data):
    a = len(selected["location"])
    b = len(selected["function"])
    count = np.zeros((a, b))
    for i in range(len(data)):
        x = data[i]["location"][0][:2]
        y = data[i]["function"][0][:2]
        count[selected["location"].index(x)][selected["function"].index(y)] += 1
    make_csv(count, "statistic")


def make_csv(data, name):
    print("save count")
    f_val = open(name + ".csv", "w", encoding="utf-8")
    csv_writer = csv.writer(f_val)
    csv_writer.writerow([""] + selected["function"])
    for i in range(len(data)):
        csv_writer.writerow([selected["location"][i]] + list(data[i]))
    f_val.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    prt.init_distributed_mode(args)
    loaders = prepare_dataloaders(args)
    # for i_batch, sample_batch in enumerate(tqdm(loaders["train"])):
    #     print(sample_batch["image"].size())
    #     break





