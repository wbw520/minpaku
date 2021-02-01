import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from loaders.transform_func import make_transform
from tqdm.auto import tqdm
import argparse
from args import get_args_parser
import tools.prepare_things as prt
import json
from loaders.pre_prosess import selected, Minpaku
from loaders.samplers import CategoriesSampler
import csv


class MinpakuGenetator(Dataset):
    def __init__(self, args, data, class_index, cls_num=None, transform=None):
        super(MinpakuGenetator, self).__init__()
        self.args = args
        self.use_index = args.use_index
        self.use_label_num = args.use_label_num
        self.image_root = args.dataset_dir
        self.cls_num = cls_num
        self.data, self.index_records = self.reconstruction(data)
        self.class_index = class_index
        self.transform = transform

    def deal_label(self, input, class_index):
        return class_index.index(input)

    def reconstruction(self, data):
        index_records = Minpaku(self.args).construction()
        inference_record = [[] for i in range(len(selected[self.args.data_type]))]
        all_data = []
        start = 0
        for i, ll in enumerate(selected["location"]):
            for j, ff in enumerate(selected["function"]):
                current_data = data[ll][ff]
                length = len(current_data)
                if length > 0:
                    index_records[ll][ff].extend(list(np.arange(start, start+length)))
                else:
                    index_records[ll][ff].append(None)
                start += length
                if self.args.data_type == "function":
                    inference_record[j].extend(current_data)
                all_data += current_data

        if self.cls_num is None:
            return all_data, index_records
        else:
            inference_list = []
            # clsss = np.random.permutation(len(selected[self.args.data_type]))[:self.cls_num]
            clsss = np.arange(self.cls_num)
            for id in clsss:
                current_ff = inference_record[id]
                sl = np.random.permutation(len(current_ff))[:50]
                choice = []
                for pp in sl:
                    choice.append(current_ff[pp])
                    print(current_ff[pp])
                inference_list.extend(choice)
            class_name = []
            for cc in clsss:
                class_name.append(selected[self.args.data_type][cc])
            return inference_list, class_name

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
    with open(os.path.join("config/train.json"), "r") as load_train:
        train_data = json.load(load_train)
        train_count = statistic(train_data)

    with open(os.path.join("config/valid.json"), "r") as load_val:
        val_data = json.load(load_val)
        val_count = statistic(val_data)

    with open(os.path.join("config/test.json"), "r") as load_test:
        test_data = json.load(load_test)
        test_count = statistic(test_data)

    with open(os.path.join("config/category.json"), "r") as load_category:
        category = json.load(load_category)

    return {"train": train_data, "val": val_data, "test": test_data}, category, {"train": train_count, "val": val_count, "test": test_count}


def prepare_dataloaders(args, sample=None):
    print("start load data")
    total_data, index, count = get_minpaku_data()
    dataset_train = MinpakuGenetator(args, total_data["train"], index, transform=make_transform(args, "train"))
    dataset_val = MinpakuGenetator(args, total_data["val"], index, transform=make_transform(args, "val"))
    dataset_test = MinpakuGenetator(args, total_data["test"], index, transform=make_transform(args, "inference"))
    if sample["train"] is not None:
        sampler_train = CategoriesSampler(args, count["train"], selected, dataset_train.index_records, *sample["train"])
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)

    if sample["val"] is not None:
        sampler_val = CategoriesSampler(args, count["val"], selected, dataset_val.index_records, *sample["val"])
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_val = torch.utils.data.BatchSampler(sampler_val, args.batch_size, drop_last=False)

    if sample["test"] is not None:
        sampler_test = CategoriesSampler(args, count["test"], selected, dataset_test.index_records, *sample["test"])
    else:
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)
        sampler_test = torch.utils.data.BatchSampler(sampler_test, args.batch_size, drop_last=False)

    data_loader_train = DataLoader(dataset_train, batch_sampler=sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, batch_sampler=sampler_val, num_workers=args.num_workers)
    data_loader_test = DataLoader(dataset_test, batch_sampler=sampler_test, num_workers=args.num_workers)
    print("load data over")
    return {"train": data_loader_train, "val": data_loader_val, "test": data_loader_test}, index, {"location": len(index["location"]), "function": len(index["function"])}


def statistic(data):
    a = len(selected["location"])
    b = len(selected["function"])
    count = np.zeros((a, b))
    for ll in selected["location"]:
        for ff in selected["function"]:
            for current in data[ll][ff]:
                x = current["location"][0][:2]
                y = current["function"][0][:2]
                count[selected["location"].index(x)][selected["function"].index(y)] += 1
    # print("co-occurrence matrix")
    # print(count)
    # make_csv(count, "statistic")
    return count


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
    sampler = {"train": [100, 15, 20], "val": [300, 15, 20], "test": [300, 15, 20]}
    loaders, category, num_classes = prepare_dataloaders(args, sampler)
    for i_batch, sample_batch in enumerate(tqdm(loaders["val"])):
        print(sample_batch["image"].size())
        break





