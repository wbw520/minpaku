import json
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from prefetch_generator import BackgroundGenerator
from loaders.transform_func import make_transform
from tqdm.auto import tqdm
from torch.utils.data import DistributedSampler
from tools.draw_tools import make_distribution
# import argparse
# from args import get_args_parser
# import tools.prepare_things as prt


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


class Minpaku():
    def __init__(self, args):
        self.data_type = {"location": "owc", "function": "ocm"}
        self.data_type_current = self.data_type[args.data_type]
        self.json_root = args.json_root
        self.image_root = args.dataset_dir
        self.use_index = args.use_index
        self.use_label_num = args.use_label_num
        self.cat_dict = {}
        self.multi_cat = []

    def get_record(self, file_root, phase):
        self.cat_dict = {}
        self.multi_cat = []
        current_data = []
        no_label = 0
        no_image = 0
        with open(os.path.join(file_root, "minpaku-"+phase+".json"), "r") as load_f:
            all_data = json.load(load_f)
            for iterm in tqdm(all_data):
                current_cat = iterm[self.data_type_current]
                if len(current_cat) == 0:
                    no_label += 1
                    continue
                if not os.path.exists(os.path.join(self.image_root, 'images', iterm["path"])):
                    no_image += 1
                    continue
                current_data.append([iterm["id"], current_cat, iterm["path"]])
                repeat_cat = ""
                for cat in current_cat[:self.use_label_num+1]:
                    if repeat_cat != cat[:self.use_index]:
                        repeat_cat = cat[:self.use_index]
                        if cat[:self.use_index] not in self.cat_dict:
                            self.cat_dict.update({cat[:self.use_index]: 1})
                        else:
                            self.cat_dict[cat[:self.use_index]] += 1
                    else:
                        continue
        print(len(current_data))
        print("no label: ", no_label)
        print("no image: ", no_image)
        return current_data, self.convert_to_dict(sorted(self.cat_dict.items(), key=lambda d: d[1], reverse=True))

    # def statistic(self):
    #     train_data, train_category = self.get_record(self.json_root, "train")
    #     make_distribution(train_category, args, "train")
    #     print(len(train_category))
    #     val_data, val_category = self.get_record(self.json_root, "valid")
    #     print(len(val_category))
    #     test_data, test_category = self.get_record(self.json_root, "test")
    #     print(len(test_category))

    def convert_to_dict(self, in_data):
        D = {}
        for data in in_data:
            D.update({data[0]: data[1]})
        return D

    def cal_aug_weight(self, in_data):
        weight = []
        for v, k in in_data.items():
            weight.append(round(1/(k/5000), 3))
        return weight

    def load(self):
        test_data, test_category = self.get_record(self.json_root, "test")
        train_data, train_category = self.get_record(self.json_root, "train")
        val_data, val_category = self.get_record(self.json_root, "valid")
        aug_weight_ = self.cal_aug_weight(train_category)
        print("train category: ", len(train_category))
        print(train_category)
        return {"train": train_data, "val": val_data, "test": test_data}, list(train_category.keys()), aug_weight_


class MinpakuGenetator(Dataset):
    def __init__(self, args, data, class_index, transform=None):
        super(MinpakuGenetator, self).__init__()
        self.use_index = args.use_index
        self.use_label_num = args.use_label_num
        self.image_root = args.dataset_dir
        self.data = data
        self.class_index = class_index
        self.transform = transform

    def deal_label(self, input):
        return self.class_index.index(input)

    def discriminate(self, index):
        ID, type, img_dir = self.data[index]
        if type[self.use_label_num][:self.use_index] in self.class_index:
            return False
        else:
            print("class " + str(type[self.use_label_num][:self.use_index]) + " not exist in train class")
            return True

    def __getitem__(self, index):
        while self.discriminate(index):
            index += 1
        ID, type, img_dir = self.data[index]
        image_path = os.path.join(self.image_root, 'images', img_dir)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = self.deal_label(type[self.use_label_num][:self.use_index])
        label = torch.from_numpy(np.array(label))
        if self.transform is not None:
            img = self.transform(img, 2)
        return {"image": img, "label": label, "name": image_path, "label_name": type[self.use_label_num][:self.use_index]}

    def __len__(self):
        return len(self.data)


def prepare_dataloaders(args):
    print("start load data")
    total_data, index, aug_weight = Minpaku(args).load()
    print(index)
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

    data_loader_train = DataLoaderX(dataset_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    data_loader_val = DataLoaderX(dataset_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)
    data_loader_test = DataLoaderX(dataset_test, args.batch_size, sampler=sampler_test, num_workers=args.num_workers)
    print("load data over")
    return {"train": data_loader_train, "val": data_loader_val, "test": data_loader_test}, index


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     prt.init_distributed_mode(args)
#     loaders, category = prepare_dataloaders(args)
#     for i_batch, sample_batch in enumerate(tqdm(loaders["train"])):
#         print(sample_batch["image"].size())
#         print(sample_batch["label"])
#         print(sample_batch["label_name"])





