import json
import os
from tqdm.auto import tqdm
import argparse
from args import get_args_parser


class Minpaku():
    def __init__(self, args):
        self.args = args
        self.json_root = args.json_root
        self.image_root = args.dataset_dir
        self.use_index = args.use_index
        self.use_label_num = args.use_label_num
        self.cat_dict_function = {}
        self.cat_dict_location = {}
        self.no_label = {"owc": 0, "ocm": 0}
        self.no_image = {"owc": 0, "ocm": 0}
        self.not_in = {"location": 0, "function": 0}
        self.category = {}

    def get_record(self, file_root, phase):
        data = []
        with open(os.path.join(file_root, "minpaku-"+phase+".json"), "r") as load_f:
            all_data = json.load(load_f)
            for iterm in tqdm(all_data):
                current_data = {"id": iterm["id"], "path": iterm["path"]}
                condition = False
                if self.args.data_type != "function":
                    location_cat = self.filter(iterm, "owc", self.cat_dict_location)
                    current_data.update({"location": location_cat})
                    if location_cat != "empty":
                        condition = True
                        if phase != "train" and location_cat[0][:2] not in self.category["location"]:
                            self.not_in["location"] += 1
                            continue
                if self.args.data_type != "location":
                    function_cat = self.filter(iterm, "ocm", self.cat_dict_function)
                    current_data.update({"function": function_cat})
                    if function_cat != "empty":
                        condition = True
                        if phase != "train" and function_cat[0][:2] not in self.category["function"]:
                            self.not_in["function"] += 1
                            continue
                if condition:
                    data.append(current_data)

        print(f"{phase} length: ", len(data))
        print("location no image: ", self.no_image["owc"])
        print("location no label: ", self.no_label["owc"])
        print("function no image: ", self.no_image["ocm"])
        print("function no label: ", self.no_label["ocm"])

        if phase == "train":
            ll = list(self.convert_to_dict(sorted(self.cat_dict_location.items(), key=lambda d: d[1], reverse=True)).keys())
            self.category.update({"location": ll})
            ff = list(self.convert_to_dict(sorted(self.cat_dict_function.items(), key=lambda d: d[1], reverse=True)).keys())
            self.category.update({"function": ff})
            # with open(f"../config/category.json", "w") as c_f:
            #     json.dump(self.category, c_f)
        else:
            print("location not in: ", self.not_in["location"])
            print("function not in: ", self.not_in["function"])
            self.not_in = {"location": 0, "function": 0}

        # with open(f"../config/{phase}.json", "w") as d_f:
        #     json.dump(data, d_f)

        self.no_label = {"owc": 0, "ocm": 0}
        self.no_image = {"owc": 0, "ocm": 0}

    def filter(self, iterm, data_type_current, category):
        current_cat = iterm[data_type_current]
        if not os.path.exists(os.path.join(self.image_root, 'images', iterm["path"])):
            self.no_image[data_type_current] += 1
            return "empty"
        if len(current_cat) == 0:
            self.no_label[data_type_current] += 1
            return "empty"
        repeat_cat = ""
        for cat in current_cat[:self.use_label_num+1]:
            if repeat_cat != cat[:self.use_index]:
                repeat_cat = cat[:self.use_index]
                if cat[:self.use_index] not in category:
                    category.update({cat[:self.use_index]: 1})
                else:
                    category[cat[:self.use_index]] += 1
            else:
                continue
        return current_cat

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
        self.get_record(self.json_root, "train")
        self.get_record(self.json_root, "valid")
        self.get_record(self.json_root, "test")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    Minpaku(args).load()