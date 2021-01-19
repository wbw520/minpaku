import json
import os
from tqdm.auto import tqdm
import argparse
from args import get_args_parser


selected = {"location": ['AB', 'NU', 'AW', 'AD', 'AA', 'NW', 'AK', 'OJ', 'SE', 'AF', 'SO', 'AE', 'FA', 'OR', 'MS', 'AJ', 'OE', 'AO', 'FL', 'N4', 'AC', 'SF', 'AN', 'FO', 'NV', 'OA', 'NT', 'OF', 'ED', 'AH', 'OI', 'FF', 'EP', 'OC', 'AL', 'RX', 'AM', 'FE', 'SP', 'MW', 'SQ', 'MR', 'OD', 'ES', 'EW', 'MB', 'OV', 'EC', 'AP', 'FX'],
            "function": ['29', '41', '32', '53', '28', '22', '52', '30', '35', '27', '26', '25', '21', '38', '77', '23', '17', '78', '24', '39', '48', '79', '49', '37', '50', '76', '80', '51', '43', '58', '75', '33', '34', '31', '40', '13', '20']}


class Minpaku():
    def __init__(self, args):
        self.args = args
        self.json_root = args.json_root
        self.image_root = args.dataset_dir
        self.use_index = args.use_index
        self.use_label_num = args.use_label_num
        self.cat_dict = {"location": {}, "function": {}}
        self.no_label = {"owc": 0, "ocm": 0}
        self.no_image = 0
        self.not_in = {"location": 0, "function": 0}
        self.category = {}

    def construction(self):
        recorder = {}
        for location in selected["location"]:
            current = {}
            for function in selected["function"]:
                current.update({function: []})
            recorder.update({location: current})
        return recorder

    def get_record(self, file_root, phase):
        data_len = 0
        overall = self.construction()
        with open(os.path.join(file_root, "minpaku-"+phase+".json"), "r") as load_f:
            all_data = json.load(load_f)
            for iterm in tqdm(all_data):
                current_data = {"id": iterm["id"], "path": iterm["path"]}
                cat = self.filter(iterm, self.cat_dict)
                if cat != "empty":
                    current_data.update({"location": cat[0]})
                    current_data.update({"function": cat[1]})
                    ll = cat[0][0][:2]
                    ff = cat[1][0][:2]
                    if phase != "train" and ll not in self.category["location"]:
                        self.not_in["location"] += 1
                        continue
                    if phase != "train" and ff not in self.category["function"]:
                        self.not_in["function"] += 1
                        continue
                    if ll not in selected["location"] or ff not in selected["function"]:
                        continue
                    data_len += 1
                    overall[ll][ff].append(current_data)

        print(f"{phase} length: ", data_len)
        print("no image: ", self.no_image)
        print("location no label: ", self.no_label["owc"])
        print("function no label: ", self.no_label["ocm"])

        if phase == "train":
            ll = list(self.convert_to_dict(sorted(self.cat_dict["location"].items(), key=lambda d: d[1], reverse=True)).keys())
            # print(ll[:50])
            # print(sorted(self.cat_dict["location"].items(), key=lambda d: d[1], reverse=True))
            self.category.update({"location": ll})
            ff = list(self.convert_to_dict(sorted(self.cat_dict["function"].items(), key=lambda d: d[1], reverse=True)).keys())
            # print(ff[:40])
            # print(sorted(self.cat_dict["function"].items(), key=lambda d: d[1], reverse=True))
            self.category.update({"function": ff})
            with open(f"../config/category.json", "w") as c_f:
                json.dump(selected, c_f)
        else:
            print("location not in: ", self.not_in["location"])
            print("function not in: ", self.not_in["function"])
            self.not_in = {"location": 0, "function": 0}

        with open(f"../config/{phase}.json", "w") as d_f:
            json.dump(overall, d_f)

        self.no_label = {"owc": 0, "ocm": 0}
        self.no_image = 0

    def filter(self, iterm, category):
        if not os.path.exists(os.path.join(self.image_root, 'images', iterm["path"])):
            self.no_image += 1
            return "empty"
        current_cat_location = iterm["owc"]
        if len(current_cat_location) == 0:
            self.no_label["owc"] += 1
            return "empty"
        current_cat_function = iterm["ocm"]
        if len(current_cat_function) == 0:
            self.no_label["ocm"] += 1
            return "empty"
        self.deal_repeat(current_cat_location, "location", category)
        self.deal_repeat(current_cat_function, "function", category)
        return [current_cat_location, current_cat_function]

    def deal_repeat(self, current_cat, type, category):
        repeat_cat = ""
        for cat in current_cat[:self.use_label_num+1]:
            if repeat_cat != cat[:self.use_index]:
                repeat_cat = cat[:self.use_index]
                if cat[:self.use_index] not in category[type]:
                    category[type].update({cat[:self.use_index]: 1})
                else:
                    category[type][cat[:self.use_index]] += 1
            else:
                continue

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


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     Minpaku(args).load()