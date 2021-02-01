import numpy as np
from torch.utils.data import Sampler
import torch
import math

__all__ = ['CategoriesSampler']


class CategoriesSampler(Sampler):
    def __init__(self, args, count, cat, index_record, n_iter, n_location, n_function, seed=None):
        self.n_iter = n_iter
        self.n_location = n_location
        self.n_function = n_function
        self.seed = seed
        self.index_record = index_record
        self.cat = cat
        self.args = args
        self.count = count
        self.location_weight = np.log2(self.count.sum(1))
        self.function_weight = np.log2(self.count.sum(0))
        self.function_new_record = self.concat()

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for i in range(self.n_iter):
            # if self.seed is not None:
            #     torch.manual_seed(self.seed)
            #     torch.cuda.manual_seed(self.seed)
            if self.args.data_type == "union":
                yield self.union_sampler()
            else:
                yield self.single_sampler()

    def union_sampler(self):
        selected_index = []
        selected_location = np.random.permutation(len(self.cat["location"]))[:self.n_location]
        selected_function = np.random.permutation(len(self.cat["function"]))[:self.n_function]
        for ll in selected_location:
            for ff in selected_function:
                record = self.index_record[self.cat["location"][ll]][self.cat["function"][ff]]
                if record[0] is not None:
                    low, high = record
                else:
                    continue
                if low == high:
                    current_index = low
                    selected_index.append(current_index)
                else:
                    current_index = torch.randint(low, high, (1,))
                    selected_index.append(current_index[0])

    def single_sampler(self):
        selected_index = []
        selected = np.random.permutation(len(self.cat[self.args.data_type]))[:self.n_function]
        selected_weight = self.function_weight[selected]
        normed = (self.n_function * self.n_location) / selected_weight.sum()
        for ww in selected:
            num = math.ceil(normed * self.function_weight[ww])
            ccc = np.random.choice(self.function_new_record[self.cat["function"][ww]], num, replace=False)
            selected_index.extend(ccc)
        return selected_index

    def concat(self):
        new = {}
        for ff in self.cat["function"]:
            current = []
            for ll in self.cat["location"]:
                if self.index_record[ll][ff][0] is not None:
                    current.extend(self.index_record[ll][ff])
            new.update({ff: current})
        return new
