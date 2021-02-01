import argparse
import torch
from model.extractor import MainModel
from loaders.load_data import get_minpaku_data, MinpakuGenetator
import tools.prepare_things as prt
from tools.draw_tools import show_single, plot_embedding
from loaders.transform_func import make_transform
from args import get_args_parser
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Dataset
import numpy as np
from sklearn.manifold import TSNE


@torch.no_grad()
def main(args):
    prt.init_distributed_mode(args)
    device = torch.device(args.device)
    total_data, index, count = get_minpaku_data()
    num_classes = {"location": len(index["location"]), "function": len(index["function"])}
    dataset = MinpakuGenetator(args, total_data["val"], index, transform=make_transform(args, "val"), cls_num=10)
    category = dataset.index_records
    sampler = torch.utils.data.SequentialSampler(dataset)
    sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=False)
    loaders = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers)

    model = MainModel(args, num_classes)
    model.to(device)
    checkpoint = torch.load(f"{args.output_dir}/" + f"{args.data_type}_checkpoint.pth", map_location=args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print("load pre-model " + f"{args.data_type}_checkpoint.pth" + " ready")

    feature_record = []
    img_record = []
    cat_record = []
    for i_batch, sample_batch in enumerate(tqdm(loaders)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label_" + args.data_type].to(device, dtype=torch.int64)
        label_name = sample_batch["path"]
        catt = sample_batch["function_name"]
        logits, feature = model(inputs)
        feature_record.append(feature.cpu().detach().numpy())
        img_record = img_record + label_name
        cat_record = cat_record + catt
    feature_record = np.concatenate(feature_record, axis=0)

    print(np.shape(feature_record))
    print("start t-SNE embedding")
    ts = TSNE(n_components=2, init="pca", random_state=0)
    results = ts.fit_transform(feature_record)
    plot_embedding(results, cat_record, category, args.data_type)

    ## ranking
    # repeat_record = []
    # target = feature_record[7000]
    # pair = ((target[None, :] - feature_record) ** 2).mean(1)
    # indices = pair.argsort()[:11]
    # print(indices)
    # for i in range(len(list(indices))):
    #     show_single(img_record[indices[i]], cat_record[indices[i]], i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    args.use_index = 2
    main(args)