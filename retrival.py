import argparse
import torch
from model.extractor import load_backbone
from loaders.load_data import prepare_dataloaders
import tools.prepare_things as prt
from tools.draw_tools import show_single, plot_embedding
from args import get_args_parser
from tqdm.auto import tqdm
import numpy as np
from sklearn.manifold import TSNE


@torch.no_grad()
def main(args):
    prt.init_distributed_mode(args)
    device = torch.device(args.device)
    model = load_backbone(args)
    model.to(device)
    checkpoint = torch.load(f"{args.output_dir}/" + f"{args.data_type}_pair_True_checkpoint.pth", map_location=args.device)
    model.load_state_dict(checkpoint["model"], strict=True)
    print("load pre-model " + f"{args.data_type}_pair_True" + " ready")

    loaders, category = prepare_dataloaders(args)
    feature_record = []
    img_record = []
    cat_record = []
    for i_batch, sample_batch in enumerate(tqdm(loaders["test"])):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        label_name = sample_batch["name"]
        catt = sample_batch["label_full"]
        logits, feature = model(inputs)
        feature_record.append(feature.cpu().detach().numpy())
        img_record = img_record + label_name
        cat_record = cat_record + catt
    feature_record = np.concatenate(feature_record, axis=0)

    feature_record = feature_record[0:-1:20, :]
    cat_record = cat_record[0:-1:20]
    print(np.shape(feature_record))
    print("start t-SNE embedding")
    ts = TSNE(n_components=2, init="pca", random_state=0)
    results = ts.fit_transform(feature_record)
    plot_embedding(results, cat_record, category[:args.max_demo], args.data_type)

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