import argparse
import torch
from model.extractor import load_backbone
from loaders.load_data import prepare_dataloaders
import time
import datetime
import tools.prepare_things as prt
from pathlib import Path
from engine import train_one_epoch, evaluate
from tools.calculate_tool import MetricLog
from args import get_args_parser


def main(args):
    prt.init_distributed_mode(args)
    device = torch.device(args.device)
    model = load_backbone(args)
    model.to(device)
    if args.use_pre:
        checkpoint = torch.load(f"{args.output_dir}/" + f"{args.data_type}_pair_False_checkpoint.pth", map_location=args.device)
        model.load_state_dict(checkpoint["model"], strict=True)
        print("load pre-model " + f"{args.data_type}_pair_False" + " ready")

    loaders, category = prepare_dataloaders(args)
    model_without_ddp = model
    output_dir = Path(args.output_dir)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    params = [p for p in model_without_ddp.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    print("Start training")
    start_time = time.time()
    log = MetricLog()
    record = log.record

    max_acc1 = 0
    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(args, model, loaders["train"], device, record, epoch, criterion, optimizer)
        lr_scheduler.step()
        evaluate(args, model, loaders["val"], device, record, epoch, criterion)

        if args.output_dir:
            checkpoint_paths = [output_dir / f"{args.data_type}_pair_{args.distance_loss}_checkpoint.pth"]
            if record["val"]["acc_1"][epoch-1] > max_acc1:
                print("get higher acc save current model")
                max_acc1 = record["val"]["acc_1"][epoch]
                for checkpoint_path in checkpoint_paths:
                    prt.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)
        log.print_metric()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)