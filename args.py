import argparse


def get_args_parser():
    parser = argparse.ArgumentParser('Minpaku Project', add_help=False)

    # training set
    parser.add_argument('--base_model', default="resnet18", type=str)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--lr_drop', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--weight_decay', default=0.0001, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--img_size', default=224, help='path for save data')
    parser.add_argument('--aug', default=False, help='whether use augmentation')
    parser.add_argument('--pre_trained', default=True, type=bool, help='whether use pre parameter for backbone')

    # data/machine set
    parser.add_argument('--json_root', default="/home/wangbowen/data/minpaku", type=str)
    parser.add_argument('--dataset_dir', default='/home/wangbowen/data/minpaku',
                        help='path for save data')
    parser.add_argument('--data_type', default='function', help='type for data')
    parser.add_argument('--use_label_num', default=-2, help='the top k class for multi-label')
    parser.add_argument('--use_index', default=3, help='top first k index to use')
    parser.add_argument('--pre_dir', default='pre_model/',
                        help='path of pre-train model')
    parser.add_argument('--max_demo', default=20, help='max category for demonstration')
    parser.add_argument('--use_pre', default=False, help='finetune')
    parser.add_argument('--triplet', default=True, help='whether use triplet loss')
    parser.add_argument('--extract', default=True, help='whether only use embedding')
    parser.add_argument('--fix', default=False, help='whether fix feature extractor')
    parser.add_argument('--save_mode', default="loss", help='how to save model')
    parser.add_argument('--device', default='cuda:1',
                        help='device to use for training / testing')
    parser.add_argument('--output_dir', default='saved_model',
                        help='path where to save, empty for no saving')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=4, type=int,
                        help='number of distributed processes')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser