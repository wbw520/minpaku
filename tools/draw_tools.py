import matplotlib.pyplot as plt


def make_distribution(data, args, phase):
    plt.figure(figsize=(25, 5))
    plt.bar(range(len(data)), [k for v, k in data.items()])
    plt.xticks(range(len(data)), [v for v, k in data.items()], rotation=60, fontsize=12)
    plt.title(f"data distribution of {args.data_type}_{args.use_index}_index in phase {phase}", fontsize='20')
    plt.savefig(f"/home/wangbowen/data_{args.data_type}_{args.use_index}_index_{phase}.png", bbox_inches='tight')
    plt.show()