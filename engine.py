import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm
from model.extractor import pair_loss


def train_one_epoch(args, model, data_loader, device, record, epoch, criterion, optimizer):
    model.train()
    calculation(args, model, "train", data_loader, device, record, epoch, criterion, optimizer)


@torch.no_grad()
def evaluate(args, model, data_loader, device, record, epoch, criterion):
    model.eval()
    calculation(args, model, "val", data_loader, device, record, epoch, criterion)


def calculation(args, model, mode, data_loader, device, record, epoch, criterion, optimizer=None):
    L = len(data_loader)
    running_loss = 0.0
    running_p_loss = 0.0
    running_corrects_1 = 0.0
    running_corrects_5 = 0.0
    print("start " + mode + " :" + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        label_name = sample_batch["label_name"]

        if mode == "train":
            optimizer.zero_grad()
        logits, feature = model(inputs)
        loss = criterion(logits, labels)
        if args.distance_loss:
            p_loss = pair_loss(args, feature, label_name)
            running_p_loss += p_loss.item()
            loss = loss + p_loss
        if mode == "train":
            loss.backward()
            optimizer.step()

        a = loss.item()
        running_loss += a
        running_corrects_1 += cal.evaluateTop1(logits, labels)
        running_corrects_5 += cal.evaluateTop5(logits, labels)
    epoch_loss = round(running_loss/L, 3)
    pp_loss = round(running_p_loss/L, 3)
    epoch_acc_1 = round(running_corrects_1/L, 3)
    epoch_acc_5 = round(running_corrects_5/L, 3)
    record[mode]["loss"].append(epoch_loss)
    record[mode]["p_loss"].append(pp_loss)
    record[mode]["acc_1"].append(epoch_acc_1)
    record[mode]["acc_5"].append(epoch_acc_5)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

