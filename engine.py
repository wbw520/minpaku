import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm


def train_one_epoch(args, model, data_loader, device, record, epoch, criterion, optimizer):
    model.train()
    calculation(args, model, "train", data_loader, device, record, epoch, criterion, optimizer)


@torch.no_grad()
def evaluate(args, model, data_loader, device, record, epoch, criterion):
    model.eval()
    calculation(args, model, "val", data_loader, device, record, epoch, criterion)


def calculation(args, model, mode, data_loader, device, record, epoch, criterion, optimizer=None):
    L = len(data_loader)
    running_loss = {"location": 0.0, "function": 0.0}
    running_corrects_1 = {"location": 0.0, "function": 0.0}
    running_corrects_5 = {"location": 0.0, "function": 0.0}
    print("start " + mode + " :" + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        if mode == "train":
            optimizer.zero_grad()

        if args.data_type == "union":
            inputs = sample_batch["image"].to(device, dtype=torch.float32)
            labels_location = sample_batch["label_location"].to(device, dtype=torch.int64)
            labels_function = sample_batch["label_function"].to(device, dtype=torch.int64)
            # ff_name = sample_batch["function_name"]
            # ll_name = sample_batch["location_name"]
            # print(ff_name)
            # print(ll_name)
            logits, feature = model(inputs)
            if args.triplet:
                loss_location = criterion(feature, labels_location)
                loss_function = criterion(feature, labels_function)
            else:
                loss_location = criterion(logits["location"], labels_location)
                loss_function = criterion(logits["function"], labels_function)
            loss = loss_location + loss_function
            running_loss["location"] += loss_location.item()
            running_loss["function"] += loss_function.item()
            if not args.triplet:
                running_corrects_1["location"] += cal.evaluateTop1(logits["location"], labels_location)
                running_corrects_5["location"] += cal.evaluateTop5(logits["location"], labels_location)
                running_corrects_1["function"] += cal.evaluateTop1(logits["function"], labels_function)
                running_corrects_5["function"] += cal.evaluateTop5(logits["function"], labels_function)
        else:
            inputs = sample_batch["image"].to(device, dtype=torch.float32)
            labels = sample_batch["label_"+args.data_type].to(device, dtype=torch.int64)  # _"+args.data_type
            # labels_name = sample_batch[args.data_type+"_name"]
            logits, feature = model(inputs)
            if args.triplet:
                loss = criterion(feature, labels)
            else:
                loss = criterion(logits[args.data_type], labels)
            running_loss[args.data_type] += loss.item()
            if not args.triplet:
                running_corrects_1[args.data_type] += cal.evaluateTop1(logits[args.data_type], labels)
                running_corrects_5[args.data_type] += cal.evaluateTop5(logits[args.data_type], labels)
        if mode == "train":
            loss.backward()
            optimizer.step()

    if args.data_type == "union":
        location_epoch_loss = round(running_loss["location"]/L, 3)
        location_epoch_acc_1 = round(running_corrects_1["location"]/L, 3)
        location_epoch_acc_5 = round(running_corrects_5["location"]/L, 3)
        function_epoch_loss = round(running_loss["function"]/L, 3)
        function_epoch_acc_1 = round(running_corrects_1["function"]/L, 3)
        function_epoch_acc_5 = round(running_corrects_5["function"]/L, 3)
        record[mode]["location"]["loss"].append(location_epoch_loss)
        record[mode]["location"]["acc_1"].append(location_epoch_acc_1)
        record[mode]["location"]["acc_5"].append(location_epoch_acc_5)
        record[mode]["function"]["loss"].append(function_epoch_loss)
        record[mode]["function"]["acc_1"].append(function_epoch_acc_1)
        record[mode]["function"]["acc_5"].append(function_epoch_acc_5)
    else:
        epoch_loss = round(running_loss[args.data_type]/L, 3)
        epoch_acc_1 = round(running_corrects_1[args.data_type]/L, 3)
        epoch_acc_5 = round(running_corrects_5[args.data_type]/L, 3)
        record[mode][args.data_type]["loss"].append(epoch_loss)
        record[mode][args.data_type]["acc_1"].append(epoch_acc_1)
        record[mode][args.data_type]["acc_5"].append(epoch_acc_5)


# def clip_gradient(optimizer, grad_clip):
#     """
#     Clips gradients computed during backpropagation to avoid explosion of gradients.
#
#     :param optimizer: optimizer with the gradients to be clipped
#     :param grad_clip: clip value
#     """
#     for group in optimizer.param_groups:
#         for param in group["params"]:
#             if param.grad is not None:
#                 param.grad.data.clamp_(-grad_clip, grad_clip)

