import argparse
from datetime import datetime

import torch
import torchvision.models
from prettytable import PrettyTable
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from CSCI6366_GWU.CatSoundClassification.DataSets.dataload import get_dataloader
from CSCI6366_GWU.CatSoundClassification.DataSets.txt2classlist import trans
from log_generator import log_generator
from display import createConfMatrix, createLinearFig
# from CatSoundClassification.Model.SimpleNet import AudioClassificationModel


# ----------------------------------------------------------------------------------------------------------------------
# Load argument
def get_args():
    parser = argparse.ArgumentParser(description='Audio classification parameter configuration(train)')
    parser.add_argument(
        '-t',
        type=str,
        default='CatSoundClassification',
        help="the theme's name of your task"
    )
    parser.add_argument(
        '-dp',
        type=str,
        default=r'../Data/scatter/npy_data',
        help="train's directory"
    )
    parser.add_argument(
        '-classes',
        type=list,
        default=trans(r'../Data/scatter/classes.txt'),
        help="classes list"
    )
    parser.add_argument(
        '-infop',
        type=str,
        default=r'../Data/scatter/refer.csv',
        help="DIF(folder information file)'s path"
    )

    parser.add_argument(
        '-tp',
        type=float,
        default=0.9,
        help="train folder's percent of entire datasets"
    )
    parser.add_argument(
        '-bs',
        type=int,
        default=16,
        help="folder's batch size"
    )
    parser.add_argument(
        '-cn',
        type=int,
        default=10,
        help='the number of classes'
    )
    parser.add_argument(
        '-e',
        type=int,
        default=80,
        help='epoch'
    )
    parser.add_argument(
        '-lr',
        type=float,
        default=0.001,
        help='learning rate'
    )
    parser.add_argument(
        '-ld',
        type=str,
        default='../WorkDir',
        help="the training log's save directory"
    )

    return parser.parse_args()


def main():
    args = get_args()  # Obtain parameter Namespace
# ----------------------------------------------------------------------------------------------------------------------
    print("Training device information:")
    # Training equipment information
    device_table = ""
    if torch.cuda.is_available():
        device_table = PrettyTable(['number of gpu', 'applied gpu index', 'applied gpu name'], min_table_width=80)
        gpu_num = torch.cuda.device_count()
        gpu_index = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name()
        device_table.add_row([str(gpu_num), str(gpu_index), str(gpu_name)])
        print('{}\n'.format(device_table))
    else:
        print("Using cpu......")
        device_table = 'CPU'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# ----------------------------------------------------------------------------------------------------------------------
# Dataset information
    print("Use folder information file:{}\nLoading folder from path: {}......".format(args.infop, args.dp))
    train_dl, valid_dl, samples_num, train_num, valid_num = get_dataloader(args.infop, args.dp, args.bs, args.tp)
    dataset_table = PrettyTable(['number of samples', 'train number', 'valid number', 'percent'], min_table_width=80)
    dataset_table.add_row([samples_num, train_num, valid_num, args.tp])
    print("{}\n".format(dataset_table))
# ----------------------------------------------------------------------------------------------------------------------
# Training component configuration
    print("Classes information:")
    classes_table = PrettyTable(args.classes, min_table_width=80)
    classes_table.add_row(range(len(args.classes)))
    print("{}\n".format(classes_table))
    print("Train information:")
    # model = AudioClassificationModel(num_classes=args.cn).to(device)
    # Use ResNet 18
    model = torchvision.models.resnet18(pretrained=True)
    # Adjust the number of output features for the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, args.cn)
    # Ensure the model is moved to the appropriate device
    model = model.to(device)
    # Select the parameters to update
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    # optimizer = Adam(params=model.parameters(), lr=args.lr)
    optimizer = Adam(params=params_to_update, lr=args.lr)
    loss_fn = CrossEntropyLoss()
    train_table = PrettyTable(['theme', 'batch size', 'epoch', 'learning rate', 'directory of log'],
                              min_table_width=120)
    train_table.add_row([args.t, args.bs, args.e, args.lr, args.ld])
    print('{}\n'.format(train_table))
# ----------------------------------------------------------------------------------------------------------------------
# Start training
    losses = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    aucs = []
    maps = []
    history_rates = [[], [], []]
    best_checkpoint = None

    st = datetime.now()
    for epoch in range(args.e):

        prediction = []
        label = []
        score = []

        model.train()
        train_bar = tqdm(iter(train_dl), ncols=150, colour='red')
        train_loss = 0.
        i = 0
        for train_data in train_bar:
            x_train, y_train = train_data
            x_train = x_train.to(device)
            # Replicate the single channel to create three channels
            x_train = x_train.repeat(1, 3, 1, 1)
            y_train = y_train.to(device).long()
            output = model(x_train)
            loss = loss_fn(output, y_train)
            optimizer.zero_grad()
            # clone().detach()：It is possible to copy only one tensor value without affecting the tensor
            train_loss += loss.clone().detach().cpu().numpy()
            loss.backward()
            optimizer.step()
            # Display the loss of each epoch
            train_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(train_dl)))
            train_bar.set_postfix({"train loss": "%.3f" % loss.data})
            i += 1
        train_loss = train_loss / i
        # The final i obtained is the number of sample batches in one iteration
        losses.append(train_loss)

        model.eval()
        valid_bar = tqdm(iter(valid_dl), ncols=150, colour='red')
        valid_acc = 0.
        valid_pre = 0.
        # valid_recall = 0.
        valid_f1 = 0.
        valid_auc = 0.
        valid_ap = 0.
        i = 0
        for valid_data in valid_bar:
            x_valid, y_valid = valid_data
            x_valid = x_valid.to(device)
            x_valid = x_valid.repeat(1, 3, 1, 1)
            y_valid_ = y_valid.clone().detach().numpy().tolist()
            output = model(x_valid)  # shape: ( N * cls_n )
            output_ = output.clone().detach().cpu()
            _, pred = torch.max(output_, 1)  # Output the index of the maximum probability for each row (sample)
            pred_ = pred.clone().detach().numpy().tolist()
            output_ = output_.numpy().tolist()
            # acc/precision/recall/f1 of each epoch
            valid_bar.set_description("Epoch:{}/{} Step:{}/{}".format(epoch + 1, args.e, i + 1, len(valid_dl)))
            prediction = prediction + pred_
            label = label + y_valid_
            score = score + output_
            i += 1
        # The final i obtained is the batch size of samples in one iteration
        # Calculate once per epoch

        valid_acc = accuracy_score(y_true=label, y_pred=prediction)
        valid_pre = precision_score(y_true=label, y_pred=prediction, average='weighted')
        # valid_recall = recall_score(y_true=label, y_pred=prediction, average='weighted')
        valid_f1 = f1_score(y_true=label, y_pred=prediction, average='weighted')
        # valid_auc = roc_auc_score(y_true=label, y_score=score, average='weighted', multi_class="ovr")
        # valid_ap = average_precision_score(y_true=label, y_score=score)

        createConfMatrix(prediction, label, count=epoch)

        accuracies.append(valid_acc)
        precisions.append(valid_pre)
        # recalls.append(valid_recall)
        f1s.append(valid_f1)
        # aucs.append(valid_auc)
        # maps.append(valid_ap)

        # if valid_f1 >= max(f1s):
        #     # If the f1 of this epoch is greater than the maximum value stored in the f1 list
        #     # then the best_checkpoint is model
        #     best_checkpoint = model

        # Check if current model has the best F1 score
        if best_checkpoint is None or valid_f1 > best_checkpoint['f1']:
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': valid_f1
            }

        # Save model after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'f1': valid_f1
        }, f"../ModelSave/model_epoch_{epoch}.pth")

        # indicator_table = PrettyTable(['Accuracy', 'Precision', 'Recall', 'F1'])
        # indicator_table.add_row([valid_acc, valid_pre, valid_recall, valid_f1])
        indicator_table = PrettyTable(['Accuracy', 'Precision', 'F1'])
        indicator_table.add_row([valid_acc, valid_pre, valid_f1])

        # Record the rates
        history_rates[0].append(valid_acc)
        history_rates[1].append(valid_pre)
        # history_rates[2].append(valid_recall)
        history_rates[2].append(valid_f1)
        print(f"history:  {history_rates}")

        print('\n{}\n'.format(indicator_table))
    et = datetime.now()

    log_generator(args.t, args.infop, args.classes, x_train.shape, et-st, dataset_table, classes_table, device_table,
                  train_table, optimizer, model, args.e, [losses, accuracies, precisions, recalls, f1s],
                  args.ld,best_checkpoint)

    createLinearFig(history_rates)
    # createConfMatrix(label, prediction)

if __name__ == "__main__":
    main()