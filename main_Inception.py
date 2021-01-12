import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from tqdm import tqdm

from Utils.util import load_data, separate_data
from models.graphcnn_Inception import Graph_Inception
from Utils.pytorchtools import EarlyStopping

criterion = nn.CrossEntropyLoss()

def train(args, model, device, train_graphs, optimizer, epoch, A):
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:
        selected_idx = np.random.permutation(len(train_graphs))[:args.batch_size]

        batch_graph = [train_graphs[idx] for idx in selected_idx]
        output = model(batch_graph, A)

        labels = torch.LongTensor([graph.label for graph in batch_graph]).to(device)

        # compute loss
        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss = loss.detach().cpu().numpy()
        loss_accum += loss

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


###pass data to model with minibatch during testing to avoid memory overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, A, minibatch_size = 64):
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i+minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx], A).detach())
    return torch.cat(output, 0)

def test(args, model, device, train_graphs, test_graphs, epoch, num_class, A):
    model.eval()

    output = pass_data_iteratively(model, train_graphs, A)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in train_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    output = pass_data_iteratively(model, test_graphs, A)
    pred = output.max(1, keepdim=True)[1]
    labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
    correct = pred.eq(labels.view_as(pred)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test

def main():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results reported in the paper.
    parser = argparse.ArgumentParser(description='PyTorch graph convolutional neural net for whole-graph classification')
    parser.add_argument('--dataset', type=str, default="Mine_Graph_RML",
                        help='name of dataset (default: RML)')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--iters_per_epoch', type=int, default=90,
                        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed for splitting the dataset into 5 (default: 0)')
    parser.add_argument('--fold_idx', type=int, default=0,
                        help='the index of fold in 5-fold validation. Should be less then 5.')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument('--final_dropout', type=float, default=0.7,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument('--degree_as_tag', action="store_true",
                        help='let the input node features be the degree of nodes (heuristics for unlabeled graph)')
    parser.add_argument('--Normalize', type=bool, default=False, choices=[True, False],
                        help='normalizing data')
    parser.add_argument('--patience', type=int, default=40,
                        help='patience for early stopping')
    args = parser.parse_args()

    #set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    graphs, num_classes = load_data(args.dataset, args.degree_as_tag, args.Normalize)


    ##10-fold cross validation. Conduct an experiment on the fold specified by args.fold_idx.
    train_graphs, test_graphs = separate_data(graphs, args.seed, args.fold_idx)

    model = Graph_Inception(args.num_layers, train_graphs[0].node_features.shape[1],
                     num_classes, args.final_dropout,
                     device, args.dataset, args.batch_size).to(device)

    Num_Param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Trainable Parameters= %d" % (Num_Param))

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    A = np.zeros([90, 90])
    Num_hop = 1
    for i in range(A.shape[0]):
        # A[i, i] = 1
        for j in range(A.shape[0]):
            if (i - j <= Num_hop) and (i - j > 0):
                A[i, j] = 1
                A[j, i] = 1
    A = torch.FloatTensor(A).to(device)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(args, model, device, train_graphs, optimizer, epoch, A)

        if(epoch>350):
            #### Validation check
            with torch.no_grad():
                val_out = pass_data_iteratively(model, test_graphs, A)
                val_labels = torch.LongTensor([graph.label for graph in test_graphs]).to(device)
                val_loss = criterion(val_out, val_labels)
                val_loss = np.average(val_loss.detach().cpu().numpy())

            #### Check early stopping
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break


        if(epoch % 50 ==0):
            acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch, num_classes, A)


    model.load_state_dict(torch.load('./Saved_models/checkpoint.pt'))
    acc_train, acc_test = test(args, model, device, train_graphs, test_graphs, epoch, num_classes, A)
    

if __name__ == '__main__':
    main()
