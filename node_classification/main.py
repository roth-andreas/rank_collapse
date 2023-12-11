import torch
import time
import torch_geometric as pyg
import argparse
from SKP import SKP
import time
import numpy as np
import data_handling
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.transforms import LargestConnectedComponents
import matplotlib
import matplotlib.pyplot as plt


class Model(torch.nn.Module):
    def __init__(self, in_dim, h_dim, out_dim, num_heads, conv, num_layers, act, num_edges):
        super(Model, self).__init__()
        h_dim = 2*h_dim if conv == 'KP' else h_dim
        self.enc = torch.nn.Linear(in_dim, h_dim)
        self.dec = torch.nn.Linear(h_dim, out_dim, bias=True)
        self.num_layers = num_layers
        self.act = act
        num_heads = 1 if conv == 'KP' else 2

        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvLayer(h_dim, conv, num_heads, num_edges))

    def forward(self, x, edge_index):
        x = torch.nn.functional.relu(self.enc(x))
        for i in range(self.num_layers):
            x = self.layers[i](x, edge_index)
            if self.act:
                x = torch.nn.functional.relu(x)

        return self.dec(x), x
    
def plot_losses(loss_dict, name):
    
    
    color_dict = {'SKP': '#73D055', 'softmax_SKP':'#440154','KP':'#1F968B'}
    
    plt.figure(figsize=(5,2.5))
    for label,losses in loss_dict.items():

        mean = np.mean(losses,axis=0)
        std = np.std(losses,axis=0)
        min_ = np.min(losses,axis=0)
        max_ = np.max(losses,axis=0)
        y = np.arange(0,mean.shape[0]*16-1,16)
        
        plt.plot(y, mean, color=color_dict[label], label=label)
        plt.fill_between(y, min_, max_, alpha=0.1, color=color_dict[label])

    plt.ylabel('Loss')
    plt.ylim([-0.1,1.05])
    plt.xlabel('Optimization steps')
    plt.legend()
    plt.savefig(f'./figures/losses_{name}.svg',bbox_inches='tight')
    plt.close()

class ConvLayer(torch.nn.Module):
    def __init__(self, d, conv, num_heads, num_edges):
        super(ConvLayer, self).__init__()
        self.conv1 = SKP(d, d, conv=conv, heads=num_heads, num_edges=num_edges)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('--convs', type=str, nargs='+', default=['SKP'])
    parser.add_argument('--datasets', default=['texas'], nargs='+')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--heads', type=int, default=2)
    parser.add_argument('--h_dim', type=int, default=16)
    parser.add_argument('--layers', default=[1,2,4,8,16,32], type=int, nargs='+')
    parser.add_argument('-l', action='store_true',
                help='Linear: Whether to use a linear message-passing model.')
    parser.add_argument('-v', action='store_true',
                help='Verbose: Whether to print out optimization updates.')
    args = parser.parse_args()
    
    verbose = args.v
    seeds = 10
    
    

    for dataset in args.datasets:
        epochs = 2000
        
        data = data_handling.get_data(dataset)
        data.y = data.y.squeeze()
        print(data)
        print(data.x.shape, data.y.shape, torch.max(data.y))
        loss_func = torch.nn.CrossEntropyLoss()
        classes = torch.max(data.y) + 1
        
        data = LargestConnectedComponents(1, 'strong')(data)
        num_nodes = len(data.y)
        data.edge_index, data.edge_attr = remove_self_loops(
            data.edge_index, data.edge_attr)

        print(data)
        device = args.device
        data = data.to(device)

        homogeneous = False
        activation = not args.l
        patience = 2000
        for num_layers in args.layers:#1,2,4,8,16,32,64,
            loss_dict = {}
            for conv in args.convs:
                print(f'{dataset} {device}, {conv}')
                best_accs = []
                best_losses = []
                seed_losses = []

                conv_accs = []
                start = time.time()
                for seed in range(seeds):
                    torch.manual_seed(seed)
                    model = Model(data.x.shape[-1],args.h_dim, classes, args.heads, conv, num_layers, activation, data.edge_index.shape[1]).to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                    best_acc = 0
                    best_loss = torch.inf
                    not_improved = 0

                    loss_avg = []
                    all_losses = []
                    train_labels = data.y
                    for i in range(epochs):
                        out, out_convs = model(data.x, data.edge_index)
                        train_out = out#[]
                        train_loss = loss_func(train_out,train_labels)
                        loss = train_loss
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        train_acc = torch.sum(torch.eq(torch.max(train_out,dim=-1)[1], train_labels))/len(train_labels)
                        if train_acc > best_acc:
                            best_acc = train_acc
                        if train_loss < best_loss:
                            best_loss = train_loss
                        loss_avg.append(train_loss.item())
                        if i % 16 == 0:
                            all_losses.append(np.mean(loss_avg))
                            loss_avg = []
                        if verbose and i % 2000 == 0:
                            print(f'{i}, \tTrain: {train_acc.item():.2f}/{np.mean(loss_avg):.2f}, ({best_acc:.2f}/{best_loss:.2f}, {not_improved}), {torch.norm(out_convs):.2f}')
                            loss_avg = []
                    print(f'{conv}({num_layers}): T-{best_acc:.2f}/{best_loss:.2f} in {time.time() - start:.2f}s ({i} steps) [Seed: {seed}] Final norm: {torch.norm(out_convs):.3f}')
                    seed_losses.append(all_losses)
                loss_dict[conv] = seed_losses
            plot_losses(loss_dict, f'{dataset}_{num_layers}_{args.h_dim}')
                