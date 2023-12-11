import torch
import time
import torch_geometric as pyg
import numpy as np
import argparse

class Model(torch.nn.Module):
    def __init__(self, n, d, conv, num_layers, act, homogeneous, mask):
        super(Model, self).__init__()
        self.enc = torch.nn.Linear(d,d)
        self.dec = torch.nn.Linear(d,y.shape[-1], bias=True)
        self.num_layers = num_layers
        self.act = act
        degrees = 1/torch.sum(mask,dim=1,keepdim=True)

        if homogeneous:
            A1 = torch.nn.Parameter(torch.randn(n,n)*0.05 + (degrees))
            A2 = torch.nn.Parameter(torch.randn(n,n)*0.05 + (degrees))
        else:
            A1 = None
            A2 = None
        
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ConvLayer(n, d, conv, A1, A2, degrees))

    def forward(self, x, mask):
        x = torch.relu(self.enc(x))
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
            if self.act:
                x = torch.nn.functional.relu(x)
        return self.dec(x), x

class ConvLayer(torch.nn.Module):
    def __init__(self, n, d, conv, A1, A2, degrees):
        super(ConvLayer, self).__init__()
        self.conv = conv

        if A1 is None:
            self.A1 = torch.nn.Parameter(torch.randn(n,n)*0.05 + (degrees))#1/n)#1/(n*0.2))#
            self.A2 = torch.nn.Parameter(torch.randn(n,n)*0.05 + (degrees))#1/n)#1/(n*0.2))#
        else:
            self.A1 = A1
            self.A2 = A2
        self.W = torch.nn.Parameter(torch.randn(d,d)*0.05 + 1/d)
        self.W2 = torch.nn.Parameter(torch.randn(d,d)*0.05+  1/d)

    def forward(self, x, mask):
        A1 = self.A1 * mask
        if self.conv == 'softmax_SKP':
            A2 = self.A2 * mask
            A1, A2 = torch.softmax(A1,dim=-1), torch.softmax(A2,dim=-1)
            x = 0.5*torch.matmul(torch.matmul(A1, x),self.W) + 0.5*torch.matmul(torch.matmul(A2, x),self.W2)
        elif self.conv == 'KP':
            x = torch.matmul(torch.matmul(A1, x),self.W)
        elif self.conv == 'SKP':
            A1, A2 = A1,self.A2* mask
            x = (0.5*torch.matmul(torch.matmul(A1, x),self.W) + 0.5*torch.matmul(torch.matmul(A2, x),self.W2))
        else:
            print("Error!")
        return x


    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='training parameters')
    parser.add_argument('-ho', action='store_true',
                    help='Whether to use the same feature transformations for all layers.')
    parser.add_argument('-li', action='store_true',
                    help='Whether to use a linearized model without ReLU activations.')
    parser.add_argument('--conv', type=str, default='softmax_SKP',
                    help='Base model: softmax_SKP, KP, SKP')
    parser.add_argument('--num_graphs', type=int, default=50, help='Number of random graphs to evaluate')
    parser.add_argument('--start_graph', type=int, default=0, help='First graph seed to evaluate.')
    args = parser.parse_args()
    
    device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    print(device)
    n = 20
    track_loss = False

    all_accs = []
    for k in range(args.start_graph, args.num_graphs):
        torch.manual_seed(k)
        disallowed_graph = True
        while disallowed_graph:
            edge_index = pyg.utils.random.erdos_renyi_graph(n, 0.2, True)
            mask = pyg.utils.to_dense_adj(edge_index)[0].to(device)
            in_degrees = torch.sum(mask,dim=1,keepdim=True)
            out_degrees = torch.sum(mask,dim=0,keepdim=True)
            disallowed_graph = torch.sum(in_degrees == 0.0) + torch.sum(out_degrees == 0.0) > 0

        tasks = 3
        y = torch.zeros(tasks+1,tasks)
        y[0,0] = 1.0
        y[1,0] = 1.0
        y[0,1] = 1.0
        y[2,1] = 1.0
        if tasks > 2:
            y[0,2] = 1.0
            y[3,2] = 1.0
        y = y.to(device)
        d = y.shape[-1]*2



        conv_accs = []
        start = time.time()
        for seed in range(3):
            accs = []
            for num_layers in [1,2,4,8,16,32,64,128]:
                losses = []
                torch.manual_seed(seed + k*10)
                x = torch.randn(n,d).to(device)
                model = Model(n, d, args.conv, num_layers, not args.li, args.ho, mask.to('cpu')).to(device)
                optimizer = torch.optim.Adam(model.parameters())
                best_acc = 0
                best_loss = torch.inf
                for i in range(100000):
                    out, out_convs = model(x, mask)
                    out = out[:tasks+1]
                    loss = torch.mean(torch.binary_cross_entropy_with_logits(out, y))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    acc = torch.sum(torch.eq(out > 0.0, y)) / len(y.view(-1))

                    if track_loss:
                        losses.append(loss.item())
                        if i == 5000:
                            best_acc = acc
                            best_loss = loss
                            break
                    else:
                        if acc > best_acc:
                            best_acc = acc
                            if best_acc == 1.0:
                                break
                        if loss < best_loss:
                            best_loss = loss
                            not_improved = 0
                        else:
                            not_improved += 1
                            if not_improved == 500:
                                break
                if track_loss:
                    np.savetxt(f'./results/loss_{args.conv}_h{args.ho}_l{args.li}_{k}_{time.time()}.txt', torch.FloatTensor(losses).numpy(), fmt='%.4f', delimiter=',')
                accs.append(best_acc)

            conv_accs.append(accs)
        print(f'({k}) {args.conv}, Hom:{args.ho}, Linear: {args.li}: Max: {(torch.tensor(conv_accs).max(dim=0)[0])}, Mean: {torch.tensor(conv_accs).mean(dim=0)}, Std: {(torch.tensor(conv_accs).std(dim=0))} in {time.time()-start}')
        all_accs.append(torch.tensor(conv_accs).max(dim=0)[0])
        np.savetxt(f'./{args.conv}_h{args.ho}_l{args.li}_{k}_{time.time()}.txt', torch.stack(all_accs).numpy(), fmt='%.4f', delimiter=',')