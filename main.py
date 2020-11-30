from mydataset import MyDataset
import os.path as osp
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_adj, to_scipy_sparse_matrix
from sklearn import preprocessing
import numpy as np
import argparse
import torch
import scipy.sparse as sp
from random import sample
import random
from model import Meta
from earlystopping import EarlyStopping
from task_data_generator import task_data_generator

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(
        d_mat_inv_sqrt).tocoo()  # adding self-loop, normalization, symmetric normalization


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def sgc_precompute(features, adj,
                   degree):  # This is the neiborhood information aggregation part, degree represents number of aggregating times
    for i in range(degree):
        features = torch.spmm(adj, features)
    return features


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    graph_labels_list = []
    graph_adj_list = []
    graph_features_list = []

    if args.dataset == 'DHFR':
        num_attri = 3
        label_dim = 9
    elif args.dataset == 'COX2':
        num_attri = 3
        label_dim = 8
    elif args.dataset == 'Cuneiform':
        num_attri = 3
        label_dim = 7
    elif args.dataset == 'Sub_Flickr':
        num_attri = 500
        label_dim = 7
    elif args.dataset == 'BZR':
        num_attri = 3
        label_dim = 10
    elif args.dataset == 'PROTEINS_full':
        num_attri = 29
        label_dim = 3

    social_datasets = {'Sub_Flickr'}
    if args.dataset not in social_datasets:
        path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
        datasets = MyDataset(path, args.dataset)
        print('{}'.format(args.dataset))
        loader = DataLoader(datasets)
        for data in loader:
            labels = data.x[:, num_attri:]
            graph_labels_list.append(labels)
            adj = data.edge_index
            graph_adj_list.append(adj)
            features = data.x[:, 0:num_attri]
            graph_features_list.append(features)

    else:
        datasets = torch.load('./data/{}/processed/data.pt'.format(args.dataset))
        for a in range(len(datasets)):
            labels = datasets[a].y
            graph_labels_list.append(labels)
            adj = datasets[a].edge_index
            graph_adj_list.append(adj)
            features = datasets[a].x
            graph_features_list.append(features)

    train_graphs_index = sample(range(len(graph_labels_list)), int(0.6 * len(graph_labels_list)))
    val_test_graphs_index = list(set(range(len(graph_labels_list))) - set(train_graphs_index))
    test_graphs_index = sample(val_test_graphs_index, int(0.5 * len(val_test_graphs_index)))
    val_graphs_index = list(set(val_test_graphs_index) - set(test_graphs_index))

    def feature_label_generator(index, shuffle=False):
        if shuffle:
            random.shuffle(index)
        processed_features_list = []
        reformed_labels_list = []
        adjs_list = []
        for i in index:
            if (args.dataset != 'Cuneiform') & (args.dataset not in social_datasets):
                labels = torch.LongTensor(graph_labels_list[i].long())
                labels = torch.max(labels, dim=1)[1]
            elif args.dataset == 'Cuneiform':
                labels = graph_labels_list[i]
            else:
                labels = torch.LongTensor(graph_labels_list[i].long())
            reformed_labels_list.append(labels.to(device))
            adjs_list.append(graph_adj_list[i].to(device))
            adj = to_scipy_sparse_matrix(graph_adj_list[i],
                                         num_nodes=len(labels))  # there are some graphs having isolated points
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = aug_normalized_adjacency(adj)
            adj = sparse_mx_to_torch_sparse_tensor(adj).float()
            initial_feature = torch.FloatTensor(preprocessing.scale(graph_features_list[i]))
            aggregated_feature = sgc_precompute(initial_feature, adj, args.aggregation_times)
            processed_features_list.append(aggregated_feature.to(device))
        return processed_features_list, reformed_labels_list, adjs_list


    val_graphs_features_list, val_graphs_labels_list, val_adj_list = feature_label_generator(val_graphs_index)
    test_graphs_features_list, test_graphs_labels_list, test_adj_list = feature_label_generator(test_graphs_index)


    config = [('linear', [args.hidden, num_attri]),
              ('linear', [label_dim, args.hidden])]

    if args.dataset == 'Cuneiform':
        config_chemi = [('linear', [args.hidden, (12 + 1) * (12 + 1)]),
                        ('leaky_relu', [args.hidden, args.hidden])]
    else:
        config_chemi = [('linear', [args.hidden, (label_dim + 1) * (label_dim + 1)]),
                        ('leaky_relu', [args.hidden, args.hidden])]

    config_scal = [('linear', [args.hidden * (num_attri + 1) + label_dim * (args.hidden + 1), args.hidden])]

    config_trans = [('linear', [args.hidden * (num_attri + 1) + label_dim * (args.hidden + 1), args.hidden])]

    inductive_meta = Meta(config, config_chemi, config_scal, config_trans, args, num_attri, label_dim).to(device)

    patience = 20
    early_stopping = EarlyStopping(args.dataset, patience, verbose=True)

    val_x_spt, val_y_spt, val_x_qry, val_y_qry, val_idx_spt, val_idx_qry, val_chemical_bond_vectors \
        = task_data_generator(val_graphs_features_list,
                              val_graphs_labels_list,
                              val_adj_list,
                              len(val_graphs_labels_list),
                              label_dim,
                              args.dataset, device,
                              args.spt_ratio)

    test_x_spt, test_y_spt, test_x_qry, test_y_qry, test_idx_spt, test_idx_qry, test_chemical_bond_vectors \
        = task_data_generator(test_graphs_features_list,
                              test_graphs_labels_list,
                              test_adj_list,
                              len(test_graphs_labels_list),
                              label_dim,
                              args.dataset, device,
                              args.spt_ratio)

    scaler_chemi = preprocessing.StandardScaler()
    val_chemical_bond_vectors = scaler_chemi.fit_transform(val_chemical_bond_vectors)
    val_chemical_bond_vectors = (torch.from_numpy(val_chemical_bond_vectors.astype(np.float32))).to(device)
    test_chemical_bond_vectors = scaler_chemi.transform(test_chemical_bond_vectors)
    test_chemical_bond_vectors = (torch.from_numpy(test_chemical_bond_vectors.astype(np.float32))).to(device)

    for Epoch in range(args.epoch):
        inductive_meta.train()
        train_graphs_features_list, train_graphs_labels_list, train_adj_list = feature_label_generator(train_graphs_index, shuffle=True)

        train_x_spt, train_y_spt, train_x_qry, train_y_qry, train_idx_spt, train_idx_qry, train_chemical_bond_vectors \
            = task_data_generator(train_graphs_features_list,
                                  train_graphs_labels_list,
                                  train_adj_list,
                                  len(train_graphs_labels_list),
                                  label_dim,
                                  args.dataset, device,
                                  args.spt_ratio)

        train_chemical_bond_vectors = scaler_chemi.transform(train_chemical_bond_vectors)
        train_chemical_bond_vectors = (torch.from_numpy(train_chemical_bond_vectors.astype(np.float32))).to(device)

        acc, loss = inductive_meta.forward(train_x_spt, train_y_spt, train_x_qry,
                                           train_y_qry,
                                           train_chemical_bond_vectors,
                                           args.l2_coef,
                                           update_step=args.update_step,
                                           len_graphs_index=len(train_graphs_index),
                                           batch_size=args.batch_size,
                                           dataset=args.dataset, train='train',
                                           training=True, epoch=Epoch)
        if Epoch % 10 == 0:
            print('Step:', Epoch, '\t Meta_Training_Accuracy:{:.4f},loss{:.4f}'.format(acc, loss))
        inductive_meta.eval()
        accs, losses = inductive_meta.forward(val_x_spt, val_y_spt, val_x_qry,
                                              val_y_qry,
                                              val_chemical_bond_vectors,
                                              args.l2_coef,
                                              update_step=args.update_step,
                                              len_graphs_index=len(train_graphs_index),
                                              batch_size=args.batch_size,
                                              dataset=args.dataset, train='val',
                                              training=False, epoch=Epoch)

        if Epoch % 10 == 0:
            print('\t Meta_validating_Accuracy:{:.4f},loss{:.4f}'.format(accs, losses))
        early_stopping(accs, inductive_meta)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    test_model = Meta(config, config_chemi, config_scal, config_trans, args, num_attri, label_dim).to(device)
    test_model.load_state_dict(torch.load('./meta_checkpoint.pkl'.format(args.dataset)))
    test_model.eval()
    test_acc, test_loss = test_model.forward(test_x_spt, test_y_spt, test_x_qry,
                                             test_y_qry,
                                             test_chemical_bond_vectors,
                                             args.l2_coef,
                                             update_step=args.update_step,
                                             len_graphs_index=len(train_graphs_index),
                                             batch_size=args.batch_size,
                                             dataset=args.dataset, train='test',
                                             training=False, epoch=Epoch)
    print('\t Testing_Accuracy:{:.4f},loss{:.4f}'.format(test_acc, test_loss))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cuneiform', help='Dataset to use.')
    parser.add_argument('--aggregation_times', type=int, default=2, help='Aggregation times')
    parser.add_argument('--hidden', type=str, default=16, help='number of hidden neurons for gnn')
    parser.add_argument('--epoch', type=int, default=201, help='epoch number')
    parser.add_argument('--task_lr', type=float, default=0.005, help='task level adaptation learning rate')
    parser.add_argument('--meta_lr', type=float, default=0.01, help='the outer framework learning rate')
    parser.add_argument('--beta_hidden', type=int, default=16, help='number of hidden neurons for gnn')
    parser.add_argument('--batch_size', type=int, default=6, help='number of graphs per batch')
    parser.add_argument('--update_step', type=int, default=2, help='number of task level adaptation steps')
    parser.add_argument('--spt_ratio', type=float, default=0.5, help='the ratio of support set in one graph')
    parser.add_argument('--l2_coef', type=float, default=0.01, help='l2 regularization coefficient')
    parser.add_argument('--seed', type=int, default=16, help='random seed')

    args = parser.parse_args()

    main(args)



