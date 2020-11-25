import numpy as np
import torch
from random import sample



def task_data_generator(features, labels, adj, task_num, label_dim, dataset, device, spt_ratio):
    x_spt = []
    y_spt = []
    idx_spt = []
    x_qry = []
    y_qry = []
    idx_qry = []
    chemical_bond_vectors = []
    for t in range(task_num):
        train_index = sample(range(len(features[t])), int(spt_ratio * len(features[t])) if int(spt_ratio * len(features[t])) > 0 else 1)
        test_index = list(set(range(len(features[t]))) - set(train_index))
        train_attr = (features[t])[train_index]
        test_attr = (features[t])[test_index]
        train_label = (labels[t])[train_index]
        test_label = (labels[t])[test_index]
        x_spt.append(train_attr.to(device))
        y_spt.append(train_label.to(device))
        idx_spt.append((torch.from_numpy(np.array(train_index)).to(device)))
        x_qry.append(test_attr.to(device))
        y_qry.append(test_label.to(device))
        idx_qry.append((torch.from_numpy(np.array(test_index)).to(device)))

        the_adj = adj[t].cpu().clone().detach()
        mask_labels = labels[t].cpu().clone().detach()
        the_adj = the_adj.numpy()
        mask_labels = mask_labels.numpy()
        if dataset == 'Cuneiform':
            label1 = np.argmax(mask_labels[:, 0:4], axis=1)
            label2 = np.argmax(mask_labels[:, 4:], axis=1)
            label_tuple = []
            for h in range(len(label1)):
                label_tuple.append((label1[h], label2[h]))
            general_label_tuple = []
            for j in range(4):
                for k in range(3):
                    general_label_tuple.append((j, k))
            for a in range(len(label_tuple)):
                for b in range(len(general_label_tuple)):
                    if label_tuple[a] == general_label_tuple[b]:
                        label_tuple[a] = b
            mask_labels = np.array(label_tuple)
            mask_labels[test_index] = -1
        else:
            mask_labels[test_index] = -1

        for m in range(2):
            for i in range(len(the_adj[m])):
                x = the_adj[m][i]
                the_adj[m][i] = mask_labels[x]
        E = the_adj.T
        Edge = []
        for i in range(len(E)):
            Edge.append(tuple(list(E[i])))
        general_labels = []
        if dataset != 'Cuneiform':
            for i in range(-1, label_dim):
                for j in range(-1, label_dim):
                    general_labels.append((i, j))
        else:
            for i in range(-1, 12):
                for j in range(-1, 12):
                    general_labels.append((i, j))
        chemical_bond_vector = []
        for i in range(len(general_labels)):
            chemical_bond_vector.append(Edge.count(general_labels[i]))
        chemical_bond_vectors.append(chemical_bond_vector)
    return x_spt, y_spt, x_qry, y_qry, idx_spt, idx_qry, chemical_bond_vectors
