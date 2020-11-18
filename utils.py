import matplotlib.pyplot as plt
import networkx as nx
import torch

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1_score(output, label):
    # We only count the attack types while ignoring the benign.
    # I designed the prediction and label vector to indicate the benign as the first entry.
    # Thus, to calculate F1 score, slicing the vector from the second entry to the end.
    attack_pred = output[1:]
    attack_label = label[1:]
    
    # Precision
    true_predict = 0
    for pred_attack in output.nonzero():
        if label[0, pred_attack.numpy()[1]] == 1:
            true_predict += 1
    
    prec = (1+true_predict)/(1+len(output.nonzero()))

    # Recall
    predicted = 0
    for true_attack in label.nonzero():
        if output[0, true_attack.numpy()[1]] == 1:
            predicted += 1

    recall = (1+predicted)/(1+len(label.nonzero()))

    # F1 score
    f1 = 2*prec*recall/(prec+recall)

    return f1

def visualize(labels, g):
    pos = nx.spring_layout(g, seed=1)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
