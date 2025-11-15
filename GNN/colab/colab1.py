import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

G = nx.karate_club_graph()


def average_degree(num_edges, num_nodes):
    # TODO: Implement this function that takes number of edges
    # and number of nodes, and returns the average node degree of
    # the graph. Round the result to nearest integer (for example
    # 3.3 will be rounded to 3 and 3.7 will be rounded to 4)

    avg_degree = int(num_edges / num_nodes + 0.5)

    return avg_degree


def average_clustering_coefficient(G):
    # TODO: Implement this function that takes a nx.Graph
    # and returns the average clustering coefficient. Round
    # the result to 2 decimal places (for example 3.333 will
    # be rounded to 3.33 and 3.7571 will be rounded to 3.76)

    ############# Your code here ############
    ## Note:
    ## 1: Please use the appropriate NetworkX clustering function

    #########################################
    dic = nx.clustering(G)
    res = 0
    for k, v in dic.items():
        res += v
    res = res / len(dic)

    # or avg_cluster_coef = nx.average_clustering(G)
    avg_cluster_coef = int(res * 100) / 100

    return avg_cluster_coef


def one_iter_pagerank(G, beta, r0, node_id):
    # TODO: Implement this function that takes a nx.Graph, beta, r0 and node id.
    # The return value r1 is one interation PageRank value for the input node.
    # Please round r1 to 2 decimal places.

    r1 = 0

    ############# Your code here ############
    ## Note:
    ## 1: You should not use nx.pagerank

    #########################################
    neighbor = [n for n in G[node_id]]
    degrees = [d for _, d in G.degree(neighbor)]
    for d in degrees:
        r1 += beta * (r0 / d)
    r1 = round(r1 + (1 - beta) / G.number_of_nodes(), 2)

    return r1


def closeness_centrality(G, node=5):
    # TODO: Implement the function that calculates closeness centrality
    # for a node in karate club network. G is the input karate club
    # network and node is the node id in the graph. Please round the
    # closeness centrality result to 2 decimal places.

    closeness = 0

    ## Note:
    ## 1: You can use networkx closeness centrality function.
    ## 2: Notice that networkx closeness centrality returns the normalized
    ## closeness directly, which is different from the raw (unnormalized)
    ## one that we learned in the lecture.

    #########################################
    closeness = round(nx.closeness_centrality(G)[node] / (len(nx.node_connected_component(G, node)) - 1), 2)

    return closeness


import torch


def graph_to_edge_list(G):
    # TODO: Implement the function that returns the edge list of
    # an nx.Graph. The returned edge_list should be a list of tuples
    # where each tuple is a tuple representing an edge connected
    # by two nodes.

    edge_list = []
    for edge in G.edges:
        edge_list.append(edge)

    return edge_list


def edge_list_to_tensor(edge_list):
    # TODO: Implement the function that transforms the edge_list to
    # tensor. The input edge_list is a list of tuples and the resulting
    # tensor should have the shape [2 x len(edge_list)].

    edge_index = torch.tensor(edge_list, dtype=torch.long).t()

    return edge_index


pos_edge_list = graph_to_edge_list(G)
pos_edge_index = edge_list_to_tensor(pos_edge_list)
import random


def sample_negative_edges(G, num_neg_samples):
    nonedge = list(enumerate(nx.non_edges(G)))

    neg_edge_list = [random.sample(nonedge, num_neg_samples)[i][1] for i in
                     range(num_neg_samples)]

    return neg_edge_list


# Sample 78 negative edges
neg_edge_list = sample_negative_edges(G, len(pos_edge_list))
# Transform the negative edge list to tensor
neg_edge_index = edge_list_to_tensor(neg_edge_list)
# Initialize an embedding layer
# Suppose we want to have embedding for 4 items (e.g., nodes)
# Each item is represented with 8 dimensional vector

emb_sample = nn.Embedding(num_embeddings=4, embedding_dim=8)
# Please do not change / reset the random seed
torch.manual_seed(1)


def create_node_emb(num_node=34, embedding_dim=16):
    # TODO: Implement this function that will create the node embedding matrix.
    # A torch.nn.Embedding layer will be returned. You do not need to change
    # the values of num_node and embedding_dim. The weight matrix of returned
    # layer should be initialized under uniform distribution.

    emb = nn.Embedding(num_embeddings=num_node, embedding_dim=16)
    shape = emb.weight.data.shape
    emb.weight.data = torch.rand(shape)

    return emb


emb = create_node_emb()


def visualize_emb(emb):
    X = emb.weight.data.numpy()
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(6, 6))
    club1_x = []
    club1_y = []
    club2_x = []
    club2_y = []
    for node in G.nodes(data=True):
        if node[1]['club'] == 'Mr. Hi':
            club1_x.append(components[node[0]][0])
            club1_y.append(components[node[0]][1])
        else:
            club2_x.append(components[node[0]][0])
            club2_y.append(components[node[0]][1])
    plt.scatter(club1_x, club1_y, color="red", label="Mr. Hi")
    plt.scatter(club2_x, club2_y, color="blue", label="Officer")
    plt.legend()
    plt.show()


from torch.optim import SGD
import torch.nn as nn


def accuracy(pred, label):
    # TODO: Implement the accuracy function. This function takes the
    # pred tensor (the resulting tensor after sigmoid) and the label
    # tensor (torch.LongTensor). Predicted value greater than 0.5 will
    # be classified as label 1. Else it will be classified as label 0.
    # The returned accuracy should be rounded to 4 decimal places.
    # For example, accuracy 0.82956 will be rounded to 0.8296.

    accu = sum(torch.round(pred) == label) / len(pred)

    return accu


def train(emb, loss_fn, sigmoid, train_label, train_edge):
    # TODO: Train the embedding layer here. You can also change epochs and
    # learning rate. In general, you need to implement:
    # (1) Get the embeddings of the nodes in train_edge
    # (2) Dot product the embeddings between each node pair
    # (3) Feed the dot product result into sigmoid
    # (4) Feed the sigmoid output into the loss_fn
    # (5) Print both loss and accuracy of each epoch
    # (6) Update the embeddings using the loss and optimizer
    # (as a sanity check, the loss should decrease during training)

    epochs = 500
    learning_rate = 0.1

    optimizer = SGD(emb.parameters(), lr=learning_rate, momentum=0.9)

    for i in range(epochs):
        optimizer.zero_grad() #
        pred = sigmoid(torch.sum(emb(train_edge)[0].mul(emb(train_edge)[1]), 1))
        loss = loss_fn(pred, train_label) # loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        print("Epoch {} Loss: {}, Accuracy: {}".format(i, loss, accuracy(pred, train_label)))


loss_fn = nn.BCELoss()
sigmoid = nn.Sigmoid()

print(pos_edge_index.shape)

# Generate the positive and negative labels
pos_label = torch.ones(pos_edge_index.shape[1], )
neg_label = torch.zeros(neg_edge_index.shape[1], )

# Concat positive and negative labels into one tensor
train_label = torch.cat([pos_label, neg_label], dim=0)

# Concat positive and negative edges into one tensor
# Since the network is very small, we do not split the edges into val/test sets
train_edge = torch.cat([pos_edge_index, neg_edge_index], dim=1)
print(train_edge.shape)

train(emb, loss_fn, sigmoid, train_label, train_edge)
# Visualize the final learned embedding
visualize_emb(emb)
