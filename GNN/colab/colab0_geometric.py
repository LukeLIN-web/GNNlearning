# Helper function for visualization.
# %matplotlib inline
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch_geometric


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                        f'Training Accuracy: {accuracy["train"] * 100:.2f}% \n'
                        f' Validation Accuracy: {accuracy["val"] * 100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.show()


from torch.nn import Linear
from torch_geometric.nn import GCNConv

from torch_geometric.datasets import KarateClub

dataset = KarateClub()
num_layers = 3
input_dim = 34
hidden_dim = 4


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.classifier = Linear(2, dataset.num_classes)

        self.convs = torch.nn.ModuleList()

        self.convs.append(GCNConv(input_dim, hidden_dim))
        for l in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):

        for l in range(num_layers):
            x = self.convs[l](x, edge_index)
            x = x.tanh()

        # h = self.relu(h)
        h = torch.nn.functional.relu(h)
        h = torch.nn.functional.dropout(h, dropout=0.5, training=self.training)
        h = self.conv3(h, edge_index)
        embeddings = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(embeddings)

        return out, embeddings


model = GCN()
data = dataset[0]  # Get the first graph object.
_, h = model(data.x, data.edge_index)
print(f'Embedding shape: {list(h.shape)}')

visualize(h, color=data.y)