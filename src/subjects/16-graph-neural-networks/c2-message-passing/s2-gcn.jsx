import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function NormalizationViz() {
  const [normType, setNormType] = useState('sym')
  const degrees = [2, 3, 1, 4, 2]
  const labels = ['v0', 'v1', 'v2', 'v3', 'v4']

  const getWeight = (di, dj) => {
    if (normType === 'none') return 1
    if (normType === 'row') return (1 / di).toFixed(2)
    return (1 / Math.sqrt(di * dj)).toFixed(3)
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Aggregation Normalization</h3>
      <div className="flex items-center gap-2 mb-3">
        {[['none', 'A'], ['row', 'D^{-1}A'], ['sym', 'D^{-1/2}AD^{-1/2}']].map(([key, label]) => (
          <button key={key} onClick={() => setNormType(key)}
            className={`px-3 py-1 rounded text-xs ${key === normType ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>
            {label}
          </button>
        ))}
      </div>
      <div className="flex gap-3 justify-center flex-wrap">
        {labels.map((l, i) => (
          <div key={i} className="text-center px-3 py-2 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700">
            <div className="text-xs text-gray-500">{l} (deg={degrees[i]})</div>
            <div className="text-sm font-mono text-violet-600 dark:text-violet-400">
              w = {getWeight(degrees[i], degrees[Math.min(i + 1, 4)])}
            </div>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Symmetric normalization prevents high-degree nodes from dominating</p>
    </div>
  )
}

export default function GCN() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Graph Convolutional Networks (GCN) by Kipf & Welling (2017) simplified spectral graph
        convolutions to a first-order approximation, creating the most influential GNN architecture.
      </p>

      <DefinitionBlock title="GCN Layer">
        <p>The GCN propagation rule:</p>
        <BlockMath math="H^{(l+1)} = \sigma\!\left(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)}\right)" />
        <p className="mt-2">where <InlineMath math="\tilde{A} = A + I" /> (self-loops added),
        <InlineMath math="\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}" />, and <InlineMath math="W^{(l)}" /> is the trainable weight matrix.</p>
      </DefinitionBlock>

      <TheoremBlock title="Spectral Motivation" id="gcn-spectral">
        <p>GCN derives from a first-order Chebyshev approximation of spectral filters:</p>
        <BlockMath math="g_\theta \star x \approx \theta_0 x + \theta_1 (L - I) x = \theta_0 x - \theta_1 D^{-1/2} A D^{-1/2} x" />
        <p className="mt-2">Setting <InlineMath math="\theta_0 = -\theta_1 = \theta" /> and adding self-loops yields the GCN formula.
        This connects spectral theory to a simple, efficient spatial computation.</p>
      </TheoremBlock>

      <NormalizationViz />

      <ExampleBlock title="GCN on Cora">
        <p>The Cora citation network: 2708 papers, 5429 edges, 7 classes, 1433-dim bag-of-words features.</p>
        <p>A 2-layer GCN (16 hidden units) achieves ~81% test accuracy with just 140 labeled nodes
        (20 per class). This demonstrated the power of semi-supervised learning on graphs.</p>
      </ExampleBlock>

      <PythonCode
        title="GCN Implementation from Scratch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GCN(dataset.num_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training loop
for epoch in range(200):
    model.train()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
pred = model(data.x, data.edge_index).argmax(dim=1)
acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
print(f"Test accuracy: {acc:.4f}")`}
      />

      <NoteBlock type="note" title="GCN Limitations">
        <p>
          GCN uses fixed, symmetric normalization weights, treating all neighbors equally (up to
          degree). It cannot distinguish structurally different neighborhoods with the same degree.
          GAT addresses this by learning edge-specific attention weights.
        </p>
      </NoteBlock>
    </div>
  )
}
