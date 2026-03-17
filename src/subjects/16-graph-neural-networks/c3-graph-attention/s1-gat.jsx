import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AttentionWeightViz() {
  const [targetNode, setTargetNode] = useState(0)
  const neighbors = {
    0: [{ id: 1, w: 0.35 }, { id: 2, w: 0.45 }, { id: 3, w: 0.20 }],
    1: [{ id: 0, w: 0.60 }, { id: 2, w: 0.25 }, { id: 4, w: 0.15 }],
    2: [{ id: 0, w: 0.30 }, { id: 1, w: 0.30 }, { id: 3, w: 0.40 }],
  }
  const nodePositions = { 0: [80, 80], 1: [200, 40], 2: [200, 120], 3: [320, 80], 4: [320, 30] }
  const nbs = neighbors[targetNode] || []

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">GAT Attention Weights</h3>
      <div className="flex items-center gap-2 mb-3">
        {[0, 1, 2].map(n => (
          <button key={n} onClick={() => setTargetNode(n)}
            className={`px-3 py-1 rounded text-sm ${n === targetNode ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>
            v{n}
          </button>
        ))}
      </div>
      <svg width={380} height={150} className="mx-auto block">
        {nbs.map((nb, k) => {
          const [x1, y1] = nodePositions[targetNode]
          const [x2, y2] = nodePositions[nb.id]
          return <line key={k} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#7c3aed" strokeWidth={nb.w * 8} opacity={0.6} />
        })}
        {Object.entries(nodePositions).map(([id, [x, y]]) => (
          <g key={id}>
            <circle cx={x} cy={y} r={18} fill={parseInt(id) === targetNode ? '#7c3aed' : '#e5e7eb'} stroke="#7c3aed" strokeWidth={1.5} />
            <text x={x} y={y + 4} textAnchor="middle" fill={parseInt(id) === targetNode ? 'white' : '#374151'} fontSize={11} fontWeight="bold">v{id}</text>
          </g>
        ))}
        {nbs.map((nb, k) => {
          const [x1, y1] = nodePositions[targetNode]
          const [x2, y2] = nodePositions[nb.id]
          return <text key={k} x={(x1 + x2) / 2} y={(y1 + y2) / 2 - 6} textAnchor="middle" fill="#7c3aed" fontSize={10} fontWeight="bold">{nb.w.toFixed(2)}</text>
        })}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">Edge thickness proportional to learned attention weight</p>
    </div>
  )
}

export default function GAT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Graph Attention Networks (GAT) apply the attention mechanism to graphs, learning to
        weight different neighbors differently. This allows the model to focus on the most
        relevant neighbors for each node.
      </p>

      <DefinitionBlock title="GAT Attention Mechanism">
        <p>Attention coefficients between node <InlineMath math="i" /> and neighbor <InlineMath math="j" />:</p>
        <BlockMath math="e_{ij} = \text{LeakyReLU}\!\left(\mathbf{a}^\top [\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)" />
        <BlockMath math="\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}" />
        <p className="mt-2">The output:</p>
        <BlockMath math="\mathbf{h}_i' = \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij} \mathbf{W} \mathbf{h}_j\right)" />
      </DefinitionBlock>

      <DefinitionBlock title="Multi-Head Attention">
        <p>Use <InlineMath math="K" /> independent attention heads and concatenate (or average in the final layer):</p>
        <BlockMath math="\mathbf{h}_i' = \Big\|_{k=1}^K \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_j\right)" />
      </DefinitionBlock>

      <AttentionWeightViz />

      <ExampleBlock title="GAT on Cora">
        <p>Original GAT: 2 layers, 8 attention heads in layer 1 (hidden dim 8 each = 64 total),
        1 head in layer 2. Achieves ~83% accuracy on Cora, improving over GCN's ~81%.</p>
        <p>The attention weights are interpretable: we can see which neighbors each node considers most important.</p>
      </ExampleBlock>

      <PythonCode
        title="GAT in PyTorch Geometric"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        # Output layer: 1 head with averaging
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GAT(in_dim=1433, hidden_dim=8, out_dim=7, heads=8)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
# Much fewer params than Transformer: attention is only over neighbors!`}
      />

      <NoteBlock type="note" title="GAT vs Transformer Attention">
        <p>
          GAT attention is computed only over graph neighbors (sparse), while Transformer attention
          is over all tokens (dense). GAT uses a single-layer attention function
          <InlineMath math="\mathbf{a}^\top [\cdot \| \cdot]" />, which is actually limited in
          expressiveness. GATv2 addresses this with a more powerful dynamic attention mechanism.
        </p>
      </NoteBlock>
    </div>
  )
}
