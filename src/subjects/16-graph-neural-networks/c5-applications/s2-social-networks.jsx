import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CommunityDetectionViz() {
  const [threshold, setThreshold] = useState(0.5)
  const nodes = [
    { x: 60, y: 50, c: 0 }, { x: 100, y: 30, c: 0 }, { x: 130, y: 70, c: 0 },
    { x: 230, y: 40, c: 1 }, { x: 270, y: 60, c: 1 }, { x: 250, y: 100, c: 1 },
    { x: 160, y: 130, c: 2 }, { x: 120, y: 150, c: 2 },
  ]
  const edges = [[0,1],[1,2],[0,2],[3,4],[4,5],[3,5],[2,6],[5,6],[6,7]]
  const communityColors = ['#7c3aed', '#f97316', '#10b981']
  const similarities = [0.9, 0.85, 0.88, 0.92, 0.87, 0.90, 0.3, 0.35, 0.82]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Community Detection</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Similarity threshold: {threshold.toFixed(2)}
          <input type="range" min={0} max={1} step={0.05} value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={340} height={180} className="mx-auto block">
        {edges.map(([i, j], k) => (
          <line key={k} x1={nodes[i].x} y1={nodes[i].y} x2={nodes[j].x} y2={nodes[j].y}
            stroke={similarities[k] >= threshold ? '#7c3aed' : '#e5e7eb'}
            strokeWidth={similarities[k] >= threshold ? 2 : 1} opacity={similarities[k] >= threshold ? 0.6 : 0.3} />
        ))}
        {nodes.map((n, i) => (
          <circle key={i} cx={n.x} cy={n.y} r={14} fill={communityColors[n.c]} opacity={0.8} />
        ))}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">GNN embeddings reveal community structure through learned similarities</p>
    </div>
  )
}

export default function SocialNetworks() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Social networks are among the most natural applications of GNNs. Users are nodes,
        relationships are edges, and tasks include community detection, link prediction,
        influence modeling, and content recommendation.
      </p>

      <DefinitionBlock title="Link Prediction">
        <p>Predict the likelihood of a future edge between nodes <InlineMath math="u" /> and <InlineMath math="v" />:</p>
        <BlockMath math="P(e_{uv}) = \sigma\!\left(\mathbf{h}_u^\top \mathbf{h}_v\right) \quad \text{or} \quad P(e_{uv}) = \text{MLP}([\mathbf{h}_u \| \mathbf{h}_v \| \mathbf{h}_u \odot \mathbf{h}_v])" />
        <p className="mt-2">Training uses negative sampling: for each positive edge, sample <InlineMath math="K" /> random
        non-edges as negatives.</p>
      </DefinitionBlock>

      <CommunityDetectionViz />

      <DefinitionBlock title="Influence Maximization">
        <p>Find a seed set <InlineMath math="S" /> of <InlineMath math="k" /> nodes that maximizes information spread:</p>
        <BlockMath math="S^* = \arg\max_{|S| = k} \mathbb{E}[|\text{Influenced}(S)|]" />
        <p className="mt-2">GNNs can learn node influence scores by encoding network structure and historical
        cascade data, replacing expensive Monte Carlo simulations.</p>
      </DefinitionBlock>

      <ExampleBlock title="Social Network Tasks">
        <p><strong>Friend recommendation</strong>: Link prediction on the social graph (Facebook, LinkedIn).</p>
        <p><strong>Fake account detection</strong>: Node classification using structural features (suspicious connectivity patterns).</p>
        <p><strong>Content virality</strong>: Predict which posts will spread based on the poster's graph neighborhood.</p>
        <p><strong>Community detection</strong>: Unsupervised clustering of users into interest groups.</p>
      </ExampleBlock>

      <PythonCode
        title="Link Prediction with GNN"
        code={`import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import negative_sampling

class LinkPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        return self.conv2(h, edge_index)

    def decode(self, z, edge_index):
        """Dot product link predictor."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def forward(self, x, edge_index, pos_edges, neg_edges):
        z = self.encode(x, edge_index)
        pos_score = self.decode(z, pos_edges)
        neg_score = self.decode(z, neg_edges)
        return pos_score, neg_score

def link_pred_loss(pos_score, neg_score):
    pos_loss = -torch.log(torch.sigmoid(pos_score) + 1e-8).mean()
    neg_loss = -torch.log(1 - torch.sigmoid(neg_score) + 1e-8).mean()
    return pos_loss + neg_loss

model = LinkPredictor(in_dim=16, hidden_dim=64)
x = torch.randn(1000, 16)
edge_index = torch.randint(0, 1000, (2, 5000))
neg_edges = negative_sampling(edge_index, num_nodes=1000)
pos_score, neg_score = model(x, edge_index, edge_index, neg_edges)
loss = link_pred_loss(pos_score, neg_score)
print(f"Loss: {loss.item():.4f}")`}
      />

      <WarningBlock title="Scalability at Web Scale">
        <p>
          Social networks have billions of nodes and edges. Full-batch GNN training is impossible.
          Production systems use mini-batch training with neighbor sampling (PinSage at Pinterest),
          distributed training across machines, and quantization of embeddings. Inference often
          pre-computes embeddings offline and serves them via ANN indices.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Dynamic Graphs">
        <p>
          Social networks evolve over time: users join, friendships form and dissolve. Temporal
          GNNs (TGN, TGAT) incorporate timestamps into message passing, learning from the
          sequence of graph snapshots rather than a single static graph.
        </p>
      </NoteBlock>
    </div>
  )
}
