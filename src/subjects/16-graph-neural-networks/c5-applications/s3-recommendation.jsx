import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BipartiteGraphViz() {
  const [selectedUser, setSelectedUser] = useState(0)
  const users = ['U0', 'U1', 'U2', 'U3']
  const items = ['I0', 'I1', 'I2', 'I3', 'I4']
  const interactions = [[0,0],[0,2],[0,3],[1,1],[1,2],[2,0],[2,4],[3,1],[3,3],[3,4]]
  const userPositions = users.map((_, i) => ({ x: 50, y: 30 + i * 40 }))
  const itemPositions = items.map((_, i) => ({ x: 280, y: 20 + i * 38 }))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">User-Item Bipartite Graph</h3>
      <div className="flex items-center gap-2 mb-3">
        {users.map((u, i) => (
          <button key={i} onClick={() => setSelectedUser(i)}
            className={`px-3 py-1 rounded text-sm ${i === selectedUser ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>
            {u}
          </button>
        ))}
      </div>
      <svg width={340} height={200} className="mx-auto block">
        {interactions.map(([u, it], k) => {
          const isActive = u === selectedUser
          return <line key={k} x1={userPositions[u].x + 20} y1={userPositions[u].y} x2={itemPositions[it].x - 20} y2={itemPositions[it].y}
            stroke={isActive ? '#7c3aed' : '#e5e7eb'} strokeWidth={isActive ? 2 : 1} opacity={isActive ? 0.8 : 0.3} />
        })}
        {userPositions.map((p, i) => (
          <g key={`u${i}`}>
            <circle cx={p.x} cy={p.y} r={16} fill={i === selectedUser ? '#7c3aed' : '#e5e7eb'} stroke="#7c3aed" strokeWidth={1} />
            <text x={p.x} y={p.y + 4} textAnchor="middle" fill={i === selectedUser ? 'white' : '#374151'} fontSize={10} fontWeight="bold">{users[i]}</text>
          </g>
        ))}
        {itemPositions.map((p, i) => {
          const connected = interactions.some(([u, it]) => u === selectedUser && it === i)
          return (
            <g key={`i${i}`}>
              <rect x={p.x - 16} y={p.y - 14} width={32} height={28} rx={4} fill={connected ? '#f97316' : '#e5e7eb'} stroke="#f97316" strokeWidth={1} />
              <text x={p.x} y={p.y + 4} textAnchor="middle" fill={connected ? 'white' : '#374151'} fontSize={10} fontWeight="bold">{items[i]}</text>
            </g>
          )
        })}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">Select a user to see their item interactions</p>
    </div>
  )
}

export default function RecommendationSystems() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Recommendation systems naturally fit the graph framework: users and items form a bipartite
        graph connected by interactions. GNN-based methods learn embeddings that capture
        collaborative filtering signals through message passing.
      </p>

      <DefinitionBlock title="GNN-Based Collaborative Filtering">
        <p>Model the user-item bipartite graph. After <InlineMath math="L" /> layers of message passing:</p>
        <BlockMath math="\mathbf{e}_u^{(l+1)} = \text{AGG}\!\left(\left\{\mathbf{e}_i^{(l)} : i \in \mathcal{N}_u\right\}\right)" />
        <BlockMath math="\mathbf{e}_i^{(l+1)} = \text{AGG}\!\left(\left\{\mathbf{e}_u^{(l)} : u \in \mathcal{N}_i\right\}\right)" />
        <p className="mt-2">The predicted preference score:</p>
        <BlockMath math="\hat{y}_{ui} = \mathbf{e}_u^{\top} \mathbf{e}_i" />
      </DefinitionBlock>

      <BipartiteGraphViz />

      <TheoremBlock title="LightGCN Simplification" id="lightgcn">
        <p>LightGCN (He et al., 2020) removes feature transformations and nonlinearities from GCN, using only neighborhood aggregation:</p>
        <BlockMath math="\mathbf{e}_u^{(l+1)} = \sum_{i \in \mathcal{N}_u} \frac{1}{\sqrt{|\mathcal{N}_u|}\sqrt{|\mathcal{N}_i|}} \mathbf{e}_i^{(l)}" />
        <p className="mt-2">Final embeddings are the weighted sum across layers:</p>
        <BlockMath math="\mathbf{e}_u = \sum_{l=0}^{L} \alpha_l \, \mathbf{e}_u^{(l)}, \quad \alpha_l = \frac{1}{L+1}" />
        <p>This simple design outperforms more complex models, showing that for recommendation, the core benefit of GNNs is multi-hop connectivity.</p>
      </TheoremBlock>

      <ExampleBlock title="Real-World Deployments">
        <p><strong>PinSage</strong> (Pinterest): 3B nodes, 18B edges. Uses random walk-based sampling and efficient MapReduce training.</p>
        <p><strong>Uber Eats</strong>: GNN for restaurant recommendation based on user-restaurant-dish graph.</p>
        <p><strong>Amazon</strong>: Product recommendation using product co-purchase and co-view graphs.</p>
        <p><strong>TikTok</strong>: Video recommendation incorporating user-video-creator heterogeneous graph.</p>
      </ExampleBlock>

      <PythonCode
        title="LightGCN for Recommendation"
        code={`import torch
import torch.nn as nn
from torch_geometric.nn import LGConv

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embed_dim=64, n_layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        self.n_layers = n_layers
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def forward(self, edge_index):
        x = torch.cat([self.user_emb.weight, self.item_emb.weight])
        embs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            embs.append(x)
        # Layer combination (mean)
        out = torch.stack(embs, dim=0).mean(dim=0)
        return out

    def recommend(self, user_ids, edge_index, k=10):
        all_embs = self.forward(edge_index)
        n_users = self.user_emb.weight.shape[0]
        user_embs = all_embs[user_ids]
        item_embs = all_embs[n_users:]
        scores = user_embs @ item_embs.T
        _, top_k = scores.topk(k, dim=-1)
        return top_k

model = LightGCN(num_users=1000, num_items=5000, embed_dim=64)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Beyond Collaborative Filtering">
        <p>
          Modern GNN-based recommenders incorporate side information (item descriptions,
          user profiles), knowledge graphs (product categories, attributes), and temporal
          signals (session-based sequences). The graph structure enables reasoning about
          <em>why</em> an item is recommended through path-based explanations.
        </p>
      </NoteBlock>
    </div>
  )
}
