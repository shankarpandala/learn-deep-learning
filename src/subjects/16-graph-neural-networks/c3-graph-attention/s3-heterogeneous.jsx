import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function HeteroGraphViz() {
  const [showTypes, setShowTypes] = useState(true)
  const nodes = [
    { x: 60, y: 60, type: 'user', label: 'U1' },
    { x: 160, y: 30, type: 'user', label: 'U2' },
    { x: 260, y: 60, type: 'item', label: 'I1' },
    { x: 320, y: 120, type: 'item', label: 'I2' },
    { x: 100, y: 140, type: 'tag', label: 'T1' },
  ]
  const edges = [
    { from: 0, to: 2, type: 'buys' },
    { from: 1, to: 2, type: 'buys' },
    { from: 1, to: 3, type: 'buys' },
    { from: 0, to: 1, type: 'follows' },
    { from: 2, to: 4, type: 'tagged' },
    { from: 3, to: 4, type: 'tagged' },
  ]
  const typeColors = { user: '#7c3aed', item: '#f97316', tag: '#10b981' }
  const edgeColors = { buys: '#7c3aed', follows: '#f43f5e', tagged: '#10b981' }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Heterogeneous Graph</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showTypes} onChange={e => setShowTypes(e.target.checked)} className="accent-violet-500" />
          Color by type
        </label>
      </div>
      <svg width={380} height={170} className="mx-auto block">
        {edges.map((e, k) => (
          <line key={k} x1={nodes[e.from].x} y1={nodes[e.from].y} x2={nodes[e.to].x} y2={nodes[e.to].y}
            stroke={showTypes ? edgeColors[e.type] : '#d1d5db'} strokeWidth={1.5} opacity={0.6} />
        ))}
        {nodes.map((n, i) => (
          <g key={i}>
            <circle cx={n.x} cy={n.y} r={18} fill={showTypes ? typeColors[n.type] : '#7c3aed'} opacity={0.8} />
            <text x={n.x} y={n.y + 4} textAnchor="middle" fill="white" fontSize={10} fontWeight="bold">{n.label}</text>
          </g>
        ))}
      </svg>
      <div className="flex justify-center gap-4 text-xs mt-2">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-violet-500 inline-block" /> User</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-orange-500 inline-block" /> Item</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" /> Tag</span>
      </div>
    </div>
  )
}

export default function HeterogeneousGraphs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Real-world graphs often contain multiple types of nodes and edges. Heterogeneous graph
        neural networks handle this by using type-specific transformations and aggregations.
      </p>

      <DefinitionBlock title="Heterogeneous Graph">
        <p>A heterogeneous graph has node type mapping <InlineMath math="\tau: \mathcal{V} \to \mathcal{T}" /> and
        edge type mapping <InlineMath math="\phi: \mathcal{E} \to \mathcal{R}" />:</p>
        <BlockMath math="\mathcal{G} = (\mathcal{V}, \mathcal{E}, \tau, \phi), \quad |\mathcal{T}| + |\mathcal{R}| > 2" />
        <p className="mt-2">Each relation <InlineMath math="r" /> connects a source type to a target type, forming
        a <strong>metapath</strong> schema.</p>
      </DefinitionBlock>

      <HeteroGraphViz />

      <DefinitionBlock title="Relational Graph Attention (R-GAT)">
        <p>Use relation-specific transformations:</p>
        <BlockMath math="\mathbf{h}_i^{(l+1)} = \sigma\!\left(\sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)} \alpha_{ij}^r \mathbf{W}_r \mathbf{h}_j^{(l)}\right)" />
        <p className="mt-2">Each relation <InlineMath math="r" /> has its own weight matrix <InlineMath math="\mathbf{W}_r" /> and
        attention parameters. Messages from different relation types are aggregated (sum, mean, or attention).</p>
      </DefinitionBlock>

      <ExampleBlock title="HAN: Hierarchical Attention">
        <p>Heterogeneous Attention Networks use two levels of attention:</p>
        <p><strong>Node-level</strong>: Attention over neighbors within each metapath.</p>
        <p><strong>Semantic-level</strong>: Attention over different metapaths to learn which relation types are most informative.</p>
        <p>For an academic graph: Author-Paper-Author and Author-Paper-Venue-Paper-Author are two different metapaths.</p>
      </ExampleBlock>

      <PythonCode
        title="Heterogeneous GNN with PyG"
        code={`import torch
from torch_geometric.nn import HeteroConv, GATConv, Linear
from torch_geometric.data import HeteroData

# Create heterogeneous graph
data = HeteroData()
data['user'].x = torch.randn(100, 16)
data['item'].x = torch.randn(200, 32)
data['user', 'buys', 'item'].edge_index = torch.randint(0, 100, (2, 500))
data['user', 'follows', 'user'].edge_index = torch.randint(0, 100, (2, 300))

# Heterogeneous convolution: different conv per edge type
class HeteroGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Project all node types to same dimension
        self.lin_user = Linear(16, 64)
        self.lin_item = Linear(32, 64)
        # Different conv for each relation
        self.conv = HeteroConv({
            ('user', 'buys', 'item'): GATConv(64, 64),
            ('user', 'follows', 'user'): GATConv(64, 64),
        }, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            'user': self.lin_user(x_dict['user']).relu(),
            'item': self.lin_item(x_dict['item']).relu(),
        }
        return self.conv(x_dict, edge_index_dict)

model = HeteroGNN()
out = model(data.x_dict, data.edge_index_dict)
print({k: v.shape for k, v in out.items()})`}
      />

      <WarningBlock title="Parameter Explosion">
        <p>
          With <InlineMath math="|\mathcal{R}|" /> relation types, each needing its own weight matrix,
          parameter count grows linearly with the number of relations. For knowledge graphs with
          hundreds of relations, use basis decomposition: <InlineMath math="\mathbf{W}_r = \sum_b a_{rb} \mathbf{B}_b" />
          where <InlineMath math="\mathbf{B}_b" /> are shared basis matrices.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Applications">
        <p>
          Heterogeneous GNNs are used for knowledge graph reasoning, recommendation systems
          (user-item-attribute graphs), drug-target interaction prediction, and academic network
          analysis. They naturally encode the rich relational structure that homogeneous GNNs flatten.
        </p>
      </NoteBlock>
    </div>
  )
}
