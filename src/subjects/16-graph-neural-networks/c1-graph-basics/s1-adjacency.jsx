import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GraphViz() {
  const [showMatrix, setShowMatrix] = useState(false)
  const nodes = [
    { id: 0, x: 80, y: 50, label: 'v0' },
    { id: 1, x: 200, y: 30, label: 'v1' },
    { id: 2, x: 280, y: 100, label: 'v2' },
    { id: 3, x: 180, y: 140, label: 'v3' },
    { id: 4, x: 60, y: 130, label: 'v4' },
  ]
  const edges = [[0,1],[1,2],[2,3],[3,4],[4,0],[0,3]]
  const adj = Array.from({ length: 5 }, () => Array(5).fill(0))
  edges.forEach(([i, j]) => { adj[i][j] = 1; adj[j][i] = 1 })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Graph & Adjacency Matrix</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showMatrix} onChange={e => setShowMatrix(e.target.checked)} className="accent-violet-500" />
          Show adjacency matrix
        </label>
      </div>
      <div className="flex items-start gap-6 justify-center">
        <svg width={340} height={170} className="block">
          {edges.map(([i, j], k) => (
            <line key={k} x1={nodes[i].x} y1={nodes[i].y} x2={nodes[j].x} y2={nodes[j].y} stroke="#7c3aed" strokeWidth={1.5} opacity={0.4} />
          ))}
          {nodes.map(n => (
            <g key={n.id}>
              <circle cx={n.x} cy={n.y} r={18} fill="#7c3aed" opacity={0.15} stroke="#7c3aed" strokeWidth={2} />
              <text x={n.x} y={n.y + 4} textAnchor="middle" fill="#7c3aed" fontSize={12} fontWeight="bold">{n.label}</text>
            </g>
          ))}
        </svg>
        {showMatrix && (
          <table className="text-xs border-collapse font-mono">
            <thead><tr><th className="px-2 py-1" />{nodes.map(n => <th key={n.id} className="px-2 py-1 text-violet-600">{n.label}</th>)}</tr></thead>
            <tbody>{adj.map((row, i) => (
              <tr key={i}>
                <td className="px-2 py-1 text-violet-600 font-bold">{nodes[i].label}</td>
                {row.map((v, j) => <td key={j} className={`px-2 py-1 text-center ${v ? 'bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-400 font-bold' : 'text-gray-400'}`}>{v}</td>)}
              </tr>
            ))}</tbody>
          </table>
        )}
      </div>
    </div>
  )
}

export default function AdjacencyFeatureMatrices() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Graphs are the natural data structure for relational data. Before applying neural networks
        to graphs, we need efficient representations of graph topology and node/edge features.
      </p>

      <DefinitionBlock title="Graph Definition">
        <p>A graph <InlineMath math="\mathcal{G} = (\mathcal{V}, \mathcal{E})" /> consists of:</p>
        <BlockMath math="\mathcal{V} = \{v_1, \ldots, v_N\} \text{ (nodes)}, \quad \mathcal{E} \subseteq \mathcal{V} \times \mathcal{V} \text{ (edges)}" />
        <p className="mt-2">Each node <InlineMath math="v_i" /> has a feature vector <InlineMath math="\mathbf{x}_i \in \mathbb{R}^d" />.
        The full feature matrix is <InlineMath math="\mathbf{X} \in \mathbb{R}^{N \times d}" />.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Adjacency Matrix">
        <BlockMath math="A_{ij} = \begin{cases} 1 & \text{if } (v_i, v_j) \in \mathcal{E} \\ 0 & \text{otherwise} \end{cases}" />
        <p className="mt-2">The degree matrix: <InlineMath math="D_{ii} = \sum_j A_{ij}" />.
        For undirected graphs, <InlineMath math="A = A^\top" />.</p>
      </DefinitionBlock>

      <GraphViz />

      <ExampleBlock title="Common Graph Representations">
        <p><strong>Adjacency matrix</strong>: Dense <InlineMath math="O(N^2)" /> storage. Good for small, dense graphs.</p>
        <p><strong>Edge list</strong>: Store pairs <InlineMath math="(i, j)" />. <InlineMath math="O(|\mathcal{E}|)" /> space.</p>
        <p><strong>COO format</strong>: Two arrays of source and destination indices. Used by PyG and DGL.</p>
        <p>Most real-world graphs are sparse (<InlineMath math="|\mathcal{E}| \ll N^2" />), making sparse formats preferred.</p>
      </ExampleBlock>

      <PythonCode
        title="Graph Representations with PyTorch Geometric"
        code={`import torch
from torch_geometric.data import Data

# Define a small graph: 5 nodes, 6 edges
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 3],  # source
    [1, 0, 2, 1, 3, 2, 4, 3, 0, 4, 3, 0],  # target
], dtype=torch.long)

# Node features: 5 nodes, 3 features each
x = torch.randn(5, 3)

# Create a PyG Data object
data = Data(x=x, edge_index=edge_index)
print(f"Nodes: {data.num_nodes}, Edges: {data.num_edges}")
print(f"Node features shape: {data.x.shape}")
print(f"Is undirected: {data.is_undirected()}")
print(f"Average degree: {data.num_edges / data.num_nodes:.1f}")

# Convert to adjacency matrix (dense) for inspection
adj = torch.zeros(5, 5)
adj[edge_index[0], edge_index[1]] = 1
print(f"Adjacency matrix:\\n{adj}")`}
      />

      <NoteBlock type="note" title="Heterogeneous Graphs">
        <p>
          Many real-world graphs have multiple node and edge types (e.g., users and products
          connected by "purchased" and "reviewed" edges). These require typed adjacency matrices
          or separate edge stores per relation type.
        </p>
      </NoteBlock>
    </div>
  )
}
