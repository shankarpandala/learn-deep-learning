import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TaskTypeViz() {
  const [taskType, setTaskType] = useState('node')
  const nodes = [
    { x: 60, y: 50 }, { x: 160, y: 30 }, { x: 260, y: 60 },
    { x: 100, y: 130 }, { x: 210, y: 140 },
  ]
  const edges = [[0,1],[1,2],[0,3],[3,4],[1,4],[2,4]]
  const nodeColors = taskType === 'node' ? ['#7c3aed', '#f97316', '#7c3aed', '#f97316', '#7c3aed'] : Array(5).fill('#7c3aed')
  const edgeHighlight = taskType === 'edge' ? [false, false, false, true, false, false] : Array(6).fill(false)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Graph Learning Tasks</h3>
      <div className="flex items-center gap-2 mb-3">
        {['node', 'edge', 'graph'].map(t => (
          <button key={t} onClick={() => setTaskType(t)}
            className={`px-3 py-1 rounded text-sm ${t === taskType ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>
            {t}-level
          </button>
        ))}
      </div>
      <svg width={320} height={170} className="mx-auto block">
        {taskType === 'graph' && <rect x={20} y={5} width={280} height={160} rx={12} fill="none" stroke="#7c3aed" strokeWidth={2} strokeDasharray="5,3" />}
        {edges.map(([i, j], k) => (
          <line key={k} x1={nodes[i].x} y1={nodes[i].y} x2={nodes[j].x} y2={nodes[j].y}
            stroke={edgeHighlight[k] ? '#f97316' : '#d1d5db'} strokeWidth={edgeHighlight[k] ? 3 : 1.5} />
        ))}
        {nodes.map((n, i) => (
          <circle key={i} cx={n.x} cy={n.y} r={16} fill={nodeColors[i]} opacity={0.8} />
        ))}
        {taskType === 'graph' && <text x={160} y={168} textAnchor="middle" fill="#7c3aed" fontSize={10}>graph classification</text>}
        {taskType === 'edge' && <text x={155} y={155} textAnchor="middle" fill="#f97316" fontSize={10}>link prediction</text>}
        {taskType === 'node' && <text x={160} y={168} textAnchor="middle" fill="#7c3aed" fontSize={10}>node classification (2 classes)</text>}
      </svg>
    </div>
  )
}

export default function GraphTasks() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Graph learning tasks can be categorized by the level at which predictions are made:
        individual nodes, pairs of nodes (edges), or entire graphs.
      </p>

      <DefinitionBlock title="Node-Level Tasks">
        <p>Predict a label or property for each node:</p>
        <BlockMath math="\hat{y}_i = f(\mathbf{h}_i), \quad \mathbf{h}_i = \text{GNN}(\mathcal{G}, \mathbf{X})_i" />
        <p className="mt-2">Examples: citation network classification, fraud detection, protein function prediction.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Edge-Level Tasks">
        <p>Predict the existence or properties of edges (link prediction):</p>
        <BlockMath math="\hat{y}_{ij} = g(\mathbf{h}_i, \mathbf{h}_j), \quad \text{e.g., } g = \sigma(\mathbf{h}_i^\top \mathbf{h}_j)" />
        <p className="mt-2">Examples: social network friend recommendation, knowledge graph completion.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Graph-Level Tasks">
        <p>Predict a property of the entire graph using a readout function:</p>
        <BlockMath math="\hat{y} = \text{READOUT}(\{\mathbf{h}_i : v_i \in \mathcal{V}\})" />
        <p className="mt-2">Common readouts: mean/sum/max pooling, or learned hierarchical pooling.
        Examples: molecular property prediction, program analysis.</p>
      </DefinitionBlock>

      <TaskTypeViz />

      <ExampleBlock title="Benchmark Datasets">
        <p><strong>Node</strong>: Cora (2.7K papers, 7 classes), ogbn-arxiv (170K papers).</p>
        <p><strong>Edge</strong>: ogbl-collab (author collaboration), ogbl-citation2.</p>
        <p><strong>Graph</strong>: ogbg-molhiv (41K molecules), ZINC (12K molecules).</p>
      </ExampleBlock>

      <PythonCode
        title="Three Task Types in PyG"
        code={`import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MultiTaskGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, node_classes, graph_classes):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.node_head = nn.Linear(hidden_dim, node_classes)
        self.graph_head = nn.Linear(hidden_dim, graph_classes)

    def encode(self, x, edge_index):
        h = self.conv1(x, edge_index).relu()
        h = self.conv2(h, edge_index)
        return h

    def node_classification(self, x, edge_index):
        h = self.encode(x, edge_index)
        return self.node_head(h)  # (N, node_classes)

    def link_prediction(self, x, edge_index, src, dst):
        h = self.encode(x, edge_index)
        return (h[src] * h[dst]).sum(dim=-1)  # dot product score

    def graph_classification(self, x, edge_index, batch):
        h = self.encode(x, edge_index)
        h_graph = global_mean_pool(h, batch)  # (B, hidden_dim)
        return self.graph_head(h_graph)

model = MultiTaskGNN(in_dim=16, hidden_dim=64, node_classes=7, graph_classes=2)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Inductive vs Transductive">
        <p>
          <strong>Transductive</strong>: test nodes are visible during training (without labels).
          Common in node classification on a single graph.
          <strong>Inductive</strong>: the model must generalize to entirely unseen graphs.
          Graph classification is inherently inductive; GraphSAGE enables inductive node classification.
        </p>
      </NoteBlock>
    </div>
  )
}
