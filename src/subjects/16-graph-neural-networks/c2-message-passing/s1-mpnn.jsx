import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MessagePassingViz() {
  const [layer, setLayer] = useState(0)
  const nodes = [
    { x: 160, y: 40 }, { x: 60, y: 110 }, { x: 260, y: 110 },
    { x: 100, y: 190 }, { x: 220, y: 190 },
  ]
  const edges = [[0,1],[0,2],[1,3],[1,4],[2,4]]
  const receptive = [
    [[0]], [[0,1,2]], [[0,1,2,3,4]],
  ]
  const active = new Set(receptive[Math.min(layer, 2)][0] || [])
  const colors = ['#7c3aed', '#f97316', '#10b981']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Message Passing Layers (node v0)</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Layers: {layer}
          <input type="range" min={0} max={2} step={1} value={layer} onChange={e => setLayer(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <span className="text-xs text-violet-600 dark:text-violet-400">Receptive field: {active.size} nodes</span>
      </div>
      <svg width={320} height={220} className="mx-auto block">
        {edges.map(([i, j], k) => (
          <line key={k} x1={nodes[i].x} y1={nodes[i].y} x2={nodes[j].x} y2={nodes[j].y}
            stroke={active.has(i) && active.has(j) ? '#7c3aed' : '#e5e7eb'} strokeWidth={active.has(i) && active.has(j) ? 2 : 1} />
        ))}
        {nodes.map((n, i) => (
          <g key={i}>
            <circle cx={n.x} cy={n.y} r={18} fill={active.has(i) ? colors[Math.min(layer, 2)] : '#e5e7eb'} opacity={active.has(i) ? 0.8 : 0.4} />
            <text x={n.x} y={n.y + 4} textAnchor="middle" fill={active.has(i) ? 'white' : '#9ca3af'} fontSize={11} fontWeight="bold">v{i}</text>
          </g>
        ))}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">Each layer expands the receptive field by one hop</p>
    </div>
  )
}

export default function MPNN() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Message Passing Neural Network (MPNN) framework unifies most GNN architectures
        under a common aggregate-update paradigm. Each layer aggregates information from
        neighbors and updates node representations.
      </p>

      <DefinitionBlock title="Message Passing Framework">
        <p>Each layer <InlineMath math="k" /> performs two operations:</p>
        <BlockMath math="\mathbf{m}_i^{(k)} = \bigoplus_{j \in \mathcal{N}(i)} \phi\!\left(\mathbf{h}_i^{(k-1)}, \mathbf{h}_j^{(k-1)}, \mathbf{e}_{ij}\right)" />
        <BlockMath math="\mathbf{h}_i^{(k)} = \psi\!\left(\mathbf{h}_i^{(k-1)}, \mathbf{m}_i^{(k)}\right)" />
        <p className="mt-2">where <InlineMath math="\phi" /> is the message function, <InlineMath math="\bigoplus" /> is a
        permutation-invariant aggregation (sum, mean, max), and <InlineMath math="\psi" /> is the update function.</p>
      </DefinitionBlock>

      <MessagePassingViz />

      <TheoremBlock title="Expressiveness and WL Test" id="wl-test">
        <p>The Weisfeiler-Lehman (WL) graph isomorphism test provides an upper bound on GNN expressiveness:</p>
        <BlockMath math="\text{MPNN distinguishes } \mathcal{G}_1, \mathcal{G}_2 \implies \text{1-WL distinguishes } \mathcal{G}_1, \mathcal{G}_2" />
        <p className="mt-2">GIN (Graph Isomorphism Network) achieves this upper bound with sum aggregation
        and injective update functions.</p>
      </TheoremBlock>

      <ExampleBlock title="Common Instantiations">
        <p><strong>GCN</strong>: <InlineMath math="\phi(h_j) = h_j / \sqrt{d_i d_j}" />, mean-like aggregation.</p>
        <p><strong>GraphSAGE</strong>: <InlineMath math="\phi(h_j) = h_j" />, sample and aggregate (mean/LSTM/pool).</p>
        <p><strong>GAT</strong>: <InlineMath math="\phi(h_i, h_j) = \alpha_{ij} W h_j" />, learned attention weights.</p>
        <p><strong>GIN</strong>: <InlineMath math="\psi(h_i, m_i) = \text{MLP}((1+\varepsilon)h_i + m_i)" />, sum aggregation.</p>
      </ExampleBlock>

      <PythonCode
        title="Custom Message Passing Layer in PyG"
        code={`import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops

class SimpleMP(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # sum aggregation
        self.lin = nn.Linear(in_channels, out_channels)
        self.update_mlp = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.ReLU(),
        )

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x_transformed = self.lin(x)
        # Triggers message() -> aggregate() -> update()
        return self.propagate(edge_index, x=x_transformed)

    def message(self, x_j):
        return x_j  # messages are neighbor features

    def update(self, aggr_out, x):
        # Combine aggregated messages with self features
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))

layer = SimpleMP(16, 32)
x = torch.randn(5, 16)
edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
out = layer(x, edge_index)
print(f"Input: {x.shape} -> Output: {out.shape}")`}
      />

      <NoteBlock type="note" title="Over-Smoothing">
        <p>
          As the number of message passing layers increases, node representations converge
          to the same vector (over-smoothing). After <InlineMath math="K" /> layers, each node's
          representation is influenced by its <InlineMath math="K" />-hop neighborhood. Most GNNs
          use 2-4 layers. Techniques like residual connections, jumping knowledge, and DropEdge
          help mitigate this.
        </p>
      </NoteBlock>
    </div>
  )
}
