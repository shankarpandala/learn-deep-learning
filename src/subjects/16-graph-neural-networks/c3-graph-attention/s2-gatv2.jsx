import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function StaticVsDynamicViz() {
  const [version, setVersion] = useState('v1')
  const nodes = [
    { x: 160, y: 30, features: 'f1' },
    { x: 60, y: 110, features: 'f2' },
    { x: 260, y: 110, features: 'f3' },
  ]
  const v1Weights = [0.5, 0.3, 0.2]
  const v2Weights = version === 'v1' ? [0.5, 0.3, 0.2] : [0.2, 0.6, 0.2]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Static vs Dynamic Attention</h3>
      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setVersion('v1')} className={`px-3 py-1 rounded text-sm ${version === 'v1' ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>GATv1 (static)</button>
        <button onClick={() => setVersion('v2')} className={`px-3 py-1 rounded text-sm ${version === 'v2' ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>GATv2 (dynamic)</button>
      </div>
      <svg width={320} height={150} className="mx-auto block">
        {nodes.slice(1).map((n, i) => (
          <line key={i} x1={nodes[0].x} y1={nodes[0].y} x2={n.x} y2={n.y}
            stroke="#7c3aed" strokeWidth={v2Weights[i + 1] * 10} opacity={0.6} />
        ))}
        {nodes.map((n, i) => (
          <g key={i}>
            <circle cx={n.x} cy={n.y} r={20} fill={i === 0 ? '#7c3aed' : '#e5e7eb'} stroke="#7c3aed" strokeWidth={1.5} />
            <text x={n.x} y={n.y + 4} textAnchor="middle" fill={i === 0 ? 'white' : '#374151'} fontSize={10}>{n.features}</text>
          </g>
        ))}
        {nodes.slice(1).map((n, i) => (
          <text key={i} x={(nodes[0].x + n.x) / 2 + (i === 0 ? -15 : 15)} y={(nodes[0].y + n.y) / 2}
            textAnchor="middle" fill="#7c3aed" fontSize={10}>{v2Weights[i + 1].toFixed(1)}</text>
        ))}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">
        {version === 'v1' ? 'GATv1: attention ranking is fixed regardless of query node' : 'GATv2: attention ranking can change based on query features'}
      </p>
    </div>
  )
}

export default function GATv2() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        GATv2 addresses a fundamental limitation of the original GAT: its attention is "static"
        and cannot compute dynamic attention where the ranking of neighbors depends on the query node.
      </p>

      <TheoremBlock title="Static Attention Problem" id="static-attention">
        <p>In GATv1, the attention function is:</p>
        <BlockMath math="e_{ij} = \mathbf{a}^\top \text{LeakyReLU}\!\left([\mathbf{W}\mathbf{h}_i \| \mathbf{W}\mathbf{h}_j]\right)" />
        <p className="mt-2">Because <InlineMath math="\mathbf{a}" /> is split as <InlineMath math="[\mathbf{a}_L \| \mathbf{a}_R]" />,
        this becomes <InlineMath math="\mathbf{a}_L^\top \mathbf{W}\mathbf{h}_i + \mathbf{a}_R^\top \mathbf{W}\mathbf{h}_j" />.
        The contribution of key <InlineMath math="j" /> is independent of query <InlineMath math="i" />,
        so the attention ranking is the same for all queries.</p>
      </TheoremBlock>

      <DefinitionBlock title="GATv2: Dynamic Attention">
        <p>GATv2 applies the nonlinearity <strong>before</strong> the dot product with <InlineMath math="\mathbf{a}" />:</p>
        <BlockMath math="e_{ij} = \mathbf{a}^\top \text{LeakyReLU}\!\left(\mathbf{W} [\mathbf{h}_i \| \mathbf{h}_j]\right)" />
        <p className="mt-2">This allows the attention function to compute any ranking of neighbors
        as a function of the query, making it a <strong>universal approximator</strong> of attention.</p>
      </DefinitionBlock>

      <StaticVsDynamicViz />

      <ExampleBlock title="When Does It Matter?">
        <p>Consider a knowledge graph where node A is connected to nodes B and C.</p>
        <p>When predicting A's profession, neighbor B (employer) should get high attention.</p>
        <p>When predicting A's hometown, neighbor C (family) should get high attention.</p>
        <p>GATv1 always assigns the same attention ranking; GATv2 can adapt based on context.</p>
      </ExampleBlock>

      <PythonCode
        title="GATv2 Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATv2Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        # GATv2Conv applies LeakyReLU before attention dot product
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_dim * heads, out_dim, heads=1,
                               concat=False, dropout=0.6)

    def forward(self, x, edge_index, return_attention=False):
        x = F.dropout(x, p=0.6, training=self.training)
        x, attn1 = self.conv1(x, edge_index, return_attention_weights=True)
        x = x.relu()
        x = F.dropout(x, p=0.6, training=self.training)
        x, attn2 = self.conv2(x, edge_index, return_attention_weights=True)
        if return_attention:
            return x, attn1, attn2
        return x

model = GATv2Model(in_dim=16, hidden_dim=8, out_dim=7)
x = torch.randn(10, 16)
edge_index = torch.randint(0, 10, (2, 30))
out, attn1, attn2 = model(x, edge_index, return_attention=True)
print(f"Output: {out.shape}")
print(f"Attention (layer 1): edge_index {attn1[0].shape}, weights {attn1[1].shape}")`}
      />

      <NoteBlock type="note" title="Computational Cost">
        <p>
          GATv2 has the same computational complexity as GATv1 (<InlineMath math="O(|\mathcal{E}| \cdot d)" />)
          but empirically takes slightly longer due to the larger weight matrix <InlineMath math="\mathbf{W}" />.
          The expressiveness gain is well worth the small overhead on tasks requiring dynamic attention.
        </p>
      </NoteBlock>
    </div>
  )
}
