import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function AttentionModeViz() {
  const [mode, setMode] = useState('sparse')
  const nodes = [
    { x: 80, y: 50 }, { x: 180, y: 30 }, { x: 280, y: 50 },
    { x: 130, y: 130 }, { x: 230, y: 130 },
  ]
  const sparseEdges = [[0,1],[1,2],[0,3],[3,4],[1,4]]
  const fullEdges = []
  for (let i = 0; i < 5; i++) for (let j = i + 1; j < 5; j++) fullEdges.push([i, j])
  const edges = mode === 'sparse' ? sparseEdges : fullEdges

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Sparse vs Full Attention</h3>
      <div className="flex items-center gap-2 mb-3">
        <button onClick={() => setMode('sparse')} className={`px-3 py-1 rounded text-sm ${mode === 'sparse' ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>GNN (neighbors only)</button>
        <button onClick={() => setMode('full')} className={`px-3 py-1 rounded text-sm ${mode === 'full' ? 'bg-violet-500 text-white' : 'bg-gray-100 dark:bg-gray-800 text-gray-600'}`}>Graph Transformer (all pairs)</button>
      </div>
      <svg width={360} height={160} className="mx-auto block">
        {edges.map(([i, j], k) => (
          <line key={k} x1={nodes[i].x} y1={nodes[i].y} x2={nodes[j].x} y2={nodes[j].y}
            stroke={mode === 'sparse' ? '#7c3aed' : '#f97316'} strokeWidth={1.5} opacity={0.4} />
        ))}
        {nodes.map((n, i) => (
          <circle key={i} cx={n.x} cy={n.y} r={16} fill={mode === 'sparse' ? '#7c3aed' : '#f97316'} opacity={0.8} />
        ))}
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">
        {mode === 'sparse' ? `${sparseEdges.length} edges (O(|E|))` : `${fullEdges.length} edges (O(N^2))`} -- full attention captures long-range but costs more
      </p>
    </div>
  )
}

export default function GraphTransformerArchitecture() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Graph Transformers apply full self-attention to graph nodes, overcoming the limited
        receptive field of message-passing GNNs. The key challenge is encoding graph structure
        into the attention mechanism.
      </p>

      <DefinitionBlock title="Graph Transformer Layer">
        <p>Full self-attention with structural bias:</p>
        <BlockMath math="\text{Attn}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + B\right) V" />
        <p className="mt-2">where <InlineMath math="B \in \mathbb{R}^{N \times N}" /> is a structural bias
        encoding graph topology (e.g., shortest-path distances, edge features, or spatial encoding).</p>
      </DefinitionBlock>

      <AttentionModeViz />

      <TheoremBlock title="Structural Encoding via Attention Bias" id="structural-bias">
        <p>The bias term <InlineMath math="B_{ij}" /> can encode various structural features:</p>
        <BlockMath math="B_{ij} = \phi_\text{dist}(d(i,j)) + \phi_\text{edge}(e_{ij}) + \phi_\text{degree}(\deg(i), \deg(j))" />
        <p className="mt-2">Graphormer uses centrality encoding (degree), spatial encoding (shortest path),
        and edge encoding. This allows the Transformer to be aware of graph structure without being
        limited to local neighborhoods.</p>
      </TheoremBlock>

      <ExampleBlock title="Graphormer (Ying et al., 2021)">
        <p>Graphormer won 1st place in the OGB Large-Scale Challenge using three structural encodings:</p>
        <p><strong>Centrality</strong>: <InlineMath math="h_i^{(0)} = x_i + z^-_{\deg^-(i)} + z^+_{\deg^+(i)}" /></p>
        <p><strong>Spatial</strong>: <InlineMath math="B_{ij} = b_{\phi(v_i, v_j)}" /> where <InlineMath math="\phi" /> is shortest-path distance.</p>
        <p><strong>Edge</strong>: Average edge features along the shortest path between nodes.</p>
      </ExampleBlock>

      <PythonCode
        title="Graph Transformer Layer"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )

    def forward(self, x, attn_bias=None):
        """
        x: (B, N, d_model) node features
        attn_bias: (B*H, N, N) structural bias (optional)
        """
        # Self-attention with structural bias
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=attn_bias)
        x = x + h

        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        return x

# Example
layer = GraphTransformerLayer(d_model=64, n_heads=8)
x = torch.randn(2, 10, 64)  # batch of 2, 10 nodes, dim 64
out = layer(x)
print(f"Input: {x.shape} -> Output: {out.shape}")`}
      />

      <NoteBlock type="note" title="Scalability Challenge">
        <p>
          Full self-attention is <InlineMath math="O(N^2)" /> in the number of nodes, limiting graph
          transformers to small/medium graphs (thousands of nodes). For larger graphs, sparse attention
          patterns, neighborhood sampling, or hybrid approaches (GPS) that combine local MPNN with
          global attention are necessary.
        </p>
      </NoteBlock>
    </div>
  )
}
