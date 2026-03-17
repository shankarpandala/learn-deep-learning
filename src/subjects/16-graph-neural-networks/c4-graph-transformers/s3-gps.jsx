import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GPSBlockViz() {
  const [showParts, setShowParts] = useState({ mpnn: true, attn: true, ffn: true })

  const toggle = (key) => setShowParts(prev => ({ ...prev, [key]: !prev[key] }))
  const parts = [
    { key: 'mpnn', label: 'Local MPNN', color: '#7c3aed', desc: 'Neighbor aggregation' },
    { key: 'attn', label: 'Global Attn', color: '#f97316', desc: 'Full self-attention' },
    { key: 'ffn', label: 'FFN', color: '#10b981', desc: 'Feed-forward' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">GPS Layer Architecture</h3>
      <div className="flex items-center gap-2 mb-3">
        {parts.map(p => (
          <label key={p.key} className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400">
            <input type="checkbox" checked={showParts[p.key]} onChange={() => toggle(p.key)} className="accent-violet-500" />
            {p.label}
          </label>
        ))}
      </div>
      <div className="flex items-center gap-2 justify-center">
        <div className="px-3 py-2 rounded bg-gray-100 dark:bg-gray-800 text-xs text-center border">Input h</div>
        <span className="text-gray-400">&#8594;</span>
        {parts.filter(p => showParts[p.key]).map((p, i) => (
          <div key={p.key} className="flex items-center gap-2">
            <div className="px-3 py-2 rounded text-xs text-center text-white font-medium" style={{ backgroundColor: p.color }}>
              {p.label}
              <div className="text-[10px] opacity-80">{p.desc}</div>
            </div>
            <span className="text-gray-400">{i < parts.filter(pp => showParts[pp.key]).length - 1 ? '+' : '&#8594;'}</span>
          </div>
        ))}
        <div className="px-3 py-2 rounded bg-gray-100 dark:bg-gray-800 text-xs text-center border">Output h'</div>
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">GPS combines local and global processing in each layer</p>
    </div>
  )
}

export default function GPS() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        GPS (General, Powerful, Scalable) is a hybrid architecture that combines local message
        passing with global self-attention in each layer, achieving state-of-the-art results
        while remaining scalable.
      </p>

      <DefinitionBlock title="GPS Layer">
        <p>Each GPS layer applies MPNN and Transformer attention in parallel:</p>
        <BlockMath math="\mathbf{h}_i' = \mathbf{h}_i + \underbrace{\text{MPNN}(\mathbf{h}_i, \{\mathbf{h}_j : j \in \mathcal{N}(i)\})}_{\text{local}} + \underbrace{\text{Attn}(\mathbf{h}_i, \mathbf{H})}_{\text{global}}" />
        <BlockMath math="\mathbf{h}_i'' = \mathbf{h}_i' + \text{FFN}(\text{Norm}(\mathbf{h}_i'))" />
        <p className="mt-2">The MPNN captures local graph structure; global attention captures long-range dependencies.</p>
      </DefinitionBlock>

      <GPSBlockViz />

      <TheoremBlock title="Why Hybrid?" id="hybrid-motivation">
        <p>Pure MPNN: limited to <InlineMath math="K" />-hop neighborhoods after <InlineMath math="K" /> layers. Misses long-range interactions.</p>
        <p>Pure Transformer: <InlineMath math="O(N^2)" /> cost, may ignore useful graph structure.</p>
        <p className="mt-2">GPS combines both: the MPNN provides strong structural bias and the Transformer
        provides long-range information flow, with each compensating the other's weakness.</p>
      </TheoremBlock>

      <ExampleBlock title="GPS Recipe">
        <p>The recommended GPS configuration:</p>
        <p><strong>PE</strong>: Laplacian PE (k=16) + RWSE (k=16), processed by a small MLP.</p>
        <p><strong>MPNN</strong>: GatedGCN or GINE for the local component.</p>
        <p><strong>Attention</strong>: Standard multi-head self-attention (Performer for large graphs).</p>
        <p><strong>Depth</strong>: 10-16 layers with pre-norm and residual connections.</p>
        <p>GPS achieves state-of-the-art on ZINC, PCQM4Mv2, and Peptides benchmarks.</p>
      </ExampleBlock>

      <PythonCode
        title="GPS Layer Implementation"
        code={`import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, GPSConv

class GPSModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, pe_dim=16, n_layers=6):
        super().__init__()
        self.pe_encoder = nn.Linear(pe_dim, hidden_dim)
        self.node_encoder = nn.Linear(in_dim, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            # Local MPNN: GIN with edge features
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_model = GINEConv(local_nn)
            # GPS wraps local MPNN + global attention
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_model,
                heads=4,
                dropout=0.1,
                attn_type='multihead',  # or 'performer' for O(N)
            )
            self.layers.append(gps_layer)

        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, pe, edge_index, edge_attr, batch):
        h = self.node_encoder(x) + self.pe_encoder(pe)
        for layer in self.layers:
            h = layer(h, edge_index, batch, edge_attr=edge_attr)
        # Graph-level readout
        from torch_geometric.nn import global_mean_pool
        h_graph = global_mean_pool(h, batch)
        return self.output(h_graph)

model = GPSModel(in_dim=16, hidden_dim=64, out_dim=1)
print(f"GPS params: {sum(p.numel() for p in model.parameters()):,}")`}
      />

      <NoteBlock type="note" title="Scalability with Linear Attention">
        <p>
          For graphs with thousands of nodes, the <InlineMath math="O(N^2)" /> attention becomes a bottleneck.
          GPS supports Performer attention (<InlineMath math="O(N)" />) as a drop-in replacement.
          This makes GPS scalable to large molecular graphs and biological networks while retaining
          the benefits of global information flow.
        </p>
      </NoteBlock>
    </div>
  )
}
