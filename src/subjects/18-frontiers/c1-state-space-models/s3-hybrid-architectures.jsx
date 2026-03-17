import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function HybridLayerBuilder() {
  const [totalLayers, setTotalLayers] = useState(32)
  const [attnRatio, setAttnRatio] = useState(0.25)
  const attnLayers = Math.round(totalLayers * attnRatio)
  const ssmLayers = totalLayers - attnLayers

  const layerPattern = Array.from({ length: totalLayers }, (_, i) => {
    const attnInterval = attnLayers > 0 ? Math.round(totalLayers / attnLayers) : totalLayers + 1
    return (i + 1) % attnInterval === 0 ? 'attn' : 'ssm'
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Hybrid Architecture Layer Configuration</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Total layers: {totalLayers}
          <input type="range" min={8} max={64} step={4} value={totalLayers} onChange={e => setTotalLayers(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Attention ratio: {(attnRatio * 100).toFixed(0)}%
          <input type="range" min={0} max={0.5} step={0.05} value={attnRatio} onChange={e => setAttnRatio(parseFloat(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="flex gap-0.5 flex-wrap mb-2">
        {layerPattern.map((type, i) => (
          <div key={i} className={`w-5 h-5 rounded-sm text-[7px] flex items-center justify-center ${type === 'attn' ? 'bg-violet-500 text-white' : 'bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300'}`}>
            {type === 'attn' ? 'A' : 'S'}
          </div>
        ))}
      </div>
      <p className="text-xs text-gray-500">{ssmLayers} SSM layers + {attnLayers} attention layers. Attention provides in-context recall; SSMs handle long-range dependencies efficiently.</p>
    </div>
  )
}

export default function HybridSSMAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Hybrid architectures combine SSM layers with sparse attention layers, getting the efficiency
        of SSMs for most computation while retaining attention's ability for precise in-context
        recall. This approach powers models like Jamba, Zamba, and Mamba-2.
      </p>

      <DefinitionBlock title="Hybrid SSM-Attention Block">
        <p>A hybrid model interleaves SSM and attention layers, often with a ratio of ~6:1 SSM to attention:</p>
        <BlockMath math="h_l = \begin{cases} \text{SSM}(h_{l-1}) + h_{l-1} & \text{if } l \notin \mathcal{A} \\ \text{Attn}(h_{l-1}) + h_{l-1} & \text{if } l \in \mathcal{A} \end{cases}" />
        <p className="mt-2">where <InlineMath math="\mathcal{A}" /> is the set of attention layers (e.g., every 6th layer). Both paths use pre-norm and residual connections.</p>
      </DefinitionBlock>

      <HybridLayerBuilder />

      <ExampleBlock title="Why Hybrids Outperform Pure Models">
        <p>SSMs and attention have complementary strengths:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>SSMs excel at:</strong> Long-range dependencies, efficient training, linear-time inference</li>
          <li><strong>SSMs struggle with:</strong> Precise copying, in-context learning, associative recall</li>
          <li><strong>Attention excels at:</strong> Exact retrieval, in-context learning, copying patterns</li>
          <li><strong>Attention struggles with:</strong> Long sequences (quadratic cost), generation speed</li>
          <li>Jamba (AI21): 52B MoE + Mamba hybrid, outperforms Mixtral 8x7B</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Hybrid SSM-Attention Model"
        code={`import torch
import torch.nn as nn

class SSMLayer(nn.Module):
    """Placeholder SSM layer (Mamba-style)."""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)  # Simplified SSM
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.linear(self.norm(x)))

class AttentionLayer(nn.Module):
    """Standard multi-head attention layer."""
    def __init__(self, dim, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        h = self.norm(x)
        out, _ = self.attn(h, h, h)
        return out

class HybridModel(nn.Module):
    """Hybrid SSM-Attention model with configurable ratio."""
    def __init__(self, dim=512, num_layers=24, attn_every=6):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if (i + 1) % attn_every == 0:
                self.layers.append(('attn', AttentionLayer(dim)))
            else:
                self.layers.append(('ssm', SSMLayer(dim)))
        self.layers = nn.ModuleList([l[1] for l in self.layers])
        self.layer_types = ['attn' if (i+1) % attn_every == 0 else 'ssm'
                           for i in range(num_layers)]

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)  # Residual connection
        return x

model = HybridModel(dim=256, num_layers=24, attn_every=6)
x = torch.randn(2, 128, 256)
out = model(x)
ssm_count = sum(1 for t in model.layer_types if t == 'ssm')
attn_count = sum(1 for t in model.layer_types if t == 'attn')
print(f"Layers: {ssm_count} SSM + {attn_count} Attention = {ssm_count + attn_count} total")
print(f"Output: {out.shape}")`}
      />

      <NoteBlock type="note" title="The Attention Tax">
        <p>
          Even a small number of attention layers significantly improves in-context learning and
          retrieval performance. Research shows that 4 attention layers in a 24-layer model (17%)
          recovers nearly all of a pure Transformer's in-context learning ability, while keeping
          83% of the inference speed advantage from SSM layers.
        </p>
      </NoteBlock>
    </div>
  )
}
