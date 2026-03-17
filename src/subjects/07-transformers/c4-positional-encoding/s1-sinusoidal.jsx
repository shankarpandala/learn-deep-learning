import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function SinusoidalViz() {
  const [dim, setDim] = useState(8)
  const [maxPos, setMaxPos] = useState(20)
  const W = 360, H = 180

  function pe(pos, i, d) {
    const angle = pos / Math.pow(10000, (2 * Math.floor(i / 2)) / d)
    return i % 2 === 0 ? Math.sin(angle) : Math.cos(angle)
  }

  const colors = ['#8b5cf6', '#a78bfa', '#c4b5fd', '#7c3aed', '#6d28d9', '#5b21b6', '#ddd6fe', '#4c1d95']
  const positions = Array.from({ length: maxPos }, (_, i) => i)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Sinusoidal Positional Encoding</h3>
      <div className="flex gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Dimensions: {dim}
          <input type="range" min={4} max={8} step={2} value={dim} onChange={e => setDim(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Positions: {maxPos}
          <input type="range" min={10} max={40} step={5} value={maxPos} onChange={e => setMaxPos(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={30} y1={H / 2} x2={W} y2={H / 2} stroke="#d1d5db" strokeWidth={0.5} />
        {Array.from({ length: dim }, (_, d_i) => {
          const path = positions.map((p, idx) => {
            const x = 30 + (p / (maxPos - 1)) * (W - 40)
            const y = H / 2 - pe(p, d_i, dim) * (H / 2 - 10)
            return `${idx === 0 ? 'M' : 'L'}${x},${y}`
          }).join(' ')
          return <path key={d_i} d={path} fill="none" stroke={colors[d_i % colors.length]} strokeWidth={1.5} opacity={0.8} />
        })}
      </svg>
      <p className="text-xs text-center mt-1 text-gray-500 dark:text-gray-400">Each curve is one dimension of the encoding; different frequencies capture position at different scales.</p>
    </div>
  )
}

export default function SinusoidalEncoding() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Since transformers process all positions in parallel without recurrence, they need an explicit
        mechanism to encode position information. Vaswani et al. (2017) introduced sinusoidal positional
        encodings, which are added to the input embeddings.
      </p>

      <DefinitionBlock title="Sinusoidal Positional Encoding">
        <BlockMath math="PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)" />
        <BlockMath math="PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i / d_{\text{model}}}}\right)" />
        <p className="mt-2">
          Each dimension corresponds to a sinusoid with wavelength forming a geometric progression
          from <InlineMath math="2\pi" /> to <InlineMath math="10000 \cdot 2\pi" />.
        </p>
      </DefinitionBlock>

      <SinusoidalViz />

      <TheoremBlock title="Relative Position as Linear Transformation" id="pe-relative">
        <p>For any fixed offset <InlineMath math="k" />, there exists a linear transformation <InlineMath math="M_k" /> such that:</p>
        <BlockMath math="PE_{pos+k} = M_k \cdot PE_{pos}" />
        <p className="mt-2">
          This means the model can learn to attend to relative positions through linear operations
          on the sinusoidal encodings, a key property that enables length generalization.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Encoding Properties">
        <p>
          The dot product <InlineMath math="PE_{pos} \cdot PE_{pos+k}" /> depends only on the
          offset <InlineMath math="k" />, not the absolute position. Nearby positions have
          higher similarity, and the similarity decreases smoothly with distance.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Sinusoidal Positional Encoding"
        code={`import torch
import math

def sinusoidal_pe(max_len, d_model):
    """Generate sinusoidal positional encoding."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# Visualize: nearby positions have similar encodings
pe = sinusoidal_pe(50, 128)
print(f"Shape: {pe.shape}")  # (50, 128)

# Similarity between positions
sim = torch.cosine_similarity(pe[10].unsqueeze(0), pe, dim=-1)
print(f"Sim(pos=10, pos=10): {sim[10]:.4f}")  # 1.0
print(f"Sim(pos=10, pos=11): {sim[11]:.4f}")  # ~0.98
print(f"Sim(pos=10, pos=40): {sim[40]:.4f}")  # ~0.5`}
      />

      <NoteBlock type="note" title="Fixed vs Learned">
        <p>
          Sinusoidal encodings are fixed (no learnable parameters) and can theoretically generalize
          to longer sequences than seen during training. In practice, the original Transformer paper
          found no significant difference between sinusoidal and learned positional embeddings,
          but sinusoidal encodings use zero additional parameters.
        </p>
      </NoteBlock>
    </div>
  )
}
