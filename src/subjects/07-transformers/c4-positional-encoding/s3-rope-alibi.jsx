import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ALiBiBiasViz() {
  const [slope, setSlope] = useState(0.25)
  const n = 6
  const cellSize = 44

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">ALiBi Attention Bias</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Slope (m): {slope.toFixed(2)}
        <input type="range" min={0.05} max={1.0} step={0.05} value={slope} onChange={e => setSlope(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <div className="flex justify-center overflow-x-auto">
        <table className="border-collapse">
          <tbody>
            {Array.from({ length: n }, (_, i) => (
              <tr key={i}>
                {Array.from({ length: n }, (_, j) => {
                  const bias = j <= i ? -slope * (i - j) : null
                  return (
                    <td key={j} className="text-center text-xs font-mono border border-gray-200 dark:border-gray-700" style={{ width: cellSize, height: cellSize, backgroundColor: bias !== null ? `rgba(139, 92, 246, ${Math.max(0.1, 1 + bias / 3)})` : 'rgba(220, 38, 38, 0.1)' }}>
                      {bias !== null ? bias.toFixed(1) : '-inf'}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-center mt-2 text-gray-500 dark:text-gray-400">Bias = -m * |i - j|. More distant positions receive larger negative bias.</p>
    </div>
  )
}

export default function RoPEALiBi() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Rotary Position Embeddings (RoPE) and Attention with Linear Biases (ALiBi) are modern
        positional encoding methods designed to enable better length generalization while encoding
        relative positional information directly into the attention computation.
      </p>

      <DefinitionBlock title="Rotary Position Embeddings (RoPE)">
        <p>RoPE applies a rotation matrix to query and key vectors based on position:</p>
        <BlockMath math="f(x_m, m) = R_{\Theta,m} x_m = \begin{pmatrix} x_1 \cos m\theta_1 - x_2 \sin m\theta_1 \\ x_1 \sin m\theta_1 + x_2 \cos m\theta_1 \\ \vdots \end{pmatrix}" />
        <p className="mt-2">
          The dot product <InlineMath math="f(q, m)^\top f(k, n)" /> depends only on the relative
          position <InlineMath math="m - n" /> and the token content, elegantly combining both.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="RoPE Relative Position Property" id="rope-relative">
        <BlockMath math="\langle f(q, m), f(k, n) \rangle = \langle R_{\Theta, n-m} q, k \rangle = g(q, k, m - n)" />
        <p className="mt-2">
          The attention score between positions <InlineMath math="m" /> and <InlineMath math="n" /> depends
          only on relative distance <InlineMath math="m - n" />, not absolute positions. This is
          achieved without any additive encoding — the position is baked into the rotation.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="ALiBi (Attention with Linear Biases)">
        <p>ALiBi adds a static, non-learned bias to attention scores proportional to distance:</p>
        <BlockMath math="\text{softmax}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}} - m \cdot |i - j|\right)" />
        <p className="mt-2">
          Each head uses a different slope <InlineMath math="m" />, set as a geometric sequence.
          No positional embeddings are added to the input — position is encoded purely via attention biases.
        </p>
      </DefinitionBlock>

      <ALiBiBiasViz />

      <ExampleBlock title="Length Generalization">
        <p>
          ALiBi trained on 1024 tokens can generalize to 2048+ tokens at inference time.
          RoPE achieves similar extrapolation, especially with techniques like NTK-aware scaling
          that adjust the base frequency. Both significantly outperform learned positional
          embeddings at unseen lengths.
        </p>
      </ExampleBlock>

      <PythonCode
        title="RoPE Implementation"
        code={`import torch

def precompute_freqs(dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)  # (max_len, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex

def apply_rope(x, freqs):
    # x: (B, H, N, D) -> view as complex pairs
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[2]].unsqueeze(0).unsqueeze(0)
    x_rotated = x_complex * freqs
    return torch.view_as_real(x_rotated).reshape_as(x)

# Example
freqs = precompute_freqs(dim=64, max_len=2048)
q = torch.randn(2, 8, 128, 64)  # (B, heads, seq, d_k)
q_rope = apply_rope(q, freqs)
print(f"RoPE output: {q_rope.shape}")  # same as input`}
      />

      <NoteBlock type="note" title="Which to Choose?">
        <p>
          <strong>RoPE</strong> is used in LLaMA, PaLM, and most modern LLMs — it preserves the
          full dot-product structure and generalizes well. <strong>ALiBi</strong> is simpler and
          has zero learnable parameters but provides a softer form of relative positioning.
          Both vastly outperform absolute positional embeddings for long-context tasks.
        </p>
      </NoteBlock>
    </div>
  )
}
