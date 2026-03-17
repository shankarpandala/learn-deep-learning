import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SwinWindowDemo() {
  const [shifted, setShifted] = useState(false)
  const gridSize = 8
  const windowSize = 4
  const cellSize = 28
  const W = gridSize * cellSize
  const offset = shifted ? windowSize / 2 : 0

  const getWindowIdx = (r, c) => {
    const wr = Math.floor(((r + offset) % gridSize) / windowSize)
    const wc = Math.floor(((c + offset) % gridSize) / windowSize)
    return wr * 2 + wc
  }
  const windowColors = ['#8b5cf6', '#f97316', '#22c55e', '#3b82f6']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Swin Transformer Windows</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <input type="checkbox" checked={shifted} onChange={e => setShifted(e.target.checked)} className="accent-violet-500" />
        Apply window shift (SW-MSA)
      </label>
      <svg width={W} height={W} className="mx-auto block">
        {Array.from({ length: gridSize }).map((_, r) =>
          Array.from({ length: gridSize }).map((_, c) => (
            <rect key={`${r}-${c}`} x={c * cellSize} y={r * cellSize}
              width={cellSize - 1} height={cellSize - 1}
              fill={windowColors[getWindowIdx(r, c)]} opacity={0.25}
              stroke={windowColors[getWindowIdx(r, c)]} strokeWidth={1} rx={1} />
          ))
        )}
        {!shifted && Array.from({ length: 2 }).map((_, i) => (
          <g key={i}>
            <line x1={0} y1={(i + 1) * windowSize * cellSize} x2={W} y2={(i + 1) * windowSize * cellSize} stroke="#374151" strokeWidth={2} />
            <line x1={(i + 1) * windowSize * cellSize} y1={0} x2={(i + 1) * windowSize * cellSize} y2={W} stroke="#374151" strokeWidth={2} />
          </g>
        ))}
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500">
        {shifted ? 'Shifted windows enable cross-window connections' : 'Regular non-overlapping windows'}
      </p>
    </div>
  )
}

export default function DeiTSwin() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DeiT brings data-efficient training to ViT through distillation, while Swin Transformer
        introduces hierarchical vision with shifted windows, making transformers practical for dense prediction tasks.
      </p>

      <DefinitionBlock title="DeiT: Data-Efficient Image Transformer">
        <p>
          DeiT adds a distillation token alongside the CLS token to learn from a CNN teacher:
        </p>
        <BlockMath math="z_0 = [\mathbf{x}_{\text{cls}};\; \mathbf{x}_{\text{dist}};\; x_1\mathbf{E};\; \ldots;\; x_N\mathbf{E}] + \mathbf{E}_{\text{pos}}" />
        <p className="mt-2">
          The distillation loss combines hard label and teacher supervision:
        </p>
        <BlockMath math="\mathcal{L} = \frac{1}{2}\mathcal{L}_{\text{CE}}(y, \psi(z_{\text{cls}})) + \frac{1}{2}\mathcal{L}_{\text{CE}}(y_t, \psi(z_{\text{dist}}))" />
      </DefinitionBlock>

      <SwinWindowDemo />

      <TheoremBlock title="Shifted Window Attention" id="swin-attention">
        <p>
          Swin computes self-attention within local windows of size <InlineMath math="M \times M" />:
        </p>
        <BlockMath math="\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2hwC" />
        <BlockMath math="\Omega(\text{Global MSA}) = 4hwC^2 + 2(hw)^2C" />
        <p className="mt-1">
          Window attention is linear in image size (<InlineMath math="hw" />) vs quadratic for global attention.
          Shifted windows in alternating layers provide cross-window connections.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Swin Hierarchical Stages">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>Stage 1</strong>: <InlineMath math="H/4 \times W/4" />, dim=96, 2 blocks</li>
          <li><strong>Stage 2</strong>: <InlineMath math="H/8 \times W/8" />, dim=192, 2 blocks (patch merge 2x2)</li>
          <li><strong>Stage 3</strong>: <InlineMath math="H/16 \times W/16" />, dim=384, 6 blocks</li>
          <li><strong>Stage 4</strong>: <InlineMath math="H/32 \times W/32" />, dim=768, 2 blocks</li>
        </ul>
        <p className="mt-1">This mimics the multi-scale structure of CNN backbones like ResNet.</p>
      </ExampleBlock>

      <PythonCode
        title="Swin Transformer Block"
        code={`import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        # Relative position bias
        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2*window_size-1) * (2*window_size-1), num_heads))

    def forward(self, x):
        B, N, C = x.shape  # N = window_size^2
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads) ** 0.5
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=7, shift=False):
        super().__init__()
        self.shift = shift
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))  # W-MSA or SW-MSA
        x = x + self.mlp(self.norm2(x))
        return x`}
      />

      <NoteBlock type="note" title="DeiT Training Recipe">
        <p>
          DeiT achieves 83.1% ImageNet accuracy training only on ImageNet-1K (no JFT) by
          using aggressive augmentation (RandAugment, CutMix, Mixup), regularization
          (stochastic depth, repeated augmentation), and knowledge distillation from a
          RegNetY teacher. This recipe made ViTs accessible to researchers without massive datasets.
        </p>
      </NoteBlock>
    </div>
  )
}
