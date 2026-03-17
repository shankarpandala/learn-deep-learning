import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function TilingDiagram() {
  const [numTiles, setNumTiles] = useState(4)
  const gridSize = 200
  const tileSize = gridSize / numTiles

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Tiled Attention Computation</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Tiles per dimension: {numTiles}
        <input type="range" min={2} max={8} step={1} value={numTiles} onChange={e => setNumTiles(parseInt(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <div className="flex justify-center gap-8 items-start">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 text-center">Standard: full N x N in HBM</p>
          <svg width={gridSize} height={gridSize}>
            <rect x={0} y={0} width={gridSize} height={gridSize} fill="rgba(220, 38, 38, 0.2)" stroke="#dc2626" strokeWidth={1} rx={4} />
            <text x={gridSize / 2} y={gridSize / 2 + 4} textAnchor="middle" fontSize={11} fill="#dc2626">O(N^2) memory</text>
          </svg>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 text-center">Flash: tiles in SRAM</p>
          <svg width={gridSize} height={gridSize}>
            {Array.from({ length: numTiles * numTiles }, (_, idx) => {
              const r = Math.floor(idx / numTiles)
              const c = idx % numTiles
              return (
                <rect key={idx} x={c * tileSize + 1} y={r * tileSize + 1} width={tileSize - 2} height={tileSize - 2} rx={2} fill={`rgba(139, 92, 246, ${0.15 + (idx % 3) * 0.15})`} stroke="#8b5cf6" strokeWidth={0.5} />
              )
            })}
            <text x={gridSize / 2} y={gridSize / 2 + 4} textAnchor="middle" fontSize={11} fill="#7c3aed">O(N) memory</text>
          </svg>
        </div>
      </div>
    </div>
  )
}

export default function FlashAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Flash Attention is an IO-aware algorithm that computes exact attention without materializing
        the full <InlineMath math="N \times N" /> attention matrix in GPU high-bandwidth memory (HBM).
        By tiling the computation to fit in fast SRAM, it achieves significant speedups and memory savings.
      </p>

      <DefinitionBlock title="The Memory Bottleneck">
        <p>Standard attention requires storing the full attention matrix:</p>
        <BlockMath math="\underbrace{S = QK^\top}_{N \times N} \rightarrow \underbrace{P = \text{softmax}(S)}_{N \times N} \rightarrow \underbrace{O = PV}_{N \times d}" />
        <p className="mt-2">
          The <InlineMath math="N \times N" /> matrices <InlineMath math="S" /> and <InlineMath math="P" /> dominate
          memory for long sequences. Flash Attention never materializes them in full.
        </p>
      </DefinitionBlock>

      <TilingDiagram />

      <TheoremBlock title="Flash Attention IO Complexity" id="flash-io">
        <p>Flash Attention requires:</p>
        <BlockMath math="O\!\left(\frac{N^2 d^2}{M}\right) \text{ HBM accesses}" />
        <p className="mt-2">
          where <InlineMath math="M" /> is SRAM size. Standard attention requires <InlineMath math="O(N^2 + Nd)" /> HBM
          reads/writes. For typical SRAM sizes, Flash Attention is 2-4x faster despite doing the
          same FLOPs, because it is memory-access efficient.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Tiling and Online Softmax">
        <p>
          The key insight: softmax can be computed incrementally using the &quot;online softmax trick&quot;.
          For each tile, compute local softmax with a running maximum and accumulator, then rescale
          when processing the next tile. This avoids needing all scores simultaneously.
        </p>
        <BlockMath math="m_{\text{new}} = \max(m_{\text{old}}, \max(S_{\text{tile}})), \quad \ell_{\text{new}} = e^{m_{\text{old}} - m_{\text{new}}} \ell_{\text{old}} + \sum e^{S_{\text{tile}} - m_{\text{new}}}" />
      </ExampleBlock>

      <PythonCode
        title="Using Flash Attention in PyTorch"
        code={`import torch
import torch.nn.functional as F

# PyTorch 2.0+ includes Flash Attention via SDPA
# (Scaled Dot Product Attention)
Q = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
K = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)
V = torch.randn(2, 8, 4096, 64, device='cuda', dtype=torch.float16)

# This automatically uses Flash Attention when possible
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    output = F.scaled_dot_product_attention(Q, K, V)
    print(f"Output: {output.shape}")  # (2, 8, 4096, 64)

# Memory comparison (conceptual):
# Standard: 4096^2 * 2 bytes * 8 heads = 512 MB for attn matrix
# Flash: O(N) memory = only a few MB for tiles`}
      />

      <WarningBlock title="Hardware Requirements">
        <p>
          Flash Attention requires GPU hardware with sufficient SRAM (modern NVIDIA GPUs like
          A100/H100). It is an <em>exact</em> computation — not an approximation — but the
          implementation is hardware-specific. Always benchmark on your target hardware.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Flash Attention v2 and Beyond">
        <p>
          Flash Attention v2 further optimizes work partitioning across GPU thread blocks and
          warps, achieving near-optimal occupancy. It supports causal masking natively and is
          now the default attention backend in most deep learning frameworks.
        </p>
      </NoteBlock>
    </div>
  )
}
