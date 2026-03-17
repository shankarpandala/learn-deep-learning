import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PatchEmbeddingDemo() {
  const [patchSize, setPatchSize] = useState(4)
  const gridDim = 16
  const cellSize = 14
  const W = gridDim * cellSize
  const numPatches = Math.floor(gridDim / patchSize)

  const colors = ['#8b5cf6', '#f97316', '#22c55e', '#ef4444', '#3b82f6', '#eab308', '#ec4899', '#14b8a6',
    '#a855f7', '#f59e0b', '#10b981', '#f43f5e', '#6366f1', '#d97706', '#059669', '#e11d48']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Image Patch Embedding</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Patch size: {patchSize}x{patchSize}
        <input type="range" min={2} max={8} step={2} value={patchSize} onChange={e => setPatchSize(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        <span className="ml-2">= {numPatches * numPatches} tokens</span>
      </label>
      <svg width={W} height={W} className="mx-auto block">
        {Array.from({ length: numPatches }).map((_, r) =>
          Array.from({ length: numPatches }).map((_, c) => (
            <rect key={`${r}-${c}`} x={c * patchSize * cellSize} y={r * patchSize * cellSize}
              width={patchSize * cellSize - 1} height={patchSize * cellSize - 1}
              fill={colors[(r * numPatches + c) % colors.length]} opacity={0.3}
              stroke={colors[(r * numPatches + c) % colors.length]} strokeWidth={1.5} rx={2} />
          ))
        )}
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500">
        Each colored patch becomes a token via linear projection
      </p>
    </div>
  )
}

export default function ViT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Vision Transformer (ViT) applies the transformer architecture directly to image patches,
        demonstrating that pure attention-based models can match or exceed CNNs for image classification.
      </p>

      <DefinitionBlock title="ViT Architecture">
        <p>
          An image <InlineMath math="x \in \mathbb{R}^{H \times W \times C}" /> is split into
          <InlineMath math="N = HW/P^2" /> patches of size <InlineMath math="P \times P" />:
        </p>
        <BlockMath math="z_0 = [\mathbf{x}_{\text{cls}};\; x_1\mathbf{E};\; x_2\mathbf{E};\; \ldots;\; x_N\mathbf{E}] + \mathbf{E}_{\text{pos}}" />
        <p className="mt-2">
          where <InlineMath math="\mathbf{E} \in \mathbb{R}^{P^2 C \times D}" /> is the patch
          embedding projection and <InlineMath math="\mathbf{E}_{\text{pos}}" /> are learned positional embeddings.
        </p>
      </DefinitionBlock>

      <PatchEmbeddingDemo />

      <TheoremBlock title="Self-Attention Complexity" id="vit-complexity">
        <p>
          Self-attention over <InlineMath math="N" /> patch tokens has quadratic complexity:
        </p>
        <BlockMath math="\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V" />
        <BlockMath math="\mathcal{O}(N^2 \cdot D) = \mathcal{O}\!\left(\frac{H^2 W^2}{P^4} \cdot D\right)" />
        <p className="mt-1">
          Larger patch sizes reduce sequence length but lose spatial resolution.
          ViT-B/16 uses <InlineMath math="P=16" /> yielding <InlineMath math="N=196" /> tokens for 224x224 images.
        </p>
      </TheoremBlock>

      <ExampleBlock title="ViT Model Variants">
        <ul className="list-disc ml-5 space-y-1">
          <li><strong>ViT-B/16</strong>: 12 layers, 768 dim, 12 heads, 86M params (81.8% ImageNet)</li>
          <li><strong>ViT-L/16</strong>: 24 layers, 1024 dim, 16 heads, 307M params (85.2%)</li>
          <li><strong>ViT-H/14</strong>: 32 layers, 1280 dim, 16 heads, 632M params (88.6% w/ JFT)</li>
        </ul>
        <p className="mt-1">
          ViTs require large-scale pretraining (JFT-300M) or strong regularization to match CNNs on ImageNet alone.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Vision Transformer from Scratch"
        code={`import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(3, dim, patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches + 1, dim))

    def forward(self, x):
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        return x + self.pos_embed

class ViT(nn.Module):
    def __init__(self, num_classes=1000, dim=768, depth=12,
                 heads=12, patch_size=16):
        super().__init__()
        self.embed = PatchEmbedding(patch_size=patch_size, dim=dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.head(x[:, 0])  # CLS token output`}
      />

      <NoteBlock type="note" title="CNN vs Transformer Inductive Biases">
        <p>
          CNNs have built-in translation equivariance and locality. ViTs lack these biases,
          learning spatial structure entirely from data. This makes ViTs more flexible but
          data-hungry. Hybrid approaches (CNN stem + transformer layers) combine the
          efficiency of convolutions at early stages with the global reasoning of attention.
        </p>
      </NoteBlock>
    </div>
  )
}
