import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function VQVAEVisualizer() {
  const [codebookSize, setCodebookSize] = useState(8192)
  const [gridSize, setGridSize] = useState(32)
  const totalTokens = gridSize * gridSize
  const bitsPerImage = totalTokens * Math.log2(codebookSize)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">dVAE Image Tokenization</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Codebook size: {codebookSize.toLocaleString()}
          <input type="range" min={512} max={16384} step={512} value={codebookSize} onChange={e => setCodebookSize(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Grid: {gridSize}x{gridSize}
          <input type="range" min={8} max={64} step={8} value={gridSize} onChange={e => setGridSize(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm text-center">
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Image Tokens</p>
          <p className="text-lg font-bold">{totalTokens}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Codebook Size</p>
          <p className="text-lg font-bold">{codebookSize.toLocaleString()}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Bits/Image</p>
          <p className="text-lg font-bold">{(bitsPerImage / 1000).toFixed(1)}K</p>
        </div>
      </div>
    </div>
  )
}

export default function DALLEImageTokens() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DALL-E (OpenAI, 2021) pioneered text-to-image generation by treating images as sequences of
        discrete tokens and generating them autoregressively with a transformer. This approach
        unified image generation with language modeling.
      </p>

      <DefinitionBlock title="DALL-E Two-Stage Approach">
        <p><strong>Stage 1:</strong> Train a discrete VAE (dVAE) to encode images into a grid of discrete tokens:</p>
        <BlockMath math="z = \arg\min_{z_k \in \mathcal{C}} \|f_{\text{enc}}(x)_{ij} - z_k\|^2, \quad \mathcal{C} = \{z_1, \ldots, z_K\}" />
        <p className="mt-2"><strong>Stage 2:</strong> Train an autoregressive transformer on concatenated text + image tokens:</p>
        <BlockMath math="p(x|y) = \prod_{i=1}^{N} p(z_i | z_{<i}, y)" />
        <p className="mt-2">where <InlineMath math="y" /> is the text caption and <InlineMath math="z_i" /> are image tokens.</p>
      </DefinitionBlock>

      <VQVAEVisualizer />

      <ExampleBlock title="DALL-E Model Scale">
        <p>The DALL-E transformer uses 12B parameters to model the joint distribution of 256 BPE text tokens and 1024 image tokens (32x32 grid with codebook size 8192):</p>
        <BlockMath math="\text{Sequence length} = 256_{\text{text}} + 1024_{\text{image}} = 1280 \text{ tokens}" />
        <p>At inference, text tokens are provided as prefix and image tokens are sampled autoregressively.</p>
      </ExampleBlock>

      <PythonCode
        title="Simple Vector Quantization (dVAE Core)"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Vector quantization layer for image tokenization."""
    def __init__(self, num_codes=8192, code_dim=256):
        super().__init__()
        self.codebook = nn.Embedding(num_codes, code_dim)
        self.codebook.weight.data.uniform_(-1/num_codes, 1/num_codes)

    def forward(self, z_e):
        # z_e: [B, D, H, W] -> [B, H, W, D]
        z_e = z_e.permute(0, 2, 3, 1).contiguous()
        flat = z_e.view(-1, z_e.shape[-1])

        # Find nearest codebook entry
        dists = torch.cdist(flat, self.codebook.weight)
        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices).view(z_e.shape)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st.permute(0, 3, 1, 2), indices

vq = VectorQuantizer(num_codes=8192, code_dim=256)
encoded = torch.randn(2, 256, 32, 32)  # encoder output
quantized, tokens = vq(encoded)
print(f"Quantized shape: {quantized.shape}")
print(f"Image tokens: {tokens.shape}, unique codes used: {tokens.unique().numel()}")`}
      />

      <NoteBlock type="note" title="From DALL-E to DALL-E 2">
        <p>
          DALL-E 2 replaced the autoregressive approach with a diffusion model, generating CLIP
          image embeddings from text and then decoding to pixels. This produced higher-fidelity
          images but moved away from the elegant unified token-based approach.
        </p>
      </NoteBlock>

      <NoteBlock type="note" title="Modern Image Tokenizers">
        <p>
          The dVAE in DALL-E has been superseded by improved tokenizers: VQGAN uses adversarial
          training and perceptual losses for sharper reconstructions, while MAGVIT-v2 achieves
          near-lossless image compression at high compression ratios. These tokenizers are also
          used in video generation (VideoGPT, MAGVIT) and unified vision-language models that
          generate both text and images autoregressively.
        </p>
      </NoteBlock>
    </div>
  )
}
