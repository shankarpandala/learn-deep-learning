import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PerceiverResamplerViz() {
  const [numLatents, setNumLatents] = useState(64)
  const [numVisual, setNumVisual] = useState(256)
  const ratio = (numLatents / numVisual * 100).toFixed(1)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Perceiver Resampler Compression</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Visual tokens: {numVisual}
          <input type="range" min={64} max={576} step={64} value={numVisual} onChange={e => setNumVisual(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Latent queries: {numLatents}
          <input type="range" min={8} max={128} step={8} value={numLatents} onChange={e => setNumLatents(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex items-center gap-4">
        <div className="flex-1 h-6 bg-violet-100 dark:bg-violet-900/30 rounded overflow-hidden relative">
          <div className="h-full bg-violet-400" style={{ width: '100%' }} />
          <span className="absolute inset-0 flex items-center justify-center text-xs font-medium">{numVisual} visual tokens</span>
        </div>
        <span className="text-gray-400 text-lg">&rarr;</span>
        <div className="h-6 bg-violet-500 rounded flex items-center justify-center text-xs text-white font-medium px-2" style={{ width: `${Math.max(ratio, 15)}%`, minWidth: '80px' }}>
          {numLatents} latents
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Compression ratio: {ratio}% — reduces compute for cross-attention in LM layers</p>
    </div>
  )
}

export default function FlamingoVisualPrompting() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Flamingo by DeepMind interleaves visual and text tokens, enabling few-shot multimodal learning.
        It uses a frozen vision encoder and a frozen language model, connected by lightweight
        cross-attention layers and a Perceiver Resampler.
      </p>

      <DefinitionBlock title="Perceiver Resampler">
        <p>The Perceiver Resampler compresses variable-length visual features into a fixed number of latent tokens using cross-attention:</p>
        <BlockMath math="z = \text{CrossAttn}(Q=\text{latents}, K=V=\text{visual\_tokens})" />
        <p className="mt-2">where latents are <InlineMath math="N_q" /> learned queries (typically 64) and visual tokens come from the frozen vision encoder. This produces a fixed-size representation regardless of image resolution.</p>
      </DefinitionBlock>

      <PerceiverResamplerViz />

      <ExampleBlock title="Flamingo Few-Shot Performance">
        <p>With just 4 image-text examples as context (4-shot), Flamingo-80B achieves:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>VQAv2: 67.6% (vs. 56.3% zero-shot)</li>
          <li>COCO captioning: CIDEr 113.4</li>
          <li>OK-VQA: 57.8% (requires external knowledge)</li>
        </ul>
        <p className="mt-2">Key insight: frozen pretrained models + lightweight adapters = strong few-shot multimodal learning.</p>
      </ExampleBlock>

      <PythonCode
        title="Gated Cross-Attention (Flamingo-Style)"
        code={`import torch
import torch.nn as nn

class GatedCrossAttention(nn.Module):
    """Cross-attention layer inserted into frozen LM (Flamingo)."""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(1))  # tanh gate, init at 0
        self.norm = nn.LayerNorm(dim)

    def forward(self, text_hidden, visual_tokens):
        # Cross-attend from text to visual tokens
        residual = text_hidden
        x = self.norm(text_hidden)
        attn_out, _ = self.cross_attn(query=x, key=visual_tokens, value=visual_tokens)
        # Gated residual — gate starts at 0 so LM behavior is preserved initially
        return residual + torch.tanh(self.gate) * attn_out

layer = GatedCrossAttention()
text_h = torch.randn(2, 128, 768)    # [batch, seq_len, dim]
vis_tok = torch.randn(2, 64, 768)    # [batch, num_latents, dim]
out = layer(text_h, vis_tok)
print(f"Output shape: {out.shape}")   # [2, 128, 768]
print(f"Initial gate value: {torch.tanh(layer.gate).item():.4f}")`}
      />

      <NoteBlock type="note" title="Interleaved Image-Text Input">
        <p>
          Flamingo can process sequences with multiple images interleaved with text, such as
          "Image1: [img] This is a cat. Image2: [img] This is a ___". This interleaving
          enables in-context learning for multimodal tasks, similar to how GPT-3 does few-shot
          learning with text-only examples.
        </p>
      </NoteBlock>
    </div>
  )
}
