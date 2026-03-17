import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SimilarityExplorer() {
  const [temperature, setTemperature] = useState(0.07)
  const [nPairs, setNPairs] = useState(4)
  const sims = Array.from({ length: nPairs }, (_, i) =>
    Array.from({ length: nPairs }, (_, j) => {
      const raw = i === j ? 0.8 + Math.random() * 0.15 : Math.random() * 0.3
      return Math.exp(raw / temperature)
    })
  )
  const rowSums = sims.map(row => row.reduce((a, b) => a + b, 0))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">CLIP Contrastive Similarity Matrix</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Temperature: {temperature.toFixed(2)}
          <input type="range" min={0.01} max={0.5} step={0.01} value={temperature} onChange={e => setTemperature(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Batch size: {nPairs}
          <input type="range" min={2} max={8} step={1} value={nPairs} onChange={e => setNPairs(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="overflow-x-auto">
        <table className="mx-auto text-xs">
          <thead>
            <tr>
              <th className="p-1 text-gray-500">img\txt</th>
              {Array.from({ length: nPairs }, (_, j) => <th key={j} className="p-1 text-violet-600 dark:text-violet-400">T{j}</th>)}
            </tr>
          </thead>
          <tbody>
            {sims.map((row, i) => (
              <tr key={i}>
                <td className="p-1 font-medium text-violet-600 dark:text-violet-400">I{i}</td>
                {row.map((v, j) => {
                  const prob = v / rowSums[i]
                  const bg = i === j ? `rgba(139, 92, 246, ${Math.min(prob * 1.5, 0.6)})` : `rgba(156, 163, 175, ${prob * 0.5})`
                  return <td key={j} className="p-1 text-center rounded" style={{ backgroundColor: bg }}>{prob.toFixed(2)}</td>
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Diagonal entries are matched image-text pairs (should be high probability)</p>
    </div>
  )
}

export default function CLIPArchitecture() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        CLIP (Contrastive Language-Image Pre-training) by OpenAI learns a shared embedding space
        for images and text using contrastive learning on 400M image-text pairs from the internet.
        It enables powerful zero-shot transfer to downstream vision tasks.
      </p>

      <DefinitionBlock title="CLIP Contrastive Objective">
        <p>Given a batch of <InlineMath math="N" /> image-text pairs, CLIP maximizes the cosine similarity of matching pairs while minimizing it for non-matching pairs:</p>
        <BlockMath math="\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(T_i, I_j)/\tau)}\right]" />
        <p className="mt-2">where <InlineMath math="\tau" /> is a learned temperature parameter and <InlineMath math="\text{sim}(a,b) = \frac{a \cdot b}{\|a\|\|b\|}" />.</p>
      </DefinitionBlock>

      <SimilarityExplorer />

      <ExampleBlock title="Why Contrastive Learning Works at Scale">
        <p>With a batch size of <InlineMath math="N = 32{,}768" />, each image acts as a positive pair with its text and a negative pair with <InlineMath math="32{,}767" /> other texts. This gives:</p>
        <BlockMath math="\text{Negative examples per step} = N^2 - N = 32{,}768^2 - 32{,}768 \approx 10^9" />
        <p>The massive number of negatives drives the model to learn fine-grained distinctions.</p>
      </ExampleBlock>

      <PythonCode
        title="CLIP-Style Contrastive Loss"
        code={`import torch
import torch.nn.functional as F

def clip_loss(image_embeds, text_embeds, temperature=0.07):
    """Symmetric contrastive loss for CLIP."""
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Cosine similarity matrix [N, N]
    logits = image_embeds @ text_embeds.T / temperature

    # Symmetric cross-entropy loss
    N = logits.shape[0]
    labels = torch.arange(N, device=logits.device)
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2

# Example: batch of 8 image-text pairs, 512-dim embeddings
img_emb = torch.randn(8, 512)
txt_emb = torch.randn(8, 512)
loss = clip_loss(img_emb, txt_emb)
print(f"CLIP loss: {loss.item():.4f}")`}
      />

      <NoteBlock type="note" title="Dual Encoder Architecture">
        <p>
          CLIP uses separate encoders for each modality: a Vision Transformer (ViT) or ResNet for images
          and a Transformer for text. Both project to a shared <InlineMath math="d" />-dimensional space (typically 512 or 768).
          This dual-encoder design enables efficient retrieval since embeddings can be precomputed and compared with simple dot products.
        </p>
      </NoteBlock>

      <NoteBlock type="warning" title="Training Scale Matters">
        <p>
          CLIP required 400M image-text pairs and massive compute. Smaller-scale reproductions often
          underperform significantly, highlighting the critical role of data scale in contrastive
          vision-language learning.
        </p>
      </NoteBlock>
    </div>
  )
}
