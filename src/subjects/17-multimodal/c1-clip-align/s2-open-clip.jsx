import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function LossComparison() {
  const [score, setScore] = useState(0.5)
  const softmaxLoss = -Math.log(Math.exp(score / 0.07) / (Math.exp(score / 0.07) + Math.exp(0.2 / 0.07)))
  const sigmoidLoss = -Math.log(1 / (1 + Math.exp(-score / 0.07))) - Math.log(1 - 1 / (1 + Math.exp(0.2 / 0.07)))

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Softmax vs Sigmoid Loss Comparison</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Positive pair similarity: {score.toFixed(2)}
        <input type="range" min={-1} max={1} step={0.01} value={score} onChange={e => setScore(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
          <p className="font-medium text-violet-700 dark:text-violet-300">Softmax (CLIP)</p>
          <p className="text-gray-600 dark:text-gray-400">Loss: {softmaxLoss.toFixed(4)}</p>
          <p className="text-xs mt-1 text-gray-500">Requires all-to-all comparison within batch</p>
        </div>
        <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20">
          <p className="font-medium text-violet-700 dark:text-violet-300">Sigmoid (SigLIP)</p>
          <p className="text-gray-600 dark:text-gray-400">Loss: {sigmoidLoss.toFixed(4)}</p>
          <p className="text-xs mt-1 text-gray-500">Pairwise — no global normalization needed</p>
        </div>
      </div>
    </div>
  )
}

export default function OpenCLIPSigLIP() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        OpenCLIP is an open-source reproduction of CLIP trained on public datasets like LAION-2B.
        SigLIP replaces the softmax-based contrastive loss with a simpler sigmoid loss,
        removing the need for global batch normalization and enabling better scaling.
      </p>

      <DefinitionBlock title="SigLIP Sigmoid Loss">
        <p>Instead of softmax over the full batch, SigLIP applies a binary sigmoid loss to each pair independently:</p>
        <BlockMath math="\mathcal{L} = -\frac{1}{N^2}\sum_{i,j}\left[y_{ij}\log\sigma(s_{ij}/\tau) + (1-y_{ij})\log(1-\sigma(s_{ij}/\tau))\right]" />
        <p className="mt-2">where <InlineMath math="y_{ij} = \mathbb{1}[i=j]" /> and <InlineMath math="s_{ij} = \text{sim}(I_i, T_j)" />. This eliminates the softmax denominator that requires all-gather across devices.</p>
      </DefinitionBlock>

      <LossComparison />

      <ExampleBlock title="OpenCLIP Scaling Results">
        <p>OpenCLIP trained on LAION-2B with ViT-G/14 achieves:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>ImageNet zero-shot top-1: <strong>80.1%</strong> (vs. CLIP's 76.2%)</li>
          <li>Training: 34B samples seen, 1024 GPUs</li>
          <li>Key insight: open data + longer training can match proprietary data</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="SigLIP-Style Sigmoid Contrastive Loss"
        code={`import torch
import torch.nn.functional as F

def siglip_loss(image_embeds, text_embeds, temperature=0.1, bias=0.0):
    """Sigmoid contrastive loss (SigLIP).
    No softmax normalization — each pair is independent."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    # Pairwise similarities [N, N]
    logits = image_embeds @ text_embeds.T / temperature + bias

    # Binary labels: 1 on diagonal, 0 elsewhere
    N = logits.shape[0]
    labels = 2 * torch.eye(N, device=logits.device) - 1  # +1 or -1

    # Sigmoid binary cross-entropy
    loss = -F.logsigmoid(labels * logits).mean()
    return loss

# Compare losses
img = torch.randn(16, 512)
txt = torch.randn(16, 512)
print(f"SigLIP loss: {siglip_loss(img, txt).item():.4f}")`}
      />

      <NoteBlock type="note" title="Advantages of Sigmoid Loss">
        <p>
          The sigmoid formulation has two practical benefits: (1) it avoids the all-gather communication
          needed for softmax normalization in distributed training, enabling larger effective batch sizes,
          and (2) it provides a per-pair learning signal rather than competing within a batch, which
          empirically improves performance on smaller batches.
        </p>
      </NoteBlock>

      <NoteBlock type="note" title="LAION Datasets">
        <p>
          OpenCLIP was trained on LAION-400M and LAION-2B, large-scale image-text datasets
          collected from Common Crawl. These datasets enabled the research community to
          reproduce and extend CLIP-style training without proprietary data. Key processing
          steps include NSFW filtering, deduplication, and CLIP-based quality scoring to
          remove low-quality pairs.
        </p>
      </NoteBlock>

      <ExampleBlock title="SigLIP vs CLIP Performance Comparison">
        <p>SigLIP with ViT-B/16 on ImageNet zero-shot classification:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>SigLIP: 73.2% top-1 accuracy (sigmoid loss)</li>
          <li>CLIP: 71.1% top-1 accuracy (softmax loss)</li>
          <li>Improvement comes from better gradient signal per pair</li>
          <li>Advantage grows at smaller batch sizes (128-4096)</li>
        </ul>
        <p className="mt-2">The sigmoid loss also enables batch sizes up to 1M without communication overhead.</p>
      </ExampleBlock>
    </div>
  )
}
