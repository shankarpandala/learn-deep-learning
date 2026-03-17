import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function BetaSlider() {
  const [beta, setBeta] = useState(1.0)
  const reconWeight = 1.0
  const klWeight = beta

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Beta-VAE Trade-off</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        beta = {beta.toFixed(1)}
        <input type="range" min={0.1} max={10} step={0.1} value={beta} onChange={e => setBeta(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-4 items-end h-32">
        <div className="flex flex-col items-center">
          <div className="w-16 bg-violet-400 rounded-t" style={{ height: `${reconWeight / Math.max(reconWeight, klWeight) * 100}px` }} />
          <span className="text-xs text-gray-500 mt-1">Recon ({reconWeight.toFixed(1)})</span>
        </div>
        <div className="flex flex-col items-center">
          <div className="w-16 bg-violet-700 rounded-t" style={{ height: `${klWeight / Math.max(reconWeight, klWeight) * 100}px` }} />
          <span className="text-xs text-gray-500 mt-1">KL ({klWeight.toFixed(1)})</span>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        {beta < 1 ? 'Low beta: better reconstruction, less disentanglement' :
         beta === 1 ? 'Standard VAE (beta=1)' :
         'High beta: more disentanglement, blurrier reconstructions'}
      </p>
    </div>
  )
}

export default function VAEVariants() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Several important VAE variants address limitations of the standard model. Beta-VAE encourages
        disentangled representations, while VQ-VAE replaces continuous latents with discrete codebooks.
      </p>

      <DefinitionBlock title="Beta-VAE">
        <p>Beta-VAE adds a hyperparameter <InlineMath math="\beta" /> to control the KL weight:</p>
        <BlockMath math="\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q}\left[\log p(\mathbf{x}|\mathbf{z})\right] - \beta \cdot D_{\text{KL}}\left(q(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z})\right)" />
        <p className="mt-2">
          When <InlineMath math="\beta > 1" />, the model is pressured to find a more efficient,
          disentangled encoding where each latent dimension captures an independent factor of variation.
        </p>
      </DefinitionBlock>

      <BetaSlider />

      <DefinitionBlock title="VQ-VAE (Vector Quantized VAE)">
        <p>VQ-VAE uses a discrete codebook <InlineMath math="\mathbf{e} \in \mathbb{R}^{K \times D}" />. The encoder output is quantized:</p>
        <BlockMath math="z_q = \mathbf{e}_k, \quad k = \arg\min_j \| f_\theta(\mathbf{x}) - \mathbf{e}_j \|" />
        <p className="mt-2">Training loss combines reconstruction, codebook, and commitment terms:</p>
        <BlockMath math="\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 + \|\text{sg}[z_e] - e\|^2 + \beta\|z_e - \text{sg}[e]\|^2" />
      </DefinitionBlock>

      <PythonCode
        title="VQ-VAE Quantization Layer"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        self.codebook.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        self.commitment_cost = commitment_cost

    def forward(self, z_e):
        # z_e: (B, D) encoder output
        distances = torch.cdist(z_e.unsqueeze(0), self.codebook.weight.unsqueeze(0)).squeeze(0)
        indices = distances.argmin(dim=-1)
        z_q = self.codebook(indices)

        # Straight-through estimator: copy gradients from z_q to z_e
        z_q_st = z_e + (z_q - z_e).detach()

        # Losses
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        return z_q_st, vq_loss, indices

vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
z_e = torch.randn(8, 64)
z_q, loss, idx = vq(z_e)
print(f"Codebook indices: {idx[:4].tolist()}")
print(f"VQ loss: {loss.item():.4f}")`}
      />

      <ExampleBlock title="VQ-VAE-2 for High-Res Images">
        <p>
          VQ-VAE-2 uses a hierarchical codebook with two levels: a top-level captures global structure
          while a bottom-level captures fine details. Combined with a powerful autoregressive prior
          (PixelSNAIL), it generates diverse, high-fidelity images.
        </p>
      </ExampleBlock>

      <WarningBlock title="Codebook Collapse">
        <p>
          A common issue where only a fraction of codebook entries are used. Mitigation strategies
          include exponential moving average updates, codebook reset for dead entries, and
          entropy-based regularization to encourage uniform codebook usage.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Impact on Modern Generative AI">
        <p>
          VQ-VAE forms the backbone of many modern systems: DALL-E uses a VQ-VAE to tokenize images,
          and latent diffusion models (Stable Diffusion) use a VAE encoder to compress images into
          a latent space where diffusion operates more efficiently.
        </p>
      </NoteBlock>
    </div>
  )
}
