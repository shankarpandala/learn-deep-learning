import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SAEExplorer() {
  const [dModel, setDModel] = useState(768)
  const [expansionFactor, setExpansionFactor] = useState(32)
  const dSAE = dModel * expansionFactor
  const avgActive = Math.round(dSAE * 0.005)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Sparse Autoencoder Dimensions</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          d_model: {dModel}
          <input type="range" min={256} max={4096} step={256} value={dModel} onChange={e => setDModel(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Expansion: {expansionFactor}x
          <input type="range" min={4} max={128} step={4} value={expansionFactor} onChange={e => setExpansionFactor(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm text-center">
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">SAE Features</p>
          <p className="font-bold">{(dSAE).toLocaleString()}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Active per Input</p>
          <p className="font-bold">~{avgActive} ({(avgActive / dSAE * 100).toFixed(1)}%)</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Sparsity</p>
          <p className="font-bold">{(100 - avgActive / dSAE * 100).toFixed(1)}%</p>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500 text-center">Each input activates only a tiny fraction of the feature dictionary, ensuring monosemantic features.</p>
    </div>
  )
}

export default function SparseAutoencodersInterp() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Sparse autoencoders (SAEs) decompose neural network activations into a large set of
        interpretable, monosemantic features. By learning an overcomplete dictionary with sparsity
        constraints, SAEs untangle the superposition that makes individual neurons hard to interpret.
      </p>

      <DefinitionBlock title="Sparse Autoencoder for Interpretability">
        <p>Given an activation vector <InlineMath math="h \in \mathbb{R}^d" />, the SAE learns an encoder-decoder pair with a sparsity penalty:</p>
        <BlockMath math="f = \text{ReLU}(W_{\text{enc}}(h - b_d) + b_e), \quad \hat{h} = W_{\text{dec}} f + b_d" />
        <BlockMath math="\mathcal{L} = \|h - \hat{h}\|^2 + \lambda \|f\|_1" />
        <p className="mt-2">where <InlineMath math="f \in \mathbb{R}^{D}" /> with <InlineMath math="D \gg d" /> is the sparse feature vector. The L1 penalty on <InlineMath math="f" /> encourages most features to be zero for any given input, yielding monosemantic features.</p>
      </DefinitionBlock>

      <SAEExplorer />

      <ExampleBlock title="Discovered SAE Features (Claude/GPT-2)">
        <p>SAEs trained on language models have discovered interpretable features for:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Specific concepts: "Golden Gate Bridge", "DNA sequences", "Python code"</li>
          <li>Abstract patterns: "start of a list item", "the answer is about to be stated"</li>
          <li>Safety-relevant: "deceptive reasoning", "refusing harmful requests"</li>
          <li>Linguistic: "past tense verbs", "words ending in -tion"</li>
          <li>These features are more interpretable than individual neurons (95%+ vs ~30%)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Training a Sparse Autoencoder"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for mechanistic interpretability."""
    def __init__(self, d_model, d_sae, k_sparse=None):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_sae)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)
        # Normalize decoder columns to unit norm
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)
        self.k_sparse = k_sparse  # If set, use top-k instead of L1

    def forward(self, h):
        # Subtract decoder bias (centering)
        h_centered = h - self.decoder.bias
        # Encode to sparse features
        f = self.encoder(h_centered)
        if self.k_sparse:
            # TopK activation: keep only top-k features
            topk_vals, topk_idx = f.topk(self.k_sparse, dim=-1)
            f = torch.zeros_like(f).scatter(-1, topk_idx, F.relu(topk_vals))
        else:
            f = F.relu(f)
        # Decode
        h_hat = self.decoder(f)
        return h_hat, f

    def loss(self, h, lambda_l1=5e-3):
        h_hat, f = self.forward(h)
        recon_loss = (h - h_hat).pow(2).mean()
        sparsity_loss = f.abs().mean()
        return recon_loss + lambda_l1 * sparsity_loss, recon_loss, sparsity_loss

# Train on random activations (demo)
sae = SparseAutoencoder(d_model=768, d_sae=768*32)
h = torch.randn(128, 768)  # batch of activations
total_loss, recon, sparse = sae.loss(h)
_, features = sae(h)
active = (features > 0).float().mean()
print(f"Reconstruction loss: {recon.item():.4f}")
print(f"Sparsity loss: {sparse.item():.4f}")
print(f"Active features: {active.item()*100:.2f}% ({int(active.item()*768*32)} of {768*32})")`}
      />

      <NoteBlock type="note" title="TopK SAEs and Scaling">
        <p>
          Recent work replaces the L1 penalty with a TopK activation function, directly enforcing
          exactly K active features per input. This avoids the reconstruction-sparsity tradeoff
          of L1 and scales better to very large dictionaries (millions of features). Anthropic's
          work on Claude found that scaling SAE width reveals increasingly fine-grained features
          following a power law.
        </p>
      </NoteBlock>
    </div>
  )
}
