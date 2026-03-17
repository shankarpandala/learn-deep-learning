import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TokenizationViz() {
  const [vocabSize, setVocabSize] = useState(8192)
  const gridSize = 8
  const totalPatches = gridSize * gridSize

  const tokenIds = Array.from({ length: totalPatches }, (_, i) =>
    Math.floor(Math.sin(i * 7.3 + 2.1) * vocabSize / 2 + vocabSize / 2) % vocabSize
  )

  const maskedIndices = new Set([5, 12, 18, 23, 30, 37, 42, 48, 51, 55, 60])

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Visual Tokenization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Codebook size: {vocabSize.toLocaleString()}
        <input type="range" min={512} max={16384} step={512} value={vocabSize} onChange={e => setVocabSize(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex justify-center">
        <div className="grid gap-0.5" style={{ gridTemplateColumns: `repeat(${gridSize}, 1fr)` }}>
          {Array.from({ length: totalPatches }, (_, i) => {
            const isMasked = maskedIndices.has(i)
            const hue = (tokenIds[i] / vocabSize) * 270
            return (
              <div key={i} className="w-7 h-7 rounded-sm flex items-center justify-center text-[7px] text-white font-mono"
                style={{ backgroundColor: isMasked ? '#9ca3af' : `hsl(${hue}, 60%, 55%)` }}>
                {isMasked ? '?' : tokenIds[i]}
              </div>
            )
          })}
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center mt-2">
        Each patch is mapped to a discrete token ID. Masked patches (gray) must be predicted.
      </p>
    </div>
  )
}

export default function BEiT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        BEiT (Bidirectional Encoder representation from Image Transformers) adapts BERT-style
        pre-training to vision by predicting discrete visual tokens for masked patches, rather
        than raw pixels. This provides a higher-level reconstruction target.
      </p>

      <DefinitionBlock title="BEiT Pre-training">
        <p>BEiT uses a two-stage approach:</p>
        <ol className="list-decimal ml-5 mt-2 space-y-1">
          <li><strong>Stage 1</strong>: Train a discrete VAE (dVAE) tokenizer to map image patches to visual tokens <InlineMath math="v_i \in \{1, \ldots, V\}" /></li>
          <li><strong>Stage 2</strong>: Mask patches and predict their token IDs via cross-entropy</li>
        </ol>
        <BlockMath math="\mathcal{L}_{\text{BEiT}} = -\sum_{i \in \mathcal{M}} \log p_\theta(v_i | \mathbf{x}_{\setminus \mathcal{M}})" />
        <p className="mt-1">
          where <InlineMath math="v_i" /> is the visual token for patch <InlineMath math="i" /> and
          <InlineMath math="\mathcal{M}" /> is the set of masked positions.
        </p>
      </DefinitionBlock>

      <TokenizationViz />

      <ExampleBlock title="BEiT v2: Semantic Visual Tokens">
        <p>
          BEiT v2 replaces the dVAE tokenizer with a vector-quantized knowledge distillation (VQ-KD)
          tokenizer trained using a CLIP teacher. This produces semantically meaningful tokens where
          similar visual concepts share the same token ID, significantly improving the pre-training signal.
        </p>
        <BlockMath math="\text{BEiT v2 tokenizer: } v_i = \text{VQ}(\text{CLIP}_{\text{visual}}(\text{patch}_i))" />
      </ExampleBlock>

      <PythonCode
        title="BEiT-style Masked Image Modeling"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class BEiTPretraining(nn.Module):
    def __init__(self, num_patches=196, embed_dim=768, vocab_size=8192):
        super().__init__()
        self.vocab_size = vocab_size
        self.patch_embed = nn.Linear(768, embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        # Transformer encoder (simplified)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=12, dim_feedforward=3072, batch_first=True),
            num_layers=2,  # use 12+ in practice
        )
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, patches, mask, target_tokens):
        # patches: (B, N, D), mask: (B, N) bool, target_tokens: (B, N) long
        x = self.patch_embed(patches) + self.pos_embed

        # Replace masked patches with learnable mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        x = torch.where(mask_expanded, self.mask_token.expand_as(x), x)

        x = self.transformer(x)

        # Predict tokens only for masked positions
        logits = self.head(x)  # (B, N, vocab_size)
        masked_logits = logits[mask]  # (num_masked, vocab_size)
        masked_targets = target_tokens[mask]  # (num_masked,)

        loss = F.cross_entropy(masked_logits, masked_targets)
        return loss

model = BEiTPretraining(num_patches=196, vocab_size=8192)
patches = torch.randn(4, 196, 768)
mask = torch.rand(4, 196) > 0.6  # ~40% masking
tokens = torch.randint(0, 8192, (4, 196))
loss = model(patches, mask, tokens)
print(f"BEiT loss: {loss.item():.3f}")
print(f"Random baseline: {torch.log(torch.tensor(8192.0)):.3f}")`}
      />

      <WarningBlock title="Tokenizer Quality Matters">
        <p>
          BEiT's performance depends heavily on the quality of the visual tokenizer. A poor tokenizer
          produces noisy tokens that make the prediction task ill-defined. BEiT v2's CLIP-based
          tokenizer outperforms BEiT v1's dVAE by providing semantically richer targets.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="BEiT vs MAE">
        <p>
          BEiT predicts discrete tokens (classification); MAE predicts pixels (regression).
          BEiT typically shows stronger linear probe performance (the representations are more
          semantic), while MAE excels at fine-tuning. BEiT requires pre-training a tokenizer;
          MAE needs no extra components. Both achieve similar final fine-tuning accuracy.
        </p>
      </NoteBlock>
    </div>
  )
}
