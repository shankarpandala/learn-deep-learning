import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function ComparisonTable() {
  const [highlight, setHighlight] = useState(null)
  const rows = [
    { property: 'Parameters', sinusoidal: 'None (fixed)', learned: 'max_len x d_model' },
    { property: 'Length generalization', sinusoidal: 'Theoretically yes', learned: 'No — limited to training length' },
    { property: 'Expressiveness', sinusoidal: 'Fixed frequency patterns', learned: 'Data-adaptive patterns' },
    { property: 'Used in', sinusoidal: 'Original Transformer', learned: 'BERT, GPT-2, ViT' },
    { property: 'Training cost', sinusoidal: 'Zero', learned: 'Marginal (small table)' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Sinusoidal vs Learned Positional Embeddings</h3>
      <table className="w-full text-sm">
        <thead>
          <tr>
            <th className="text-left py-1 px-2 text-gray-500 dark:text-gray-400 font-medium">Property</th>
            <th className="text-left py-1 px-2 text-violet-600 dark:text-violet-400 font-medium cursor-pointer" onClick={() => setHighlight(h => h === 'sin' ? null : 'sin')}>Sinusoidal</th>
            <th className="text-left py-1 px-2 text-violet-600 dark:text-violet-400 font-medium cursor-pointer" onClick={() => setHighlight(h => h === 'learn' ? null : 'learn')}>Learned</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-t border-gray-100 dark:border-gray-800">
              <td className="py-1.5 px-2 text-gray-700 dark:text-gray-300 font-medium">{r.property}</td>
              <td className={`py-1.5 px-2 ${highlight === 'sin' ? 'bg-violet-50 dark:bg-violet-900/20' : ''} text-gray-600 dark:text-gray-400`}>{r.sinusoidal}</td>
              <td className={`py-1.5 px-2 ${highlight === 'learn' ? 'bg-violet-50 dark:bg-violet-900/20' : ''} text-gray-600 dark:text-gray-400`}>{r.learned}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs mt-2 text-gray-400">Click column headers to highlight.</p>
    </div>
  )
}

export default function LearnedPositional() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Learned positional embeddings replace the fixed sinusoidal functions with a trainable
        embedding table. Models like BERT, GPT-2, and ViT use this approach, allowing the model
        to discover optimal positional representations from data.
      </p>

      <DefinitionBlock title="Learned Positional Embeddings">
        <p>
          A learnable embedding table <InlineMath math="E_{pos} \in \mathbb{R}^{L_{\max} \times d}" /> is
          added to the token embeddings:
        </p>
        <BlockMath math="h_i^{(0)} = E_{\text{token}}(x_i) + E_{\text{pos}}(i)" />
        <p className="mt-2">
          where <InlineMath math="L_{\max}" /> is the maximum sequence length (e.g., 512 for BERT,
          1024 for GPT-2). Both embedding tables are learned end-to-end via backpropagation.
        </p>
      </DefinitionBlock>

      <ComparisonTable />

      <ExampleBlock title="BERT Positional Embeddings">
        <p>
          BERT supports sequences up to 512 tokens and learns a <InlineMath math="512 \times 768" /> position
          embedding table — only 393K parameters out of 110M total (0.36%). BERT also
          adds <strong>segment embeddings</strong> to distinguish sentence A from sentence B:
        </p>
        <BlockMath math="h_i = E_{\text{token}}(x_i) + E_{\text{pos}}(i) + E_{\text{seg}}(s_i)" />
      </ExampleBlock>

      <PythonCode
        title="Learned Positional Embedding in PyTorch"
        code={`import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.d_model = d_model

    def forward(self, x):
        B, N = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0)
        tok = self.token_emb(x) * (self.d_model ** 0.5)  # scale
        pos = self.pos_emb(positions)
        return tok + pos

emb = TokenEmbedding(vocab_size=30000, d_model=768, max_len=512)
tokens = torch.randint(0, 30000, (2, 128))
output = emb(tokens)
print(f"Embedding output: {output.shape}")  # (2, 128, 768)

# Visualize learned position similarity
with torch.no_grad():
    pe = emb.pos_emb.weight  # (512, 768)
    sim = torch.cosine_similarity(pe[0:1], pe, dim=-1)
    print(f"Position 0 vs 1: {sim[1]:.4f}")
    print(f"Position 0 vs 100: {sim[100]:.4f}")`}
      />

      <WarningBlock title="Length Extrapolation Failure">
        <p>
          Learned positional embeddings cannot extrapolate beyond the maximum training length.
          If a model trained with <InlineMath math="L_{\max} = 512" /> receives 600 tokens, positions
          513-600 have no valid embedding. This limitation motivated the development of relative
          positional encoding methods like RoPE and ALiBi.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Vision Transformer (ViT) Positions">
        <p>
          ViT treats image patches as tokens and uses learned 2D positional embeddings. Interestingly,
          the learned embeddings recover a grid structure resembling the spatial layout of patches,
          demonstrating that the model naturally discovers meaningful positional information.
        </p>
      </NoteBlock>
    </div>
  )
}
