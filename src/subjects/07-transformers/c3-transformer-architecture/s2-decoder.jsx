import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function CausalMaskDemo() {
  const [seqLen, setSeqLen] = useState(5)
  const tokens = ['<s>', 'The', 'cat', 'sat', 'down', 'on', 'the'][0]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Causal (Autoregressive) Mask</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Sequence length: {seqLen}
        <input type="range" min={3} max={7} step={1} value={seqLen} onChange={e => setSeqLen(parseInt(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <div className="flex justify-center">
        <table className="border-collapse">
          <tbody>
            {Array.from({ length: seqLen }, (_, i) => (
              <tr key={i}>
                {Array.from({ length: seqLen }, (_, j) => (
                  <td key={j} className="w-10 h-10 text-center text-xs font-mono border border-gray-200 dark:border-gray-700" style={{ backgroundColor: j <= i ? 'rgba(139, 92, 246, 0.5)' : 'rgba(220, 38, 38, 0.15)' }}>
                    {j <= i ? '1' : '0'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-center mt-2 text-gray-500 dark:text-gray-400">
        <span className="text-violet-600">Violet = visible</span>, <span className="text-red-400">Red = masked (future tokens)</span>
      </p>
    </div>
  )
}

export default function TransformerDecoder() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The transformer decoder generates output tokens autoregressively — one at a time, left to right.
        It contains masked self-attention (preventing future token visibility), cross-attention to the
        encoder, and a feed-forward network, all with residual connections.
      </p>

      <DefinitionBlock title="Decoder Block">
        <p>Each decoder block has three sub-layers:</p>
        <BlockMath math="h_1 = \text{LN}(x + \text{MaskedSelfAttn}(x))" />
        <BlockMath math="h_2 = \text{LN}(h_1 + \text{CrossAttn}(h_1, H^{enc}))" />
        <BlockMath math="\text{out} = \text{LN}(h_2 + \text{FFN}(h_2))" />
      </DefinitionBlock>

      <DefinitionBlock title="Masked Self-Attention">
        <p>The causal mask sets future positions to <InlineMath math="-\infty" /> before softmax:</p>
        <BlockMath math="\text{mask}_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}" />
        <p className="mt-2">
          This ensures token <InlineMath math="i" /> can only attend to positions <InlineMath math="\leq i" />,
          preserving the autoregressive property.
        </p>
      </DefinitionBlock>

      <CausalMaskDemo />

      <ExampleBlock title="Autoregressive Generation">
        <p>
          At inference, the decoder generates one token at a time. To produce token <InlineMath math="t" />,
          it conditions on all previous tokens <InlineMath math="y_{<t}" /> and the encoder output.
          The next-token probability is:
        </p>
        <BlockMath math="P(y_t \mid y_{<t}, X) = \text{softmax}(h_t W_{\text{vocab}})" />
      </ExampleBlock>

      <PythonCode
        title="Transformer Decoder Block"
        code={`import torch
import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.masked_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def causal_mask(seq_len, device):
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device) * float('-inf'), diagonal=1
        )

    def forward(self, x, enc_output):
        mask = self.causal_mask(x.size(1), x.device)
        h, _ = self.masked_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + h)
        h, _ = self.cross_attn(x, enc_output, enc_output)
        x = self.norm2(x + h)
        x = self.norm3(x + self.ffn(x))
        return x

dec = TransformerDecoderBlock(d_model=512, num_heads=8, d_ff=2048)
tgt = torch.randn(2, 15, 512)
memory = torch.randn(2, 20, 512)
out = dec(tgt, memory)
print(f"Decoder output: {out.shape}")  # (2, 15, 512)`}
      />

      <WarningBlock title="Training vs Inference Mismatch">
        <p>
          During training, all target positions are processed in parallel using teacher forcing
          (feeding ground truth tokens). At inference, tokens are generated one by one. This
          discrepancy can cause <em>exposure bias</em> — the model never sees its own mistakes during
          training.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Decoder-Only Models">
        <p>
          Models like GPT use only the decoder stack (no cross-attention or encoder). The entire
          input and output are treated as a single sequence with causal masking. This simplification
          has proven remarkably effective for language modeling and generation tasks.
        </p>
      </NoteBlock>
    </div>
  )
}
