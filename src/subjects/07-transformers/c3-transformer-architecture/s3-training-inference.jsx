import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function KVCacheDemo() {
  const [step, setStep] = useState(1)
  const maxSteps = 6
  const tokens = ['<s>', 'The', 'cat', 'sat', 'on', 'the']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">KV Cache During Autoregressive Generation</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Decoding step: {step}
        <input type="range" min={1} max={maxSteps} step={1} value={step} onChange={e => setStep(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <div className="flex gap-2 mb-2">
        {tokens.slice(0, step).map((t, i) => (
          <div key={i} className={`px-2 py-1 rounded text-sm font-mono ${i < step - 1 ? 'bg-violet-100 text-violet-700 dark:bg-violet-900/50 dark:text-violet-300' : 'bg-violet-500 text-white font-bold'}`}>
            {t}
          </div>
        ))}
      </div>
      <div className="text-sm text-gray-600 dark:text-gray-400">
        <p>Cached K/V: <strong className="text-violet-600 dark:text-violet-400">{step - 1}</strong> positions</p>
        <p>New computation: only for token <strong className="text-violet-600 dark:text-violet-400">&quot;{tokens[step - 1]}&quot;</strong></p>
        <p className="mt-1">Without cache: {step} Q/K/V computations. With cache: <strong>1</strong> new + {step - 1} cached.</p>
      </div>
    </div>
  )
}

export default function TrainingInference() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Training and inference in transformers involve fundamentally different computational patterns.
        Training leverages parallelism with teacher forcing, while inference requires sequential
        generation with optimization techniques like KV caching and beam search.
      </p>

      <DefinitionBlock title="Teacher Forcing">
        <p>
          During training, the decoder receives the ground-truth target tokens as input rather than
          its own predictions. For a target sequence <InlineMath math="y = (y_1, \ldots, y_T)" />:
        </p>
        <BlockMath math="\mathcal{L} = -\sum_{t=1}^{T} \log P(y_t \mid y_1, \ldots, y_{t-1}, X)" />
        <p className="mt-2">
          All positions are computed in parallel since we have the full target during training.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="KV Caching">
        <p>
          During autoregressive inference, previously computed key and value vectors are cached
          and reused. At step <InlineMath math="t" />, only the new token's Q, K, V are computed:
        </p>
        <BlockMath math="K_t = [K_{\text{cache}}; k_t], \quad V_t = [V_{\text{cache}}; v_t]" />
        <p className="mt-2">
          This reduces per-step complexity from <InlineMath math="O(t \cdot d)" /> to <InlineMath math="O(d)" />
          for the projection, though the attention still requires <InlineMath math="O(t)" />.
        </p>
      </DefinitionBlock>

      <KVCacheDemo />

      <ExampleBlock title="Beam Search">
        <p>
          Beam search maintains <InlineMath math="B" /> candidate sequences at each step, expanding
          each by the top-<InlineMath math="k" /> tokens and keeping the <InlineMath math="B" /> best overall:
        </p>
        <BlockMath math="\text{score}(y) = \frac{1}{|y|^\alpha} \sum_{t=1}^{|y|} \log P(y_t \mid y_{<t})" />
        <p>
          The length penalty <InlineMath math="\alpha" /> (typically 0.6-0.7) prevents the model from
          favoring short sequences.
        </p>
      </ExampleBlock>

      <PythonCode
        title="KV Cache Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class CachedSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, kv_cache=None):
        B, N, D = x.shape
        qkv = self.W_qkv(x).reshape(B, N, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, D)
        return self.W_o(out), (k, v)

# Simulate autoregressive generation with KV cache
attn = CachedSelfAttention(d_model=256, num_heads=8)
cache = None
for step in range(5):
    token = torch.randn(1, 1, 256)  # one new token
    out, cache = attn(token, kv_cache=cache)
    print(f"Step {step}: cache K shape = {cache[0].shape}")`}
      />

      <WarningBlock title="KV Cache Memory">
        <p>
          KV cache grows linearly with sequence length and batch size. For a model with
          <InlineMath math="L" /> layers, <InlineMath math="h" /> heads, and dimension <InlineMath math="d" />:
          memory = <InlineMath math="2 \times L \times n \times d \times \text{bytes}" /> per sequence.
          For long contexts, this can exceed GPU memory.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Speculative Decoding">
        <p>
          Speculative decoding uses a small draft model to propose several tokens, then verifies
          them in parallel with the large model. Accepted tokens skip individual decoding steps,
          providing 2-3x speedup without changing the output distribution.
        </p>
      </NoteBlock>
    </div>
  )
}
