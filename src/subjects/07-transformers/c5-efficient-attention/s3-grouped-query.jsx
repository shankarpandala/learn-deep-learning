import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function GQADiagram() {
  const [mode, setMode] = useState('gqa')
  const numQueryHeads = 8

  const config = {
    mha: { kvHeads: 8, label: 'Multi-Head (MHA)', ratio: '1:1' },
    gqa: { kvHeads: 2, label: 'Grouped-Query (GQA)', ratio: '4:1' },
    mqa: { kvHeads: 1, label: 'Multi-Query (MQA)', ratio: '8:1' },
  }

  const cur = config[mode]
  const groupSize = numQueryHeads / cur.kvHeads

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">KV Head Sharing</h3>
      <div className="flex gap-2 mb-4 mt-2">
        {Object.keys(config).map(k => (
          <button key={k} onClick={() => setMode(k)} className={`px-3 py-1 rounded-lg text-sm font-medium transition ${mode === k ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {config[k].label}
          </button>
        ))}
      </div>
      <div className="space-y-3">
        <div>
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">Query Heads ({numQueryHeads})</p>
          <div className="flex gap-1">
            {Array.from({ length: numQueryHeads }, (_, i) => (
              <div key={i} className="w-9 h-8 rounded text-xs flex items-center justify-center font-mono" style={{ backgroundColor: `rgba(139, 92, 246, ${0.3 + (i % groupSize) * 0.15})`, border: '1px solid #8b5cf6' }}>
                Q{i}
              </div>
            ))}
          </div>
        </div>
        <div className="text-center text-gray-400 text-xs">shares KV with ratio {cur.ratio}</div>
        <div>
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 mb-1">KV Heads ({cur.kvHeads})</p>
          <div className="flex gap-1">
            {Array.from({ length: cur.kvHeads }, (_, i) => (
              <div key={i} className="h-8 rounded text-xs flex items-center justify-center font-mono bg-violet-200 dark:bg-violet-800 text-violet-800 dark:text-violet-200 border border-violet-400" style={{ width: `${(numQueryHeads / cur.kvHeads) * 40 - 4}px` }}>
                KV{i}
              </div>
            ))}
          </div>
        </div>
      </div>
      <p className="text-sm mt-3 text-gray-600 dark:text-gray-400">
        KV cache memory: <strong className="text-violet-600 dark:text-violet-400">{((cur.kvHeads / numQueryHeads) * 100).toFixed(0)}%</strong> of full MHA
      </p>
    </div>
  )
}

export default function GroupedQueryAttention() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) reduce the KV cache
        memory footprint by sharing key-value heads across multiple query heads. This is critical
        for serving large language models efficiently during inference.
      </p>

      <DefinitionBlock title="Multi-Query Attention (MQA)">
        <p>All query heads share a single set of keys and values:</p>
        <BlockMath math="Q_i = XW_i^Q \quad (h \text{ different}), \quad K = XW^K, \quad V = XW^V \quad (\text{shared})" />
        <p className="mt-2">
          KV cache is reduced by factor <InlineMath math="h" /> (number of heads), but quality
          can degrade due to reduced capacity.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Grouped-Query Attention (GQA)">
        <p>Query heads are divided into <InlineMath math="g" /> groups, each sharing one KV head:</p>
        <BlockMath math="\text{head}_i = \text{Attn}(Q_i, K_{\lfloor i/G \rfloor}, V_{\lfloor i/G \rfloor})" />
        <p className="mt-2">
          where <InlineMath math="G = h / g" /> is the group size. GQA interpolates between MHA (<InlineMath math="g = h" />)
          and MQA (<InlineMath math="g = 1" />).
        </p>
      </DefinitionBlock>

      <GQADiagram />

      <ExampleBlock title="KV Cache Memory Savings">
        <p>For LLaMA-2 70B with 64 query heads, 8 KV heads, 128 dim/head, 80 layers:</p>
        <BlockMath math="\text{KV cache per token} = 2 \times 80 \times 8 \times 128 \times 2 \text{ bytes} = 327 \text{ KB}" />
        <p>
          Compared to full MHA: <InlineMath math="2 \times 80 \times 64 \times 128 \times 2 = 2.6" /> MB per token.
          That is an <strong>8x reduction</strong>, enabling much larger batch sizes during serving.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Grouped-Query Attention Implementation"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_q_heads, num_kv_heads):
        super().__init__()
        assert num_q_heads % num_kv_heads == 0
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_q_heads // num_kv_heads
        self.d_k = d_model // num_q_heads

        self.W_q = nn.Linear(d_model, num_q_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, _ = x.shape
        Q = self.W_q(x).view(B, N, self.num_q_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, N, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, N, self.num_kv_heads, self.d_k).transpose(1, 2)

        # Repeat KV heads to match query heads
        K = K.repeat_interleave(self.group_size, dim=1)
        V = V.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V).transpose(1, 2).reshape(B, N, -1)
        return self.W_o(out)

# LLaMA-2 style: 32 query heads, 8 KV heads
gqa = GroupedQueryAttention(d_model=4096, num_q_heads=32, num_kv_heads=8)
x = torch.randn(1, 128, 4096)
print(f"Output: {gqa(x).shape}")  # (1, 128, 4096)
print(f"KV params: {sum(p.numel() for p in [gqa.W_k, gqa.W_v]):,}")
print(f"Q params:  {gqa.W_q.weight.numel():,}")`}
      />

      <WarningBlock title="Converting MHA to GQA">
        <p>
          Existing MHA models can be converted to GQA by mean-pooling the KV heads within each
          group, then fine-tuning briefly. This &quot;uptrained&quot; GQA model recovers most of the
          original quality while getting the inference memory benefits. Simply dropping heads
          without fine-tuning degrades performance significantly.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Industry Adoption">
        <p>
          GQA is now standard in production LLMs: LLaMA-2/3 (8 KV heads), Mistral (8 KV heads),
          and Gemma all use it. The quality-efficiency trade-off has proven favorable at scale,
          with GQA matching MHA quality while enabling 4-8x larger batch sizes during inference.
        </p>
      </NoteBlock>
    </div>
  )
}
