import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SelectionMechanismViz() {
  const [inputType, setInputType] = useState('relevant')
  const inputs = {
    relevant: { label: 'Relevant Token', delta: 0.8, bScale: 1.0, desc: 'Large delta -> retain in state; large B -> strong input projection' },
    irrelevant: { label: 'Irrelevant Token', delta: 0.05, bScale: 0.1, desc: 'Small delta -> skip/forget; small B -> weak input projection' },
    reset: { label: 'Reset Token', delta: 2.0, bScale: 0.0, desc: 'Very large delta -> clear history; zero B -> no new information stored' },
  }
  const inp = inputs[inputType]
  const decay = Math.exp(-inp.delta)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Mamba Selective State Space</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(inputs).map(([key, val]) => (
          <button key={key} onClick={() => setInputType(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${inputType === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm text-center mb-2">
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Delta</p>
          <p className="font-bold">{inp.delta.toFixed(2)}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">State Decay</p>
          <p className="font-bold">{decay.toFixed(3)}</p>
        </div>
        <div className="p-2 rounded bg-violet-50 dark:bg-violet-900/20">
          <p className="text-violet-700 dark:text-violet-300 font-medium">Input Scale</p>
          <p className="font-bold">{inp.bScale.toFixed(2)}</p>
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center">{inp.desc}</p>
    </div>
  )
}

export default function MambaSelectiveSSM() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Mamba introduces input-dependent selection into SSMs, making the state transition parameters
        vary with the input. This gives SSMs the ability to selectively remember or forget
        information — a capability previously unique to attention mechanisms.
      </p>

      <DefinitionBlock title="Selective State Space (Mamba)">
        <p>Unlike S4 where <InlineMath math="A, B, C, \Delta" /> are fixed, Mamba makes <InlineMath math="B, C, \Delta" /> functions of the input:</p>
        <BlockMath math="B_t = \text{Linear}(x_t), \quad C_t = \text{Linear}(x_t), \quad \Delta_t = \text{softplus}(\text{Linear}(x_t))" />
        <p className="mt-2">The discretized recurrence becomes input-dependent:</p>
        <BlockMath math="h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t, \quad y_t = C_t h_t" />
        <p className="mt-1">where <InlineMath math="\bar{A}_t = \exp(\Delta_t A)" />. This <strong>breaks the convolution</strong> structure but enables content-based reasoning.</p>
      </DefinitionBlock>

      <SelectionMechanismViz />

      <ExampleBlock title="Mamba vs Transformer Efficiency">
        <p>For sequence length <InlineMath math="L" /> with model dimension <InlineMath math="D" /> and state dimension <InlineMath math="N" />:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Transformer attention: <InlineMath math="O(L^2 D)" /> FLOPs, <InlineMath math="O(L^2)" /> memory</li>
          <li>Mamba (parallel scan): <InlineMath math="O(L D N)" /> FLOPs, <InlineMath math="O(L D N)" /> memory</li>
          <li>Mamba generation: <InlineMath math="O(DN)" /> per token (constant, no KV-cache growth)</li>
          <li>At L=8192: Mamba is 5x faster inference, 3x faster training than Transformer++</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Mamba with mamba-ssm Library"
        code={`from mamba_ssm import Mamba, Mamba2
import torch

# Mamba: selective state space model with hardware-aware scan
mamba_layer = Mamba(
    d_model=256,    # model dimension
    d_state=16,     # SSM state expansion factor (N in paper)
    d_conv=4,       # local convolution width
    expand=2,       # block expansion factor (E in paper)
).to("cuda")

x = torch.randn(2, 128, 256).to("cuda")  # (batch, seq_len, d_model)
y = mamba_layer(x)
print(f"Mamba: {x.shape} -> {y.shape}")  # same shape

# Mamba-2: improved with structured state space duality (SSD)
mamba2_layer = Mamba2(
    d_model=256,
    d_state=64,     # larger state in Mamba-2
    d_conv=4,
    expand=2,
    headdim=64,     # SSD head dimension
).to("cuda")
y2 = mamba2_layer(x)
print(f"Mamba-2: {x.shape} -> {y2.shape}")

# Full model: stack Mamba blocks like Transformer layers
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
model = MambaLMHeadModel(
    d_model=768,
    n_layer=24,
    vocab_size=50277,
    ssm_cfg={"d_state": 16, "d_conv": 4, "expand": 2},
).to("cuda")

# Efficient autoregressive generation (constant memory per step)
input_ids = torch.randint(0, 50277, (1, 64)).to("cuda")
out = model(input_ids)
print(f"LM logits: {out.logits.shape}")  # [1, 64, 50277]
# Generation: O(DN) per token — no KV cache growth!
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model size: {params:.0f}M parameters")`}
      />

      <NoteBlock type="note" title="Hardware-Aware Algorithm">
        <p>
          Mamba uses a hardware-aware parallel scan algorithm that avoids materializing the full
          <InlineMath math="(B, L, D, N)" /> state tensor in GPU HBM. Instead, it fuses the discretization,
          scan, and output computation in a single kernel, achieving near-optimal memory bandwidth
          utilization. This engineering is as important as the algorithmic innovation.
        </p>
      </NoteBlock>
    </div>
  )
}
