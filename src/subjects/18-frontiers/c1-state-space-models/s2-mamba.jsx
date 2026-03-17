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
        title="Mamba Selective Scan (Simplified)"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMambaBlock(nn.Module):
    """Simplified Mamba block with selective state space."""
    def __init__(self, d_model=256, d_state=16, d_conv=4):
        super().__init__()
        self.d_state = d_state
        # Input projections
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(d_model, d_model, d_conv, padding=d_conv-1, groups=d_model)
        # Selection parameters (input-dependent)
        self.x_proj = nn.Linear(d_model, d_state * 2 + 1, bias=False)  # B, C, dt
        self.dt_proj = nn.Linear(1, d_model, bias=True)
        # Fixed A (diagonal, log-parameterized)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))

    def forward(self, x):
        B, L, D = x.shape
        xz = self.in_proj(x)             # [B, L, 2D]
        x_main, z = xz.chunk(2, dim=-1)  # each [B, L, D]

        # Causal conv1d
        x_main = self.conv1d(x_main.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_main = F.silu(x_main)

        # Input-dependent parameters
        x_proj = self.x_proj(x_main)
        B_t = x_proj[:, :, :self.d_state]         # [B, L, N]
        C_t = x_proj[:, :, self.d_state:2*self.d_state]  # [B, L, N]
        dt = F.softplus(x_proj[:, :, -1:])         # [B, L, 1]

        # Selective scan (sequential for clarity)
        A = -torch.exp(self.A_log)                  # [N]
        h = torch.zeros(B, D, self.d_state, device=x.device)
        ys = []
        for t in range(L):
            A_bar = torch.exp(dt[:, t] * A)         # [B, 1] * [N] -> [B, N]
            h = h * A_bar.unsqueeze(1) + x_main[:, t, :, None] * B_t[:, t, None, :]
            y_t = (h * C_t[:, t, None, :]).sum(-1)  # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                  # [B, L, D]
        return y * F.silu(z)                        # Gated output

mamba = SimpleMambaBlock(d_model=256, d_state=16)
x = torch.randn(2, 128, 256)
out = mamba(x)
print(f"Input: {x.shape} -> Output: {out.shape}")`}
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
