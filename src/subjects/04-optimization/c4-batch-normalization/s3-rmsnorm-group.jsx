import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function GroupNormViz() {
  const [numGroups, setNumGroups] = useState(2)
  const C = 8, H = 4
  const cellW = 36, cellH = 28, pad = 2

  const groupColors = ['#8b5cf6', '#f97316', '#10b981', '#ef4444', '#3b82f6', '#f59e0b', '#ec4899', '#06b6d4']

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Group Normalization Groups</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Groups: {numGroups}
        <input type="range" min={1} max={8} step={1} value={numGroups} onChange={e => setNumGroups(parseInt(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <div className="flex justify-center">
        <svg width={C * (cellW + pad) + 80} height={H * (cellH + pad) + 30}>
          <text x={5} y={12} fill="#6b7280" fontSize={9}>Channels →</text>
          <text x={5} y={24 + H * (cellH + pad) / 2} fill="#6b7280" fontSize={9}>Spatial ↓</text>
          {Array.from({ length: H }, (_, h) => (
            Array.from({ length: C }, (_, c) => {
              const g = Math.floor(c / (C / numGroups))
              return (
                <rect key={`${h}-${c}`} x={45 + c * (cellW + pad)} y={20 + h * (cellH + pad)} width={cellW} height={cellH} rx={3} fill={groupColors[g % groupColors.length]} opacity={0.45} stroke={groupColors[g % groupColors.length]} strokeWidth={1.5} />
              )
            })
          ))}
          {Array.from({ length: numGroups }, (_, g) => {
            const groupSize = C / numGroups
            const startC = g * groupSize
            return (
              <text key={`gl-${g}`} x={45 + (startC + groupSize / 2) * (cellW + pad)} y={18} textAnchor="middle" fill={groupColors[g % groupColors.length]} fontSize={9}>
                G{g}
              </text>
            )
          })}
        </svg>
      </div>
      <p className="text-center text-xs text-gray-500 mt-1">
        {numGroups} group{numGroups > 1 ? 's' : ''} of {C / numGroups} channel{C / numGroups > 1 ? 's' : ''} each.
        {numGroups === C ? ' (= Instance Norm)' : numGroups === 1 ? ' (= Layer Norm)' : ''}
      </p>
    </div>
  )
}

export default function RMSNormGroup() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        RMSNorm simplifies LayerNorm by removing the mean centering, while GroupNorm provides
        a flexible middle ground between BatchNorm and InstanceNorm for convolutional networks.
      </p>

      <DefinitionBlock title="RMSNorm">
        <BlockMath math="\text{RMS}(x) = \sqrt{\frac{1}{n}\sum_{i=1}^{n} x_i^2}" />
        <BlockMath math="y_i = \frac{x_i}{\text{RMS}(x) + \epsilon} \cdot \gamma_i" />
        <p className="mt-2">
          No mean subtraction, no learned bias — just scale by the root mean square. This reduces
          computation by ~7-10% compared to LayerNorm with negligible quality loss.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Why Removing the Mean Works" id="rmsnorm-theory">
        <p>
          The re-centering in LayerNorm provides invariance to shifts in activation
          distributions. In practice, the learned parameters <InlineMath math="\gamma" /> and
          the subsequent linear layers can compensate for this. Empirically, the scaling
          (RMS) component does most of the heavy lifting:
        </p>
        <BlockMath math="\text{LayerNorm}(x) \approx \text{RMSNorm}(x) \text{ when } \mu_x \approx 0" />
      </TheoremBlock>

      <DefinitionBlock title="Group Normalization">
        <BlockMath math="\mu_g = \frac{1}{|S_g|}\sum_{i \in S_g} x_i, \quad \sigma_g^2 = \frac{1}{|S_g|}\sum_{i \in S_g}(x_i - \mu_g)^2" />
        <p className="mt-2">
          Channels are divided into <InlineMath math="G" /> groups, each normalized independently.
          When <InlineMath math="G = 1" />, it is LayerNorm; when <InlineMath math="G = C" />, it is InstanceNorm.
        </p>
      </DefinitionBlock>

      <GroupNormViz />

      <ExampleBlock title="RMSNorm in Modern LLMs">
        <p>
          LLaMA, Mistral, Gemma, and most recent LLMs use RMSNorm instead of LayerNorm. The
          savings compound across billions of tokens and hundreds of layers:
          for a 70B parameter model, RMSNorm saves significant compute per forward pass.
        </p>
      </ExampleBlock>

      <PythonCode
        title="RMSNorm & GroupNorm Implementation"
        code={`import torch
import torch.nn as nn

# RMSNorm (not built into PyTorch, easy to implement)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# Usage comparison
d_model = 512
rmsnorm = RMSNorm(d_model)
layernorm = nn.LayerNorm(d_model)
groupnorm = nn.GroupNorm(num_groups=32, num_channels=d_model)

x = torch.randn(8, 10, d_model)
print(f"RMSNorm:   {rmsnorm(x).shape}")
print(f"LayerNorm: {layernorm(x).shape}")

# GroupNorm for conv features (B, C, H, W)
x_conv = torch.randn(8, d_model, 16, 16)
print(f"GroupNorm: {groupnorm(x_conv).shape}")

# GroupNorm works with any batch size
x_single = torch.randn(1, d_model, 16, 16)
print(f"GroupNorm batch=1: {groupnorm(x_single).shape}")`}
      />

      <NoteBlock type="note" title="Choosing the Right Norm">
        <p>
          <strong>RMSNorm</strong>: LLMs and Transformers (fast, effective).
          <strong> GroupNorm</strong>: CNNs with small batch sizes (detection, segmentation).
          <strong> LayerNorm</strong>: General Transformers. <strong>BatchNorm</strong>: CNNs with
          large batches (classification). The trend is clearly toward simpler norms.
        </p>
      </NoteBlock>
    </div>
  )
}
