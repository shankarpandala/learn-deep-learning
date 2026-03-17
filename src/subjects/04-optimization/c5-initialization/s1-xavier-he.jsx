import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import ProofBlock from '../../../components/content/ProofBlock.jsx'

function VariancePropViz() {
  const [fanIn, setFanIn] = useState(256)
  const [initType, setInitType] = useState('xavier')
  const W = 380, H = 180, layers = 10

  const variances = [1.0]
  for (let l = 1; l <= layers; l++) {
    const prev = variances[l - 1]
    let wVar
    if (initType === 'xavier') wVar = 2.0 / (fanIn + fanIn)
    else if (initType === 'he') wVar = 2.0 / fanIn
    else wVar = 1.0 / fanIn
    const activationFactor = initType === 'he' ? 0.5 : 1.0
    variances.push(prev * fanIn * wVar * activationFactor)
  }

  const maxVar = Math.max(...variances) * 1.2
  const minVar = Math.min(...variances.filter(v => v > 0))
  const sx = W / (layers + 1), barW = sx * 0.7

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Activation Variance Through Layers</h3>
      <div className="flex items-center gap-4 mb-3 flex-wrap">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          fan_in: {fanIn}
          <input type="range" min={32} max={1024} step={32} value={fanIn} onChange={e => setFanIn(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <div className="flex gap-2">
          {['xavier', 'he', 'naive'].map(t => (
            <button key={t} onClick={() => setInitType(t)}
              className={`px-3 py-1 rounded text-xs font-medium ${initType === t ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
              {t === 'naive' ? '1/n' : t}
            </button>
          ))}
        </div>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H - 20} x2={W} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={0} y1={H - 20 - (1.0 / maxVar) * (H - 40)} x2={W} y2={H - 20 - (1.0 / maxVar) * (H - 40)} stroke="#f97316" strokeWidth={0.8} strokeDasharray="3,3" />
        {variances.map((v, i) => {
          const h = Math.min((v / maxVar) * (H - 40), H - 30)
          return (
            <g key={i}>
              <rect x={i * sx + (sx - barW) / 2} y={H - 20 - h} width={barW} height={h} fill="#8b5cf6" rx={3} opacity={0.7} />
              <text x={i * sx + sx / 2} y={H - 7} textAnchor="middle" fill="#6b7280" fontSize={8}>L{i}</text>
            </g>
          )
        })}
      </svg>
      <div className="mt-1 text-center text-xs text-gray-500">Orange dashed = ideal variance (1.0)</div>
    </div>
  )
}

export default function XavierHe() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Proper weight initialization ensures that activations and gradients maintain stable
        variance through the network. Xavier init targets linear/tanh activations, while He init
        accounts for the ReLU non-linearity.
      </p>

      <DefinitionBlock title="Xavier (Glorot) Initialization">
        <BlockMath math="W \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right) \quad \text{or} \quad W \sim U\!\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]" />
        <p className="mt-2">
          Designed for linear or tanh activations. Preserves variance in both forward and backward passes.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="He (Kaiming) Initialization">
        <BlockMath math="W \sim \mathcal{N}\!\left(0, \frac{2}{n_{\text{in}}}\right)" />
        <p className="mt-2">
          Accounts for ReLU zeroing out half the activations. The factor of 2 compensates for
          the <InlineMath math="1/2" /> reduction in variance from ReLU.
        </p>
      </DefinitionBlock>

      <VariancePropViz />

      <ProofBlock title="Derivation Sketch (He Init)">
        <p>For a layer <InlineMath math="y = Wx" /> followed by ReLU:</p>
        <BlockMath math="\text{Var}(y_j) = n_{\text{in}} \cdot \text{Var}(w) \cdot \text{Var}(x)" />
        <p>ReLU zeroes negative half, so <InlineMath math="\text{Var}(\text{ReLU}(y)) = \frac{1}{2}\text{Var}(y)" />.</p>
        <p>Setting <InlineMath math="\text{Var}(w) = 2/n_{\text{in}}" /> gives <InlineMath math="\text{Var}(\text{ReLU}(y)) = \text{Var}(x)" />.</p>
      </ProofBlock>

      <ExampleBlock title="When to Use Each">
        <p>
          <strong>Xavier</strong>: sigmoid, tanh, linear layers, SELU.
          <strong> He</strong>: ReLU, Leaky ReLU, ELU, GELU. Using He init with tanh will
          cause exploding activations; using Xavier with ReLU will cause dying neurons.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Xavier & He Init in PyTorch"
        code={`import torch
import torch.nn as nn

layer = nn.Linear(512, 256)

# Xavier (Glorot) — for tanh/sigmoid
nn.init.xavier_uniform_(layer.weight)    # uniform variant
nn.init.xavier_normal_(layer.weight)     # normal variant

# He (Kaiming) — for ReLU
nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')

# Verify variance
w = layer.weight.data
print(f"Weight std: {w.std().item():.4f}")
print(f"Expected (He): {(2/512)**0.5:.4f}")

# Initialize full model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
model.apply(init_weights)
print("Model initialized with He init")`}
      />

      <NoteBlock type="note" title="Modern Frameworks Handle This">
        <p>
          PyTorch uses Kaiming uniform by default for linear and conv layers. You rarely need
          to manually initialize unless using custom architectures. However, understanding the
          theory helps diagnose training instabilities in deep or unusual networks.
        </p>
      </NoteBlock>
    </div>
  )
}
