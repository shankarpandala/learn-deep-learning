import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ScalingLawPlot() {
  const [lawType, setLawType] = useState('kaplan')
  const W = 400, H = 200, pad = 40

  const laws = {
    kaplan: { name: 'Kaplan (OpenAI)', alpha: 0.076, label: 'L(N) = (N_c/N)^0.076', color: '#8b5cf6' },
    chinchilla: { name: 'Chinchilla (DeepMind)', alpha: 0.10, label: 'L(N) = (N_c/N)^0.10', color: '#8b5cf6' },
  }
  const law = laws[lawType]

  const points = Array.from({ length: 50 }, (_, i) => {
    const logN = 6 + i * 0.12
    const N = Math.pow(10, logN)
    const loss = 2.0 * Math.pow(1e13 / N, law.alpha) + 1.5
    return { logN, loss }
  })

  const xScale = (v) => pad + (v - 6) / 6 * (W - 2 * pad)
  const yScale = (v) => H - pad - (v - 1.5) / 1.2 * (H - 2 * pad)
  const path = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(p.logN)},${yScale(p.loss)}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Neural Scaling Law: Loss vs Parameters</h3>
      <div className="flex gap-2 mb-3">
        {Object.entries(laws).map(([key, val]) => (
          <button key={key} onClick={() => setLawType(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${lawType === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path} fill="none" stroke={law.color} strokeWidth={2.5} />
        <text x={W / 2} y={H - 5} textAnchor="middle" className="text-[10px] fill-gray-500">log10(Parameters)</text>
        <text x={12} y={H / 2} textAnchor="middle" transform={`rotate(-90,12,${H / 2})`} className="text-[10px] fill-gray-500">Loss</text>
      </svg>
      <p className="mt-1 text-xs text-gray-500 text-center">{law.label} — loss decreases as a power law with model size</p>
    </div>
  )
}

export default function KaplanChinchillaScaling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Neural scaling laws describe predictable power-law relationships between model performance
        and the key resources: parameters, training data, and compute. These laws enable
        principled decisions about resource allocation before training.
      </p>

      <TheoremBlock title="Kaplan Scaling Laws (2020)" id="kaplan-scaling">
        <p>Cross-entropy loss follows power laws in each scaling variable independently:</p>
        <BlockMath math="L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}" />
        <p className="mt-2">With <InlineMath math="\alpha_N \approx 0.076" />, <InlineMath math="\alpha_D \approx 0.095" />, <InlineMath math="\alpha_C \approx 0.050" />. These exponents suggest model size matters more than data quantity — a conclusion later revised by Chinchilla.</p>
      </TheoremBlock>

      <ScalingLawPlot />

      <DefinitionBlock title="Chinchilla Optimal Allocation">
        <p>For a fixed compute budget <InlineMath math="C" />, Chinchilla showed that parameters and tokens should scale equally:</p>
        <BlockMath math="L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}" />
        <p className="mt-2">with <InlineMath math="\alpha \approx 0.34" />, <InlineMath math="\beta \approx 0.28" />, and <InlineMath math="E" /> is the irreducible entropy. Minimizing <InlineMath math="L" /> subject to <InlineMath math="C = 6ND" /> gives <InlineMath math="N^* \propto C^{0.5}" /> and <InlineMath math="D^* \propto C^{0.5}" />.</p>
      </DefinitionBlock>

      <PythonCode
        title="Fitting and Predicting with Scaling Laws"
        code={`import numpy as np

def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """Chinchilla parametric loss model.

    Args:
        N: number of parameters
        D: number of training tokens
        A, B, alpha, beta, E: fitted constants from Chinchilla paper
    """
    return E + A / N**alpha + B / D**beta

def optimal_allocation(C, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Find optimal N, D for compute budget C (FLOPs = 6*N*D)."""
    # Analytical solution from Lagrange multiplier
    a = alpha
    b = beta
    # N* proportional to C^(b/(a+b)), D* proportional to C^(a/(a+b))
    ratio = (a * B) / (b * A)
    N_star = (C / 6 * ratio**(b/(a+b)))**(1/(1 + b/a))
    D_star = C / (6 * N_star)
    return N_star, D_star

# Predict loss for different compute budgets
for log_c in [21, 22, 23, 24, 25]:
    C = 10**log_c
    N, D = optimal_allocation(C)
    loss = chinchilla_loss(N, D)
    print(f"C=10^{log_c}: N={N/1e9:.1f}B, D={D/1e9:.0f}B tokens, Loss={loss:.3f}")

# Compare: GPT-3 (undertrained) vs Chinchilla-optimal
gpt3_loss = chinchilla_loss(175e9, 300e9)
chin_loss = chinchilla_loss(70e9, 1.4e12)
print(f"\\nGPT-3 (175B, 300B tok): {gpt3_loss:.3f}")
print(f"Chinchilla (70B, 1.4T tok): {chin_loss:.3f}")`}
      />

      <ExampleBlock title="Key Takeaways from Scaling Laws">
        <ul className="list-disc list-inside space-y-1">
          <li>Loss follows smooth, predictable power laws across many orders of magnitude</li>
          <li>Small-scale experiments can predict large-scale performance</li>
          <li>Kaplan overestimated the importance of model size (fixed training tokens)</li>
          <li>Chinchilla showed data and parameters should scale equally with compute</li>
          <li>Modern practice (LLaMA-3) deliberately over-trains for inference efficiency</li>
        </ul>
      </ExampleBlock>

      <NoteBlock type="note" title="Scaling Laws Beyond Language">
        <p>
          Power-law scaling has been observed across modalities: vision (ViT), speech, code generation,
          mathematical reasoning, and multimodal models. However, the exponents and constants vary.
          Downstream task performance sometimes deviates from training loss scaling, particularly
          for tasks requiring specific capabilities that may emerge suddenly.
        </p>
      </NoteBlock>
    </div>
  )
}
