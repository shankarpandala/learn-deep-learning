import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import ProofBlock from '../../../components/content/ProofBlock.jsx'

function sigmoid(x) { return 1 / (1 + Math.exp(-x)) }

function ApproximationDemo() {
  const [neurons, setNeurons] = useState(3)
  const W = 420, H = 250, padL = 30, padB = 25
  const plotW = W - padL, plotH = H - padB

  // Target function: sin(2*pi*x) on [0,1]
  const target = x => Math.sin(2 * Math.PI * x)

  // Random but deterministic weights (seeded by neuron index)
  function getParams(n) {
    const params = []
    for (let i = 0; i < n; i++) {
      const w = 4 + i * 3.5
      const b = -w * (i / Math.max(n - 1, 1))
      const a = (i % 2 === 0 ? 1 : -1) * 2.0 / n
      params.push({ w, b, a })
    }
    return params
  }

  const params = getParams(neurons)
  const approx = x => params.reduce((sum, { w, b, a }) => sum + a * sigmoid(w * x + b), 0)

  const range = Array.from({ length: 200 }, (_, i) => i / 199)
  const toSVG = (x, y) => `${padL + x * plotW},${plotH / 2 - y * (plotH / 3)}`

  const targetPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, target(x))}`).join(' ')
  const approxPath = range.map((x, i) => `${i === 0 ? 'M' : 'L'}${toSVG(x, approx(x))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Approximating sin(2 pi x) with Sigmoid Neurons</h3>
      <label className="flex items-center gap-2 mb-3 text-sm text-gray-600 dark:text-gray-400">
        Hidden neurons: {neurons}
        <input type="range" min={1} max={20} step={1} value={neurons} onChange={e => setNeurons(parseInt(e.target.value))} className="w-36 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={padL} y1={plotH / 2} x2={W} y2={plotH / 2} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={padL} y1={0} x2={padL} y2={plotH} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={targetPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={approxPath} fill="none" stroke="#f97316" strokeWidth={2} strokeDasharray="5,3" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Target f(x)</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Approximation ({neurons} neurons)</span>
      </div>
    </div>
  )
}

export default function UniversalApproximation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Universal Approximation Theorem (UAT) is one of the most important theoretical
        results in neural network theory. It guarantees that even a single hidden layer network
        can approximate any continuous function to arbitrary precision.
      </p>

      <TheoremBlock title="Cybenko's Theorem (1989)" id="cybenko">
        <p>
          Let <InlineMath math="\sigma" /> be any continuous sigmoidal function. Then finite sums
          of the form:
        </p>
        <BlockMath math="G(x) = \sum_{j=1}^{N} \alpha_j \sigma(w_j^T x + b_j)" />
        <p>
          are dense in <InlineMath math="C(I_n)" />, the space of continuous functions on the
          unit hypercube <InlineMath math="I_n = [0,1]^n" />. That is, for any <InlineMath math="f \in C(I_n)" /> and
          any <InlineMath math="\varepsilon > 0" />, there exist <InlineMath math="N, \alpha_j, w_j, b_j" /> such
          that <InlineMath math="|G(x) - f(x)| < \varepsilon" /> for all <InlineMath math="x \in I_n" />.
        </p>
      </TheoremBlock>

      <TheoremBlock title="Hornik's Theorem (1991)" id="hornik">
        <p>
          The universality result extends beyond sigmoids. A single hidden layer feedforward
          network is a universal approximator if and only if the activation function is
          non-polynomial. This includes ReLU, tanh, sigmoid, and many others.
        </p>
        <BlockMath math="\sigma \text{ is non-polynomial} \iff \overline{\text{span}\{\sigma(w^Tx + b)\}} = C(K)" />
      </TheoremBlock>

      <ProofBlock title="Proof Sketch (Cybenko)">
        <p>
          The proof uses the Hahn-Banach theorem. Assume for contradiction that the span
          of <InlineMath math="\sigma(w^T x + b)" /> is not dense. Then there exists a nonzero
          bounded linear functional <InlineMath math="\mu" /> (a signed measure) such that:
        </p>
        <BlockMath math="\int_{I_n} \sigma(w^T x + b) \, d\mu(x) = 0 \quad \forall w, b" />
        <p>
          One then shows this forces <InlineMath math="\mu = 0" />, a contradiction. The key step
          uses properties of the Fourier transform and the sigmoidal nature
          of <InlineMath math="\sigma" />.
        </p>
      </ProofBlock>

      <ApproximationDemo />

      <WarningBlock title="Existence vs. Construction">
        <p>
          The UAT is an existence theorem: it says an approximating network exists but tells
          us nothing about how to find the weights (SGD may fail), how many neurons are needed
          (could be astronomically large), or whether the resulting network will generalize.
          The gap between approximation theory and practical deep learning is enormous.
        </p>
      </WarningBlock>

      <PythonCode
        title="Verifying UAT: Fitting an Arbitrary Function"
        code={`import torch
import torch.nn as nn
import numpy as np

# Target: a complex non-smooth function
def target_fn(x):
    return np.sin(3 * x) + 0.5 * np.sign(x) + 0.3 * np.cos(7 * x)

x = torch.linspace(-3, 3, 500).unsqueeze(1)
y = torch.tensor(target_fn(x.numpy()), dtype=torch.float32)

# Single hidden layer network (UAT architecture)
model = nn.Sequential(
    nn.Linear(1, 128),
    nn.Sigmoid(),
    nn.Linear(128, 1)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(2000):
    pred = model(x)
    loss = nn.MSELoss()(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1}: MSE = {loss.item():.6f}")

# With enough neurons, loss -> 0 (UAT guarantee)`}
      />

      <ExampleBlock title="Width Required for Epsilon-Approximation">
        <p>
          For a Lipschitz-continuous function <InlineMath math="f: [0,1]^d \to \mathbb{R}" /> with
          Lipschitz constant <InlineMath math="L" />, the number of neurons needed for
          epsilon-approximation scales as:
        </p>
        <BlockMath math="N = O\left(\left(\frac{L}{\varepsilon}\right)^d\right)" />
        <p>
          This exponential dependence on dimension <InlineMath math="d" /> is the curse of
          dimensionality. Depth can overcome this limitation in many practical cases.
        </p>
      </ExampleBlock>

      <NoteBlock type="note" title="Modern Perspective">
        <p>
          The UAT explains why neural networks can work but not why they work so well in
          practice. Modern theory focuses on the implicit bias of SGD, the lottery ticket
          hypothesis, and the neural tangent kernel to explain generalization.
          The depth efficiency results (next section) explain why deep networks are preferred.
        </p>
      </NoteBlock>
    </div>
  )
}
