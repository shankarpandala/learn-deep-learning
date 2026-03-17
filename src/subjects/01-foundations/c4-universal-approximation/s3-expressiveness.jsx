import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import ExerciseBlock from '../../../components/content/ExerciseBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function VCDimensionViz() {
  const [points, setPoints] = useState(3)
  const W = 420, H = 220

  const configs = Math.pow(2, points)
  const maxShown = Math.min(configs, 16)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Shattering & VC Dimension</h3>
      <label className="flex items-center gap-2 mb-3 text-sm text-gray-600 dark:text-gray-400">
        Points: {points}
        <input type="range" min={1} max={5} step={1} value={points} onChange={e => setPoints(parseInt(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: maxShown }, (_, ci) => {
          const row = Math.floor(ci / 4)
          const col = ci % 4
          const bx = col * 105 + 10
          const by = row * 50 + 10
          return (
            <g key={ci}>
              <line x1={bx} y1={by + 20} x2={bx + 90} y2={by + 20} stroke="#d1d5db" strokeWidth={0.5} />
              {Array.from({ length: points }, (_, pi) => {
                const px = bx + 15 + pi * (60 / Math.max(points - 1, 1))
                const label = (ci >> pi) & 1
                return (
                  <circle key={pi} cx={px} cy={by + 20} r={5}
                    fill={label ? '#8b5cf6' : '#f97316'} stroke="white" strokeWidth={1} />
                )
              })}
              <text x={bx + 45} y={by + 38} fontSize={8} fill="#9ca3af" textAnchor="middle">
                {ci.toString(2).padStart(points, '0')}
              </text>
            </g>
          )
        })}
        {configs > maxShown && (
          <text x={W / 2} y={H - 5} fontSize={10} fill="#9ca3af" textAnchor="middle">
            ...and {configs - maxShown} more labelings ({configs} total)
          </text>
        )}
      </svg>
      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400 text-center">
        A hypothesis class shatters {points} points if it can realize all {configs} labelings.
        {points <= 3 ? ' A linear classifier in 2D can shatter up to 3 points (VC dim = 3).' : ' A linear classifier cannot shatter this many collinear points.'}
      </p>
    </div>
  )
}

export default function Expressiveness() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        How powerful is a neural network as a function class? Expressiveness theory
        quantifies the capacity of hypothesis classes through VC dimension and Rademacher
        complexity, connecting what networks can represent to how well they generalize.
      </p>

      <DefinitionBlock title="VC Dimension">
        <p>
          The Vapnik-Chervonenkis dimension of a hypothesis class <InlineMath math="\mathcal{H}" /> is
          the largest number of points that can be <em>shattered</em> by <InlineMath math="\mathcal{H}" />:
        </p>
        <BlockMath math="\text{VC}(\mathcal{H}) = \max\{n : \Pi_{\mathcal{H}}(n) = 2^n\}" />
        <p className="mt-2">
          where <InlineMath math="\Pi_{\mathcal{H}}(n)" /> is the growth function counting the
          maximum number of distinct labelings achievable on any <InlineMath math="n" /> points.
        </p>
      </DefinitionBlock>

      <ExampleBlock title="VC Dimension of Linear Classifiers">
        <p>
          A linear classifier in <InlineMath math="\mathbb{R}^d" /> (hyperplane) has
          VC dimension <InlineMath math="d + 1" />. In 2D, we can shatter any 3 non-collinear
          points but no set of 4 points. For a single neuron
          with <InlineMath math="d" /> inputs, <InlineMath math="\text{VC} = d + 1" />.
        </p>
      </ExampleBlock>

      <VCDimensionViz />

      <TheoremBlock title="VC Dimension of Neural Networks" id="vc-nn">
        <p>
          For a ReLU network with <InlineMath math="W" /> total weights
          and <InlineMath math="L" /> layers:
        </p>
        <BlockMath math="\text{VC}(\mathcal{H}) = O(WL \log W)" />
        <p>
          This means capacity grows nearly linearly with the number of parameters. However,
          modern overparameterized networks have VC dimension far exceeding the training set
          size, yet still generalize well, which classical VC theory cannot explain.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Rademacher Complexity">
        <p>Measures how well a function class can fit random noise:</p>
        <BlockMath math="\hat{\mathcal{R}}_n(\mathcal{H}) = \mathbb{E}_{\sigma}\left[\sup_{h \in \mathcal{H}} \frac{1}{n}\sum_{i=1}^{n} \sigma_i h(x_i)\right]" />
        <p className="mt-2">
          where <InlineMath math="\sigma_i \in \{-1, +1\}" /> are i.i.d. Rademacher variables.
          High Rademacher complexity means the class can memorize random labels.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Generalization Bound" id="gen-bound">
        <p>For any <InlineMath math="h \in \mathcal{H}" />, with probability at least <InlineMath math="1 - \delta" />:</p>
        <BlockMath math="R(h) \leq \hat{R}_n(h) + 2\mathcal{R}_n(\mathcal{H}) + \sqrt{\frac{\log(1/\delta)}{2n}}" />
        <p>The gap between true risk <InlineMath math="R(h)" /> and empirical risk <InlineMath math="\hat{R}_n(h)" /> shrinks with more data and lower complexity.</p>
      </TheoremBlock>

      <WarningBlock title="The Generalization Puzzle">
        <p>
          Modern deep networks can memorize random labels (Zhang et al., 2017), yet generalize
          well on real data. This shows classical capacity measures are too coarse. The
          explanation lies in implicit regularization: SGD, architecture, and data structure
          conspire to find simple solutions within the vast hypothesis space.
        </p>
      </WarningBlock>

      <PythonCode
        title="Memorization vs Generalization Experiment"
        code={`import torch
import torch.nn as nn

torch.manual_seed(42)
n, d = 200, 10
X = torch.randn(n, d)
y_real = (X[:, 0] + X[:, 1] > 0).float()   # structured labels
y_rand = torch.randint(0, 2, (n,)).float()  # random labels

def make_net():
    return nn.Sequential(
        nn.Linear(d, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 1), nn.Sigmoid())

def train(model, X, y, epochs=2000):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        pred = model(X).squeeze()
        loss = nn.BCELoss()(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = ((pred > 0.5).float() == y).float().mean()
    return loss.item(), acc.item()

loss_r, acc_r = train(make_net(), X, y_real)
print(f"Real labels   -> Loss: {loss_r:.4f}, Acc: {acc_r:.2%}")

loss_n, acc_n = train(make_net(), X, y_rand)
print(f"Random labels -> Loss: {loss_n:.4f}, Acc: {acc_n:.2%}")
# Both achieve ~100% training accuracy!`}
      />

      <ExerciseBlock title="Capacity vs Generalization">
        <p>
          A network with 10M parameters trained on 1000 samples achieves 95% test accuracy.
          VC theory gives a vacuous bound. What mechanisms explain the good generalization?
          Consider: implicit bias of SGD, weight decay, batch normalization, dropout, and
          the low intrinsic dimensionality of the loss landscape.
        </p>
      </ExerciseBlock>

      <NoteBlock type="note" title="Beyond Classical Theory">
        <p>
          Active research bridging theory and practice includes PAC-Bayes bounds, norm-based
          bounds (depending on weight norms rather than parameter count), neural tangent kernel
          theory, and compression-based bounds. Generalization in deep learning remains one of
          the field&apos;s most important open problems.
        </p>
      </NoteBlock>
    </div>
  )
}
