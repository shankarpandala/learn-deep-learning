import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DoublDescentPlot() {
  const [paramScale, setParamScale] = useState(50)
  const W = 420, H = 220

  const risk = (p) => {
    if (p < 40) return 1.2 - 0.015 * p
    if (p < 60) return 0.6 + 0.04 * (p - 40)
    return 1.4 * Math.exp(-0.02 * (p - 60)) + 0.2
  }

  const trainRisk = (p) => {
    if (p < 50) return Math.max(0.01, 0.8 - 0.016 * p)
    return 0.01
  }

  const points = Array.from({ length: 100 }, (_, i) => i + 1)
  const toSVG = (x, y) => `${25 + (x / 100) * (W - 50)},${H - 25 - y * (H - 45) / 2}`

  const testPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toSVG(p, risk(p))}`).join(' ')
  const trainPath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toSVG(p, trainRisk(p))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Double Descent Curve</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Model Parameters: {paramScale}x
        <input type="range" min={1} max={100} step={1} value={paramScale} onChange={e => setParamScale(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={25} y1={H - 25} x2={W - 25} y2={H - 25} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={25} y1={5} x2={25} y2={H - 25} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={25 + (50 / 100) * (W - 50)} y1={5} x2={25 + (50 / 100) * (W - 50)} y2={H - 25} stroke="#ef4444" strokeWidth={0.8} strokeDasharray="4,4" opacity={0.5} />
        <text x={25 + (50 / 100) * (W - 50)} y={15} textAnchor="middle" fontSize={9} fill="#ef4444">interpolation</text>
        <path d={trainPath} fill="none" stroke="#8b5cf6" strokeWidth={2} strokeDasharray="4,4" />
        <path d={testPath} fill="none" stroke="#f97316" strokeWidth={2.5} />
        <line x1={25 + (paramScale / 100) * (W - 50)} y1={5} x2={25 + (paramScale / 100) * (W - 50)} y2={H - 25} stroke="#9ca3af" strokeWidth={0.8} strokeDasharray="3,3" />
        <circle cx={25 + (paramScale / 100) * (W - 50)} cy={parseFloat(toSVG(paramScale, risk(paramScale)).split(',')[1])} r={4} fill="#f97316" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Test Risk: {risk(paramScale).toFixed(3)}</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500 opacity-60" style={{ borderTop: '1px dashed' }} /> Train Risk</span>
      </div>
    </div>
  )
}

export default function DoubleDescent() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The double descent phenomenon challenges the classical U-shaped bias-variance curve.
        As model complexity increases past the interpolation threshold, test error can actually
        <em> decrease again</em>, contradicting traditional wisdom.
      </p>

      <DefinitionBlock title="Interpolation Threshold">
        <p>
          The interpolation threshold is the point where the model has just enough parameters
          to perfectly fit (<InlineMath math="\hat{f}(x_i) = y_i" /> for all training points).
          At this threshold, the model is maximally sensitive to noise, causing a peak in test error.
        </p>
      </DefinitionBlock>

      <DoublDescentPlot />

      <TheoremBlock title="Double Descent Regions" id="double-descent-regions">
        <p>The test risk curve has three distinct regions:</p>
        <BlockMath math="\text{Risk}(p) = \begin{cases} \text{decreasing (classical)} & p \ll n \\ \text{peak at interpolation} & p \approx n \\ \text{decreasing (modern)} & p \gg n \end{cases}" />
        <p className="mt-2">
          where <InlineMath math="p" /> is the number of parameters and <InlineMath math="n" /> is the
          number of training samples.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Why Overparameterization Helps">
        <p>
          With <InlineMath math="p \gg n" />, there are many solutions that interpolate the training data.
          SGD and implicit regularization select the smoothest among these, which generalizes well.
          This is why modern networks with billions of parameters can still generalize.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Observing Double Descent with Varying Width"
        code={`import torch
import torch.nn as nn

n_train, n_test, d = 50, 200, 20
x_train = torch.randn(n_train, d)
y_train = (x_train[:, 0] > 0).float().unsqueeze(1)
x_test = torch.randn(n_test, d)
y_test = (x_test[:, 0] > 0).float().unsqueeze(1)

for width in [10, 50, 100, 500, 2000]:
    model = nn.Sequential(nn.Linear(d, width), nn.ReLU(), nn.Linear(width, 1), nn.Sigmoid())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(2000):
        loss = nn.BCELoss()(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        test_loss = nn.BCELoss()(model(x_test), y_test)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Width={width:4d} Params={n_params:6d} TestLoss={test_loss:.4f}")`}
      />

      <NoteBlock type="note" title="Epoch-Wise Double Descent">
        <p>
          Double descent also occurs along the training time axis: test error can first decrease,
          then increase (classical overfitting), then decrease again with longer training.
          This is called <strong>epoch-wise double descent</strong> and is especially pronounced
          with label noise.
        </p>
      </NoteBlock>
    </div>
  )
}
