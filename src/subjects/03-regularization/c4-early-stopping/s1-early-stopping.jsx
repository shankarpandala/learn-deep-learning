import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EarlyStoppingPlot() {
  const [patience, setPatience] = useState(10)
  const W = 400, H = 220

  const trainLoss = (e) => 1.5 * Math.exp(-0.035 * e) + 0.05
  const valLoss = (e) => 1.5 * Math.exp(-0.025 * e) + 0.2 + 0.002 * Math.max(0, e - 25)

  const epochs = Array.from({ length: 100 }, (_, i) => i + 1)
  const toSVG = (x, y) => `${25 + (x / 100) * (W - 50)},${H - 25 - y * (H - 45) / 1.8}`

  let bestEpoch = 1, bestVal = Infinity
  for (let e = 1; e <= 100; e++) {
    const v = valLoss(e)
    if (v < bestVal) { bestVal = v; bestEpoch = e }
  }

  const stopEpoch = Math.min(100, bestEpoch + patience)

  const trainPath = epochs.filter(e => e <= stopEpoch).map((e, i) => `${i === 0 ? 'M' : 'L'}${toSVG(e, trainLoss(e))}`).join(' ')
  const valPath = epochs.filter(e => e <= stopEpoch).map((e, i) => `${i === 0 ? 'M' : 'L'}${toSVG(e, valLoss(e))}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Early Stopping Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Patience: {patience}
        <input type="range" min={1} max={40} step={1} value={patience} onChange={e => setPatience(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={25} y1={H - 25} x2={W - 25} y2={H - 25} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={trainPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={valPath} fill="none" stroke="#f97316" strokeWidth={2.5} />
        <line x1={25 + (bestEpoch / 100) * (W - 50)} y1={5} x2={25 + (bestEpoch / 100) * (W - 50)} y2={H - 25} stroke="#22c55e" strokeWidth={1.5} strokeDasharray="4,4" />
        <line x1={25 + (stopEpoch / 100) * (W - 50)} y1={5} x2={25 + (stopEpoch / 100) * (W - 50)} y2={H - 25} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="4,4" />
        <text x={25 + (bestEpoch / 100) * (W - 50)} y={15} textAnchor="middle" fontSize={9} fill="#22c55e">best</text>
        <text x={25 + (stopEpoch / 100) * (W - 50)} y={15} textAnchor="middle" fontSize={9} fill="#ef4444">stop</text>
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Train</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Val</span>
        <span>Best epoch: {bestEpoch}</span>
        <span>Stop epoch: {stopEpoch}</span>
      </div>
    </div>
  )
}

export default function EarlyStopping() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Early stopping halts training when validation performance stops improving,
        preventing the model from overfitting. It acts as a form of implicit regularization
        by limiting the effective complexity of the learned function.
      </p>

      <DefinitionBlock title="Early Stopping">
        <p>
          Monitor a validation metric after each epoch. If the metric has not improved
          for <InlineMath math="P" /> consecutive epochs (the patience), stop training and
          restore the model weights from the best epoch.
        </p>
      </DefinitionBlock>

      <EarlyStoppingPlot />

      <TheoremBlock title="Early Stopping as Regularization" id="early-stopping-reg">
        <p>For linear models with gradient descent, early stopping after <InlineMath math="t" /> steps with learning rate <InlineMath math="\eta" /> is equivalent to L2 regularization with:</p>
        <BlockMath math="\lambda_{\text{eff}} \approx \frac{1}{\eta t}" />
        <p className="mt-2">Fewer training steps correspond to stronger regularization, limiting how far weights can move from initialization.</p>
      </TheoremBlock>

      <ExampleBlock title="Checkpointing Strategy">
        <ul className="list-disc ml-4 space-y-1">
          <li>Save model weights whenever validation loss reaches a new minimum</li>
          <li>Track the best metric value and the epoch it occurred</li>
          <li>After stopping, load the best checkpoint (not the final weights)</li>
          <li>Typical patience values: 5-20 epochs depending on dataset size</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Early Stopping with Checkpointing"
        code={`import torch
import torch.nn as nn

class EarlyStoppingTrainer:
    def __init__(self, model, patience=10, min_delta=1e-4):
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def check(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            return False  # continue training
        self.counter += 1
        return self.counter >= self.patience  # stop if True

    def restore_best(self):
        if self.best_state:
            self.model.load_state_dict(self.best_state)

# Usage
model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
trainer = EarlyStoppingTrainer(model, patience=10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(500):
    train_loss = nn.MSELoss()(model(torch.randn(32, 10)), torch.randn(32, 1))
    optimizer.zero_grad(); train_loss.backward(); optimizer.step()
    with torch.no_grad():
        val_loss = nn.MSELoss()(model(torch.randn(32, 10)), torch.randn(32, 1))
    if trainer.check(val_loss.item()):
        print(f"Early stop at epoch {epoch}, best loss: {trainer.best_loss:.4f}")
        trainer.restore_best()
        break`}
      />

      <NoteBlock type="note" title="Patience Selection">
        <p>
          Too small a patience may stop too early (missing further improvements after a plateau).
          Too large wastes compute. A good heuristic: set patience to 10-20% of expected total
          training epochs. Also consider using a learning rate scheduler before early stopping
          to give the optimizer a chance to escape local minima.
        </p>
      </NoteBlock>
    </div>
  )
}
