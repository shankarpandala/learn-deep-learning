import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TrainingCurvePlot() {
  const [epoch, setEpoch] = useState(60)
  const W = 400, H = 220

  const trainLoss = (e) => 1.8 * Math.exp(-0.04 * e) + 0.05
  const valLoss = (e) => 1.8 * Math.exp(-0.03 * e) + 0.15 + 0.003 * Math.max(0, e - 30)

  const epochs = Array.from({ length: 100 }, (_, i) => i + 1)
  const toSVG = (x, y) => `${20 + (x / 100) * (W - 40)},${H - 20 - y * (H - 40) / 2}`

  const trainPath = epochs.map((e, i) => `${i === 0 ? 'M' : 'L'}${toSVG(e, trainLoss(e))}`).join(' ')
  const valPath = epochs.map((e, i) => `${i === 0 ? 'M' : 'L'}${toSVG(e, valLoss(e))}`).join(' ')

  const gap = valLoss(epoch) - trainLoss(epoch)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Training vs Validation Loss</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Epoch: {epoch}
        <input type="range" min={1} max={100} step={1} value={epoch} onChange={e => setEpoch(parseInt(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={20} y1={H - 20} x2={W - 20} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={20} y1={0} x2={20} y2={H - 20} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={trainPath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <path d={valPath} fill="none" stroke="#f97316" strokeWidth={2.5} />
        <line x1={20 + (epoch / 100) * (W - 40)} y1={0} x2={20 + (epoch / 100) * (W - 40)} y2={H - 20} stroke="#9ca3af" strokeWidth={0.8} strokeDasharray="3,3" />
      </svg>
      <div className="mt-2 flex justify-center gap-6 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Train: {trainLoss(epoch).toFixed(3)}</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-orange-500" /> Val: {valLoss(epoch).toFixed(3)}</span>
        <span className="font-semibold text-violet-600">Gap: {gap.toFixed(3)}</span>
      </div>
    </div>
  )
}

export default function OverfittingDetection() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Detecting overfitting early is crucial for training effective models. The primary
        diagnostic tool is monitoring training and validation loss curves throughout training.
      </p>

      <DefinitionBlock title="Generalization Gap">
        <BlockMath math="\text{Gap} = \mathcal{L}_{\text{val}} - \mathcal{L}_{\text{train}}" />
        <p className="mt-2">
          A growing gap between validation and training loss signals overfitting.
          The model is memorizing training data instead of learning general patterns.
        </p>
      </DefinitionBlock>

      <TrainingCurvePlot />

      <ExampleBlock title="Diagnostic Checklist">
        <ul className="list-disc ml-4 space-y-1">
          <li><strong>Both losses high</strong>: underfitting (increase capacity or train longer)</li>
          <li><strong>Train low, val high</strong>: overfitting (add regularization)</li>
          <li><strong>Both losses low and close</strong>: good fit</li>
          <li><strong>Val loss oscillates</strong>: learning rate may be too high</li>
        </ul>
      </ExampleBlock>

      <WarningBlock title="Common Pitfall: Data Leakage">
        <p>
          If validation loss is <em>lower</em> than training loss, suspect data leakage
          or incorrect data splitting. Ensure no overlap between train and validation sets,
          and that preprocessing is fit only on training data.
        </p>
      </WarningBlock>

      <PythonCode
        title="Tracking Overfitting in PyTorch"
        code={`import torch
import torch.nn as nn

model = nn.Sequential(nn.Linear(10, 128), nn.ReLU(), nn.Linear(128, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Simulated data
x_train, y_train = torch.randn(500, 10), torch.randn(500, 1)
x_val, y_val = torch.randn(100, 10), torch.randn(100, 1)

best_val_loss = float('inf')
patience_counter = 0

for epoch in range(200):
    model.train()
    train_loss = criterion(model(x_train), y_train)
    optimizer.zero_grad(); train_loss.backward(); optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = criterion(model(x_val), y_val)

    gap = val_loss.item() - train_loss.item()
    if epoch % 20 == 0:
        print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} gap={gap:.4f}")`}
      />

      <NoteBlock type="note" title="Beyond Loss Curves">
        <p>
          Also monitor task-specific metrics (accuracy, F1, BLEU) on validation data.
          Weight norms, gradient magnitudes, and activation distributions provide
          additional insight into model health during training.
        </p>
      </NoteBlock>
    </div>
  )
}
