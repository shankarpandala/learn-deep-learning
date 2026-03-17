import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TrainingDiagram() {
  const [epoch, setEpoch] = useState(1)
  const stages = [
    { name: 'Load Batch', color: '#8b5cf6' },
    { name: 'Forward Pass', color: '#7c3aed' },
    { name: 'Compute Loss', color: '#6d28d9' },
    { name: 'Backward Pass', color: '#5b21b6' },
    { name: 'Update Params', color: '#4c1d95' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Training Loop Pipeline</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Epoch: {epoch}
        <input type="range" min={1} max={10} step={1} value={epoch} onChange={e => setEpoch(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={460} height={80} className="mx-auto block">
        {stages.map((s, i) => {
          const x = i * 90 + 5
          return (
            <g key={i}>
              <rect x={x} y={20} width={82} height={36} rx={6} fill={s.color} opacity={0.85} />
              <text x={x + 41} y={42} textAnchor="middle" fontSize={10} fill="white" fontWeight="bold">{s.name}</text>
              {i < stages.length - 1 && (
                <text x={x + 85} y={42} textAnchor="middle" fontSize={16} fill="#a78bfa">&#8594;</text>
              )}
            </g>
          )
        })}
        <text x={230} y={75} textAnchor="middle" fontSize={11} fill="#7c3aed">Repeat for each batch in epoch {epoch}</text>
      </svg>
    </div>
  )
}

export default function TrainingLoop() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The training loop is the core algorithm for fitting a neural network. It iterates over
        the dataset in epochs, processing mini-batches, computing gradients, and updating parameters.
        A well-structured loop includes validation, logging, and checkpointing.
      </p>

      <DefinitionBlock title="Training Loop Components">
        <p><strong>Epoch:</strong> One complete pass through the training dataset.</p>
        <p><strong>Batch:</strong> A subset of <InlineMath math="B" /> samples processed together.</p>
        <p><strong>Iteration:</strong> One parameter update (one batch). Per epoch: <InlineMath math="N/B" /> iterations.</p>
        <p><strong>Validation:</strong> Evaluation on held-out data after each epoch to monitor generalization.</p>
      </DefinitionBlock>

      <TrainingDiagram />

      <ExampleBlock title="The Five Steps">
        <p>Each training iteration follows this exact sequence:</p>
        <p><strong>1.</strong> <code>optimizer.zero_grad()</code> — Clear old gradients</p>
        <p><strong>2.</strong> <code>output = model(x)</code> — Forward pass</p>
        <p><strong>3.</strong> <code>loss = criterion(output, y)</code> — Compute loss</p>
        <p><strong>4.</strong> <code>loss.backward()</code> — Backward pass (compute gradients)</p>
        <p><strong>5.</strong> <code>optimizer.step()</code> — Update parameters</p>
      </ExampleBlock>

      <WarningBlock title="Common Training Loop Bugs">
        <p>
          <strong>Forgetting zero_grad:</strong> Gradients accumulate across iterations, causing divergence.{' '}
          <strong>Forgetting model.eval():</strong> Dropout and batch norm behave differently during
          validation. <strong>Data leakage:</strong> Validating on training data gives misleading metrics.
        </p>
      </WarningBlock>

      <PythonCode
        title="Complete PyTorch Training Loop"
        code={`import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# Data
X = torch.randn(1000, 20)
y = (X[:, :5].sum(dim=1, keepdim=True) > 0).float()
dataset = TensorDataset(X, y)
train_ds, val_ds = random_split(dataset, [800, 200])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

# Model, loss, optimizer
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)

    # Validation
    model.eval()
    val_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += criterion(pred, yb).item() * len(xb)
            correct += ((pred > 0.5) == yb).sum().item()

    print(f"Epoch {epoch}: train_loss={train_loss/800:.4f}, "
          f"val_loss={val_loss/200:.4f}, val_acc={correct/200:.2%}")`}
      />

      <NoteBlock type="note" title="Production Training Best Practices">
        <p>
          Real training loops also include: <strong>learning rate scheduling</strong> (cosine decay,
          warmup), <strong>gradient clipping</strong> (prevent explosion), <strong>checkpointing</strong>{' '}
          (save best model by validation metric), <strong>early stopping</strong> (stop when validation
          loss plateaus), <strong>mixed precision</strong> (FP16 for speed), and{' '}
          <strong>distributed training</strong> (multi-GPU with DDP).
        </p>
      </NoteBlock>

      <ExampleBlock title="Learning Rate Scheduling">
        <p>A common pattern is warmup + cosine decay:</p>
        <BlockMath math="\eta_t = \eta_{\max} \cdot \frac{1}{2}\left(1 + \cos\left(\frac{\pi \cdot t}{T}\right)\right)" />
        <p>
          This starts at <InlineMath math="\eta_{\max}" /> and smoothly decays to zero over{' '}
          <InlineMath math="T" /> steps, helping the model converge to flatter minima.
        </p>
      </ExampleBlock>
    </div>
  )
}
