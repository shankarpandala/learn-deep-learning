import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function KFoldViz() {
  const [k, setK] = useState(5)
  const [activeFold, setActiveFold] = useState(0)
  const W = 400, H = 140

  const foldW = (W - 40) / k

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">K-Fold Cross-Validation</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          K = {k}
          <input type="range" min={2} max={10} step={1} value={k} onChange={e => { setK(parseInt(e.target.value)); setActiveFold(0) }} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Fold: {activeFold + 1}
          <input type="range" min={0} max={k - 1} step={1} value={activeFold} onChange={e => setActiveFold(parseInt(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: k }, (_, i) => {
          const x = 20 + i * foldW
          const isVal = i === activeFold
          return (
            <g key={i}>
              <rect x={x + 1} y={20} width={foldW - 2} height={60} rx={4}
                fill={isVal ? '#f97316' : '#8b5cf6'} opacity={0.7} />
              <text x={x + foldW / 2} y={55} textAnchor="middle" fontSize={11} fill="white" fontWeight="bold">
                {isVal ? 'Val' : 'Train'}
              </text>
              <text x={x + foldW / 2} y={100} textAnchor="middle" fontSize={9} fill="#6b7280">
                Fold {i + 1}
              </text>
            </g>
          )
        })}
        <text x={W / 2} y={125} textAnchor="middle" fontSize={10} fill="#6b7280">
          Train: {((k - 1) / k * 100).toFixed(0)}% | Val: {(1 / k * 100).toFixed(0)}%
        </text>
      </svg>
    </div>
  )
}

export default function CrossValidation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Cross-validation provides a more robust estimate of model performance by using
        every data point for both training and validation. However, it poses unique
        challenges for deep learning due to high computational cost.
      </p>

      <DefinitionBlock title="K-Fold Cross-Validation">
        <p>
          Partition data into <InlineMath math="K" /> equal folds. For each fold <InlineMath math="k" />,
          train on <InlineMath math="K-1" /> folds and validate on fold <InlineMath math="k" />. The final
          performance estimate is the average:
        </p>
        <BlockMath math="\hat{\mathcal{L}} = \frac{1}{K} \sum_{k=1}^K \mathcal{L}_k" />
      </DefinitionBlock>

      <KFoldViz />

      <TheoremBlock title="Variance of CV Estimate" id="cv-variance">
        <p>The variance of the K-fold CV estimator is approximately:</p>
        <BlockMath math="\text{Var}(\hat{\mathcal{L}}) \approx \frac{\sigma^2}{K} + \frac{K-1}{K}\rho\sigma^2" />
        <p className="mt-2">
          where <InlineMath math="\rho" /> is the correlation between fold estimates and <InlineMath math="\sigma^2" />
          is the per-fold variance. Larger <InlineMath math="K" /> increases the correlation term, so
          more folds is not always better.
        </p>
      </TheoremBlock>

      <WarningBlock title="Challenges for Deep Learning">
        <p>
          K-fold CV requires training <InlineMath math="K" /> separate models, each for the full
          training schedule. For large models (GPT, ViT), this is computationally prohibitive.
          Alternatives include: single train/val split, bootstrap estimation, or training once
          and evaluating with multiple random seeds.
        </p>
      </WarningBlock>

      <ExampleBlock title="When to Use CV for Deep Learning">
        <ul className="list-disc ml-4 space-y-1">
          <li><strong>Small datasets</strong> (medical imaging, specialized NLP): CV is essential</li>
          <li><strong>Hyperparameter selection</strong>: use CV to select, then retrain on full data</li>
          <li><strong>Large-scale pretraining</strong>: single split is standard practice</li>
          <li><strong>Competition settings</strong>: stratified K-fold is common (K=5 or K=10)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="K-Fold Cross-Validation for Deep Learning"
        code={`import torch
import torch.nn as nn
from torch.utils.data import Subset, DataLoader, TensorDataset
import numpy as np

def kfold_cv(dataset, k=5, epochs=50):
    n = len(dataset)
    indices = np.random.permutation(n)
    fold_size = n // k
    scores = []

    for fold in range(k):
        val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])

        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=64)

        model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(epochs):
            for xb, yb in train_loader:
                loss = nn.MSELoss()(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        val_loss = np.mean([nn.MSELoss()(model(xb), yb).item() for xb, yb in val_loader])
        scores.append(val_loss)
        print(f"Fold {fold+1}: val_loss = {val_loss:.4f}")

    print(f"Mean: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# Example usage
X = torch.randn(500, 20); y = torch.randn(500, 1)
dataset = TensorDataset(X, y)
kfold_cv(dataset, k=5, epochs=30)`}
      />

      <NoteBlock type="note" title="Stratified K-Fold">
        <p>
          For classification, always use stratified K-fold to preserve the class distribution
          in each fold. This is especially important with imbalanced datasets where random
          splits may leave some classes underrepresented in certain folds.
        </p>
      </NoteBlock>
    </div>
  )
}
