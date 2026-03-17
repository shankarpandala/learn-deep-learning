import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SearchStrategyViz() {
  const [strategy, setStrategy] = useState('grid')
  const W = 300, H = 300

  const gridPoints = []
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    gridPoints.push({ x: 30 + i * 60, y: 30 + j * 60 })
  }

  const randomSeed = [0.12, 0.87, 0.34, 0.56, 0.91, 0.23, 0.67, 0.45, 0.78, 0.09,
    0.55, 0.38, 0.72, 0.15, 0.83, 0.41, 0.62, 0.29, 0.94, 0.51,
    0.17, 0.76, 0.44, 0.88, 0.33]
  const randomPoints = randomSeed.map((v, i) => ({
    x: 15 + v * (W - 30),
    y: 15 + randomSeed[(i + 7) % 25] * (H - 30)
  }))

  const bayesPoints = [
    { x: 150, y: 150 }, { x: 90, y: 200 }, { x: 200, y: 100 },
    { x: 170, y: 80 }, { x: 185, y: 65 }, { x: 195, y: 55 },
    { x: 200, y: 50 }, { x: 205, y: 48 }, { x: 202, y: 45 },
    { x: 198, y: 42 }, { x: 200, y: 40 }, { x: 201, y: 38 },
  ]

  const points = strategy === 'grid' ? gridPoints : strategy === 'random' ? randomPoints : bayesPoints

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Search Strategy Comparison</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        <select value={strategy} onChange={e => setStrategy(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
          <option value="grid">Grid Search</option>
          <option value="random">Random Search</option>
          <option value="bayesian">Bayesian Optimization</option>
        </select>
      </label>
      <svg width={W} height={H} className="mx-auto block bg-gray-50 dark:bg-gray-800 rounded">
        <text x={W / 2} y={H - 5} textAnchor="middle" fontSize={10} fill="#6b7280">Learning Rate</text>
        <text x={10} y={H / 2} textAnchor="middle" fontSize={10} fill="#6b7280" transform={`rotate(-90 10 ${H / 2})`}>Weight Decay</text>
        {points.map((p, i) => (
          <circle key={i} cx={p.x} cy={p.y} r={5} fill="#8b5cf6" opacity={strategy === 'bayesian' ? 0.3 + 0.7 * (i / points.length) : 0.7} />
        ))}
        {strategy === 'bayesian' && points.length > 1 && (
          <polyline fill="none" stroke="#8b5cf6" strokeWidth={1} strokeDasharray="3,3" opacity={0.4}
            points={points.map(p => `${p.x},${p.y}`).join(' ')} />
        )}
      </svg>
      <p className="text-xs text-center text-gray-500 mt-2">
        {strategy === 'grid' ? 'Evenly spaced grid — wastes budget on unimportant dimensions' :
         strategy === 'random' ? 'Random samples — better coverage of important dimensions' :
         'Bayesian — converges toward optimum guided by surrogate model'}
      </p>
    </div>
  )
}

export default function HyperparameterTuning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Hyperparameter tuning systematically searches for the best configuration of
        non-learnable parameters (learning rate, weight decay, dropout rate, architecture
        choices) to maximize validation performance.
      </p>

      <DefinitionBlock title="Grid Search">
        <p>
          Evaluate all combinations of discrete hyperparameter values. For <InlineMath math="d" /> hyperparameters
          with <InlineMath math="n" /> values each, this requires <InlineMath math="n^d" /> trials — exponential in dimensionality.
        </p>
      </DefinitionBlock>

      <SearchStrategyViz />

      <TheoremBlock title="Random Search Superiority" id="random-search">
        <p>
          Bergstra & Bengio (2012) showed that for hyperparameter spaces where only a few
          dimensions matter, random search finds good configurations in fewer trials:
        </p>
        <BlockMath math="P(\text{miss all good values}) = \left(1 - \frac{g}{G}\right)^T" />
        <p className="mt-2">
          where <InlineMath math="g/G" /> is the fraction of the space with good values and
          <InlineMath math="T" /> is the number of trials. Random search explores each dimension
          more densely than grid search for the same budget.
        </p>
      </TheoremBlock>

      <DefinitionBlock title="Bayesian Optimization">
        <p>
          Fit a surrogate model (typically a Gaussian Process) to the observed
          (hyperparameter, performance) pairs, then use an acquisition function
          to choose the next configuration to evaluate:
        </p>
        <BlockMath math="\mathbf{x}_{\text{next}} = \arg\max_{\mathbf{x}} \alpha(\mathbf{x} \mid \mathcal{D}_{1:t})" />
        <p className="mt-2">Common acquisition functions: Expected Improvement (EI), Upper Confidence Bound (UCB).</p>
      </DefinitionBlock>

      <ExampleBlock title="Practical Hyperparameter Ranges">
        <ul className="list-disc ml-4 space-y-1">
          <li><strong>Learning rate</strong>: log-uniform in <InlineMath math="[10^{-5}, 10^{-1}]" /></li>
          <li><strong>Weight decay</strong>: log-uniform in <InlineMath math="[10^{-6}, 10^{-1}]" /></li>
          <li><strong>Dropout</strong>: uniform in <InlineMath math="[0.0, 0.5]" /></li>
          <li><strong>Batch size</strong>: powers of 2 from 16 to 512</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Hyperparameter Search with Optuna"
        code={`import torch
import torch.nn as nn
# pip install optuna
import optuna

def objective(trial):
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    wd = trial.suggest_float('weight_decay', 1e-6, 1e-1, log=True)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    hidden = trial.suggest_categorical('hidden_dim', [64, 128, 256])

    model = nn.Sequential(
        nn.Linear(20, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    x_train, y_train = torch.randn(400, 20), torch.randn(400, 1)
    x_val, y_val = torch.randn(100, 20), torch.randn(100, 1)

    for _ in range(50):
        loss = nn.MSELoss()(model(x_train), y_train)
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        val_loss = nn.MSELoss()(model(x_val), y_val).item()
    return val_loss

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)
print(f"Best params: {study.best_params}")
print(f"Best val loss: {study.best_value:.4f}")`}
      />

      <NoteBlock type="note" title="Successive Halving & Hyperband">
        <p>
          For expensive deep learning runs, early stopping-based methods like Hyperband
          allocate more budget to promising configurations. Start many trials with small
          budgets, then progressively increase the budget for the best performers. This
          is 10-50x more efficient than standard random search.
        </p>
      </NoteBlock>
    </div>
  )
}
