import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function MetricComparison() {
  const [outlierMag, setOutlierMag] = useState(0)
  const actual =   [3.0, 4.5, 2.8, 5.1, 3.7, 4.2, 3.9, 5.0]
  const predicted = [3.2, 4.1, 3.0, 4.8, 3.5, 4.5, 3.7, 5.2]
  const preds = predicted.map((p, i) => i === 3 ? p + outlierMag : p)

  const mae = actual.reduce((s, a, i) => s + Math.abs(a - preds[i]), 0) / actual.length
  const rmse = Math.sqrt(actual.reduce((s, a, i) => s + (a - preds[i]) ** 2, 0) / actual.length)
  const mape = (actual.reduce((s, a, i) => s + Math.abs((a - preds[i]) / a), 0) / actual.length) * 100

  const W = 360, H = 140, px = W / (actual.length + 1)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Metric Sensitivity to Outliers</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Outlier magnitude: {outlierMag.toFixed(1)}
        <input type="range" min={0} max={5} step={0.2} value={outlierMag} onChange={e => setOutlierMag(parseFloat(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {actual.map((a, i) => {
          const x = (i + 1) * px, ya = H - ((a - 1) / 6) * H, yp = H - ((preds[i] - 1) / 6) * H
          return (
            <g key={i}>
              <line x1={x} y1={ya} x2={x} y2={yp} stroke="#e5e7eb" strokeWidth={1} strokeDasharray="2,2" />
              <circle cx={x} cy={ya} r={4} fill="#8b5cf6" />
              <circle cx={x} cy={yp} r={4} fill="#f97316" />
            </g>
          )
        })}
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-violet-500" /> Actual</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-orange-500" /> Predicted</span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3 text-center text-sm">
        <div className="rounded-lg bg-violet-50 p-2 dark:bg-violet-900/20">
          <div className="font-bold text-violet-700 dark:text-violet-300">MAE</div>
          <div className="text-gray-700 dark:text-gray-300">{mae.toFixed(3)}</div>
        </div>
        <div className="rounded-lg bg-violet-50 p-2 dark:bg-violet-900/20">
          <div className="font-bold text-violet-700 dark:text-violet-300">RMSE</div>
          <div className="text-gray-700 dark:text-gray-300">{rmse.toFixed(3)}</div>
        </div>
        <div className="rounded-lg bg-violet-50 p-2 dark:bg-violet-900/20">
          <div className="font-bold text-violet-700 dark:text-violet-300">MAPE</div>
          <div className="text-gray-700 dark:text-gray-300">{mape.toFixed(1)}%</div>
        </div>
      </div>
    </div>
  )
}

export default function ForecastingEvaluation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Evaluating forecasting models requires metrics that capture different error
        properties and evaluation protocols that prevent data leakage across time.
      </p>

      <DefinitionBlock title="Core Forecasting Metrics">
        <BlockMath math="\text{MAE} = \frac{1}{H}\sum_{h=1}^{H}|y_{t+h} - \hat{y}_{t+h}|" />
        <BlockMath math="\text{RMSE} = \sqrt{\frac{1}{H}\sum_{h=1}^{H}(y_{t+h} - \hat{y}_{t+h})^2}" />
        <BlockMath math="\text{MAPE} = \frac{100\%}{H}\sum_{h=1}^{H}\left|\frac{y_{t+h} - \hat{y}_{t+h}}{y_{t+h}}\right|" />
      </DefinitionBlock>

      <MetricComparison />

      <TheoremBlock title="MAE vs RMSE Sensitivity" id="mae-rmse-comparison">
        <p>RMSE penalizes large errors disproportionately due to squaring:</p>
        <BlockMath math="\text{MAE} \leq \text{RMSE} \leq \sqrt{H} \cdot \text{MAE}" />
        <p>When <InlineMath math="\text{RMSE} \gg \text{MAE}" />, it signals the presence of occasional large errors (outliers).</p>
      </TheoremBlock>

      <ExampleBlock title="Rolling Window Evaluation">
        <p>
          Instead of a single train-test split, use expanding or sliding window cross-validation:
          train on <InlineMath math="[1, t]" />, predict <InlineMath math="[t+1, t+H]" />, then slide
          forward by a step size <InlineMath math="s" />. Average metrics across all folds for a robust estimate.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Forecasting Metrics & Walk-Forward Evaluation"
        code={`import numpy as np

def mae(y, yhat): return np.mean(np.abs(y - yhat))
def rmse(y, yhat): return np.sqrt(np.mean((y - yhat)**2))
def mape(y, yhat): return 100 * np.mean(np.abs((y - yhat) / y))

# Walk-forward evaluation
def walk_forward_eval(series, model_fn, lookback=24, horizon=6, step=6):
    scores = []
    for t in range(lookback, len(series) - horizon, step):
        X = series[t-lookback:t]
        y_true = series[t:t+horizon]
        y_pred = model_fn(X)  # your model's forecast
        scores.append(mae(y_true, y_pred))
    return np.mean(scores), np.std(scores)

# Example with naive persistence baseline
series = np.sin(np.linspace(0, 20*np.pi, 500)) + np.random.randn(500)*0.1
naive_fn = lambda x: np.full(6, x[-1])  # repeat last value
mean_mae, std_mae = walk_forward_eval(series, naive_fn)
print(f"Naive baseline MAE: {mean_mae:.4f} +/- {std_mae:.4f}")`}
      />

      <NoteBlock type="note" title="Scaled Metrics for Cross-Series Comparison">
        <p>
          When comparing across series with different scales, use <strong>MASE</strong> (Mean
          Absolute Scaled Error), which normalizes by the in-sample naive forecast error.
          This makes MASE scale-independent: values below 1.0 indicate the model beats the
          naive baseline.
        </p>
      </NoteBlock>
    </div>
  )
}
