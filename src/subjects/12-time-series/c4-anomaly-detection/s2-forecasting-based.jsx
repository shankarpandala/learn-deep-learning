import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ForecastAnomalyViz() {
  const [sensitivity, setSensitivity] = useState(2.0)
  const N = 50, W = 400, H = 150

  const series = Array.from({ length: N }, (_, i) => {
    const base = Math.sin(i * 0.25) + 0.4 * Math.cos(i * 0.6)
    const anomaly = (i >= 28 && i <= 31) ? 2.2 : (i === 40 ? -1.8 : 0)
    return base + anomaly
  })
  const forecast = series.map((_, i) => Math.sin(i * 0.25) + 0.4 * Math.cos(i * 0.6))
  const residuals = series.map((v, i) => v - forecast[i])
  const resStd = Math.sqrt(residuals.reduce((s, r) => s + r * r, 0) / N)

  const yMin = -3, yMax = 4
  const toSVG = (i, v) => `${(i / N) * W},${H * 0.65 - ((v - yMin) / (yMax - yMin)) * H * 0.65}`
  const seriesPath = series.map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(i, v)}`).join(' ')
  const forecastPath = forecast.map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(i, v)}`).join(' ')

  const upperBand = forecast.map((v, i) => toSVG(i, v + sensitivity * resStd))
  const lowerBand = forecast.map((v, i) => toSVG(i, v - sensitivity * resStd)).reverse()
  const bandPath = `M${upperBand.join(' L')} L${lowerBand.join(' L')} Z`

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Forecast-Based Anomaly Detection</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Sensitivity (k-sigma): {sensitivity.toFixed(1)}
        <input type="range" min={1} max={4} step={0.2} value={sensitivity} onChange={e => setSensitivity(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <path d={bandPath} fill="#8b5cf6" opacity={0.1} />
        <path d={forecastPath} fill="none" stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="4,3" />
        <path d={seriesPath} fill="none" stroke="#374151" strokeWidth={1.5} />
        {series.map((v, i) => {
          const isAnomaly = Math.abs(residuals[i]) > sensitivity * resStd
          if (!isAnomaly) return null
          const [x, y] = toSVG(i, v).split(',')
          return <circle key={i} cx={parseFloat(x)} cy={parseFloat(y)} r={4} fill="#ef4444" />
        })}
      </svg>
      <div className="mt-2 flex justify-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-gray-700" /> Actual</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-0.5 bg-violet-500" /> Forecast</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded-full bg-red-500" /> Anomaly</span>
      </div>
    </div>
  )
}

export default function ForecastingBasedDetection() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Forecasting-based anomaly detection compares model predictions with actual observations.
        Large prediction errors signal anomalous behavior. This approach naturally handles
        non-stationary data since the model learns to track normal dynamics.
      </p>

      <DefinitionBlock title="Prediction Error Anomaly Score">
        <p>Given a trained forecaster <InlineMath math="\hat{x}_{t} = f_\theta(x_{t-L:t-1})" />, the anomaly score at time <InlineMath math="t" /> is:</p>
        <BlockMath math="a_t = \frac{|x_t - \hat{x}_t|}{\hat{\sigma}_t}" />
        <p className="mt-2">where <InlineMath math="\hat{\sigma}_t" /> is the estimated prediction uncertainty. Points with <InlineMath math="a_t > k" /> (e.g., <InlineMath math="k=3" />) are anomalies.</p>
      </DefinitionBlock>

      <ForecastAnomalyViz />

      <TheoremBlock title="Conformal Prediction Intervals" id="conformal-anomaly">
        <p>For calibrated anomaly detection, compute nonconformity scores on a calibration set:</p>
        <BlockMath math="s_i = |x_i - \hat{x}_i|, \quad i \in \mathcal{D}_{\text{cal}}" />
        <p>The prediction interval at level <InlineMath math="1-\alpha" /> uses the <InlineMath math="\lceil(1-\alpha)(n+1)\rceil/n" /> quantile of <InlineMath math="\{s_i\}" />:</p>
        <BlockMath math="C_t = [\hat{x}_t - q_{1-\alpha},\; \hat{x}_t + q_{1-\alpha}]" />
        <p>This guarantees <InlineMath math="P(x_t \in C_t) \geq 1-\alpha" /> without distributional assumptions.</p>
      </TheoremBlock>

      <ExampleBlock title="Multi-Step Forecast Residuals">
        <p>
          For multi-step forecasting, compute anomaly scores across the full horizon. An anomaly
          at step <InlineMath math="h" /> is weighted by the expected error at that horizon:
          <InlineMath math="a_{t+h} = |x_{t+h} - \hat{x}_{t+h}| / \hat{\sigma}_h" />, where
          <InlineMath math="\hat{\sigma}_h" /> grows with <InlineMath math="h" />.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Conformal Anomaly Detection"
        code={`import torch
import numpy as np

def conformal_anomaly_detector(model, cal_data, test_data, alpha=0.05):
    """Detect anomalies with guaranteed coverage via conformal prediction."""
    # Compute calibration nonconformity scores
    cal_preds = model(cal_data['X'])  # shape: (n_cal, 1)
    cal_scores = torch.abs(cal_data['y'] - cal_preds).squeeze().detach().numpy()

    # Conformal quantile
    n = len(cal_scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q_hat = np.quantile(cal_scores, min(q_level, 1.0))

    # Detect anomalies on test set
    test_preds = model(test_data['X'])
    test_scores = torch.abs(test_data['y'] - test_preds).squeeze().detach().numpy()
    anomalies = test_scores > q_hat

    return anomalies, q_hat

# Example with a simple model
n_cal, n_test = 200, 50
cal_scores = np.abs(np.random.randn(n_cal))  # normal residuals
q_hat = np.quantile(cal_scores, 0.95)
test_scores = np.concatenate([np.abs(np.random.randn(45)), np.abs(np.random.randn(5)) + 3])
anomalies = test_scores > q_hat
print(f"Threshold (95%): {q_hat:.3f}")
print(f"Anomalies detected: {anomalies.sum()}/{n_test}")`}
      />

      <NoteBlock type="note" title="Combining Forecast + Reconstruction">
        <p>
          Ensemble approaches combine forecasting and reconstruction anomaly scores for
          robustness. Forecasting detects point anomalies well (sudden spikes), while
          reconstruction catches contextual anomalies (subtle distribution shifts within
          normal value ranges).
        </p>
      </NoteBlock>
    </div>
  )
}
