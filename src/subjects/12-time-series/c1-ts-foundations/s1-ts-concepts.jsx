import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function StationarityDemo() {
  const [trend, setTrend] = useState(0.5)
  const [seasonality, setSeason] = useState(1.0)
  const W = 420, H = 180, N = 120

  const points = Array.from({ length: N }, (_, i) => {
    const t = i / N
    const trendComp = trend * t * 3
    const seasonComp = seasonality * Math.sin(2 * Math.PI * t * 4)
    const noise = Math.sin(i * 7.3) * 0.3 + Math.cos(i * 13.1) * 0.2
    return trendComp + seasonComp + noise
  })
  const yMin = Math.min(...points) - 0.3
  const yMax = Math.max(...points) + 0.3
  const toSVG = (i, v) => `${(i / N) * W},${H - ((v - yMin) / (yMax - yMin)) * H}`
  const path = points.map((v, i) => `${i === 0 ? 'M' : 'L'}${toSVG(i, v)}`).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Stationarity Explorer</h3>
      <div className="flex flex-wrap gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Trend: {trend.toFixed(1)}
          <input type="range" min={0} max={2} step={0.1} value={trend} onChange={e => setTrend(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Seasonality: {seasonality.toFixed(1)}
          <input type="range" min={0} max={3} step={0.1} value={seasonality} onChange={e => setSeason(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2} />
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        {trend === 0 && seasonality === 0 ? '✓ Approximately stationary (noise only)' : 'Non-stationary — has trend and/or seasonality'}
      </p>
    </div>
  )
}

export default function TimeSeriesConcepts() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Time series data consists of observations recorded sequentially over time. Before applying
        deep learning, understanding fundamental properties like stationarity, trend, and
        seasonality is critical for proper modeling and evaluation.
      </p>

      <DefinitionBlock title="Stationarity">
        <p>A time series <InlineMath math="X_t" /> is <strong>strictly stationary</strong> if its joint distribution is invariant to time shifts. In practice we use <strong>weak stationarity</strong>:</p>
        <BlockMath math="\mathbb{E}[X_t] = \mu \quad \text{(constant)}, \qquad \text{Cov}(X_t, X_{t+h}) = \gamma(h) \quad \text{(depends only on lag } h\text{)}" />
      </DefinitionBlock>

      <DefinitionBlock title="Autocorrelation Function (ACF)">
        <BlockMath math="\rho(h) = \frac{\gamma(h)}{\gamma(0)} = \frac{\text{Cov}(X_t, X_{t+h})}{\text{Var}(X_t)}" />
        <p className="mt-2">The ACF reveals repeating patterns, trend persistence, and seasonal cycles in the data.</p>
      </DefinitionBlock>

      <StationarityDemo />

      <TheoremBlock title="Classical Decomposition" id="ts-decomposition">
        <p>Any time series can be decomposed into three components:</p>
        <BlockMath math="X_t = T_t + S_t + R_t" />
        <p>where <InlineMath math="T_t" /> is the trend, <InlineMath math="S_t" /> is the seasonal component, and <InlineMath math="R_t" /> is the residual. A multiplicative variant uses <InlineMath math="X_t = T_t \cdot S_t \cdot R_t" />.</p>
      </TheoremBlock>

      <ExampleBlock title="Differencing for Stationarity">
        <p>First-order differencing removes a linear trend:</p>
        <BlockMath math="\nabla X_t = X_t - X_{t-1}" />
        <p>Seasonal differencing at period <InlineMath math="m" />: <InlineMath math="\nabla_m X_t = X_t - X_{t-m}" /></p>
      </ExampleBlock>

      <PythonCode
        title="Stationarity Testing & Decomposition"
        code={`import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Generate sample series with trend + seasonality
t = np.arange(200)
series = 0.05 * t + 2 * np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 0.5

# Augmented Dickey-Fuller test for stationarity
result = adfuller(series)
print(f"ADF statistic: {result[0]:.4f}, p-value: {result[1]:.4f}")
print("Stationary" if result[1] < 0.05 else "Non-stationary")

# Decompose (period=12 for monthly seasonality)
decomp = seasonal_decompose(series, model='additive', period=12)
print(f"Trend range: [{decomp.trend[~np.isnan(decomp.trend)].min():.2f}, "
      f"{decomp.trend[~np.isnan(decomp.trend)].max():.2f}]")`}
      />

      <NoteBlock type="note" title="Why Stationarity Matters for DL">
        <p>
          While deep learning models can implicitly learn trends and seasonality, making a
          series stationary before training often improves convergence and generalization.
          Many state-of-the-art models like N-BEATS apply <strong>reversible instance
          normalization</strong> — a learned form of stationarization.
        </p>
      </NoteBlock>
    </div>
  )
}
