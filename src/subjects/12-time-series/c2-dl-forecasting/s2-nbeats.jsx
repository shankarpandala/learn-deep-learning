import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function StackDiagram() {
  const [numBlocks, setNumBlocks] = useState(3)
  const blockW = 80, blockH = 50, gap = 16, arrowLen = 30
  const totalW = numBlocks * (blockW + gap + arrowLen) + 80
  const totalH = 140

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">N-BEATS Doubly Residual Stack</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Blocks: {numBlocks}
        <input type="range" min={2} max={5} step={1} value={numBlocks} onChange={e => setNumBlocks(parseInt(e.target.value))} className="w-24 accent-violet-500" />
      </label>
      <div className="overflow-x-auto">
        <svg width={totalW} height={totalH} className="mx-auto block">
          <text x={4} y={totalH / 2 - 15} className="text-[10px] fill-gray-500">Input</text>
          <text x={4} y={totalH / 2 + 5} className="text-[10px] fill-violet-500">x</text>
          {Array.from({ length: numBlocks }, (_, i) => {
            const bx = 40 + i * (blockW + gap + arrowLen)
            const by = (totalH - blockH) / 2
            return (
              <g key={i}>
                <line x1={bx - arrowLen} y1={totalH / 2} x2={bx} y2={totalH / 2} stroke="#8b5cf6" strokeWidth={1.5} markerEnd="url(#arrow)" />
                <rect x={bx} y={by} width={blockW} height={blockH} rx={6} fill="#8b5cf6" opacity={0.15} stroke="#8b5cf6" strokeWidth={1.5} />
                <text x={bx + blockW / 2} y={by + 20} textAnchor="middle" className="text-[10px] fill-violet-700 dark:fill-violet-300 font-semibold">Block {i + 1}</text>
                <text x={bx + blockW / 2} y={by + 35} textAnchor="middle" className="text-[9px] fill-gray-500">{i === 0 ? 'Trend' : i === 1 ? 'Season' : 'Generic'}</text>
                <line x1={bx + blockW / 2} y1={by + blockH} x2={bx + blockW / 2} y2={by + blockH + 20} stroke="#f97316" strokeWidth={1} />
                <text x={bx + blockW / 2} y={by + blockH + 30} textAnchor="middle" className="text-[9px] fill-orange-500">f{i + 1}</text>
              </g>
            )
          })}
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={5} markerHeight={5} orient="auto-start-auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#8b5cf6" />
            </marker>
          </defs>
          <text x={totalW - 30} y={totalH - 10} className="text-[10px] fill-orange-500">y = sum(fi)</text>
        </svg>
      </div>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        Each block produces a backcast (residual for next block) and a partial forecast. Final forecast is the sum of all partial forecasts.
      </p>
    </div>
  )
}

export default function NBeatsNHits() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        N-BEATS (Neural Basis Expansion Analysis for Time Series) is a pure deep learning
        architecture that achieves state-of-the-art results without requiring time series-specific
        feature engineering. N-HiTS extends it with hierarchical interpolation for long horizons.
      </p>

      <DefinitionBlock title="N-BEATS Block">
        <p>Each block receives input <InlineMath math="\mathbf{x}_\ell" /> and produces:</p>
        <BlockMath math="\mathbf{h} = \text{FC}_4 \circ \text{FC}_3 \circ \text{FC}_2 \circ \text{FC}_1(\mathbf{x}_\ell)" />
        <BlockMath math="\hat{\mathbf{x}}_\ell = \mathbf{V}_b^\top \boldsymbol{\theta}_b(\mathbf{h}) \quad \text{(backcast)}, \qquad \hat{\mathbf{y}}_\ell = \mathbf{V}_f^\top \boldsymbol{\theta}_f(\mathbf{h}) \quad \text{(forecast)}" />
        <p className="mt-2">where <InlineMath math="\mathbf{V}_b, \mathbf{V}_f" /> are basis matrices — either learned or constrained (trend/seasonal).</p>
      </DefinitionBlock>

      <StackDiagram />

      <TheoremBlock title="Interpretable Basis Functions" id="nbeats-basis">
        <p>Trend basis uses polynomial coefficients up to degree <InlineMath math="p" />:</p>
        <BlockMath math="\mathbf{V}_{\text{trend}} = \begin{bmatrix} 1 & t & t^2 & \cdots & t^p \end{bmatrix}" />
        <p>Seasonality basis uses Fourier terms with period <InlineMath math="S" />:</p>
        <BlockMath math="\mathbf{V}_{\text{season}} = \begin{bmatrix} \cos(2\pi t/S) & \sin(2\pi t/S) & \cos(4\pi t/S) & \cdots \end{bmatrix}" />
      </TheoremBlock>

      <ExampleBlock title="N-HiTS: Hierarchical Interpolation">
        <p>
          N-HiTS adds multi-rate signal sampling — each block operates at a different temporal
          resolution. Lower blocks capture fine-grained patterns, higher blocks capture long-range
          structure. The forecast is assembled by interpolating outputs from all levels.
        </p>
      </ExampleBlock>

      <PythonCode
        title="N-BEATS / N-HiTS with NeuralForecast"
        code={`from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS
from neuralforecast.losses.pytorch import MAE
import pandas as pd
import numpy as np

# Prepare data in NeuralForecast format: unique_id, ds, y
dates = pd.date_range("2020-01-01", periods=365, freq="D")
series = []
for i in range(50):
    trend = np.linspace(0, 3, 365)
    season = 2 * np.sin(2 * np.pi * np.arange(365) / 7)
    y = trend + season + np.random.randn(365) * 0.5
    df = pd.DataFrame({"unique_id": f"s{i}", "ds": dates, "y": y})
    series.append(df)
data = pd.concat(series).reset_index(drop=True)

horizon = 14

# N-BEATS: interpretable stacks (trend + seasonality)
nbeats = NBEATS(
    h=horizon,
    input_size=2 * horizon,       # lookback = 2x horizon
    stack_types=["trend", "seasonality"],  # interpretable config
    n_blocks=[3, 3],
    mlp_units=[[256, 256]] * 2,
    loss=MAE(),
    max_steps=100,
)

# N-HiTS: hierarchical interpolation for long horizons
nhits = NHITS(
    h=horizon,
    input_size=2 * horizon,
    n_pool_kernel_size=[4, 2, 1],  # multi-rate downsampling
    loss=MAE(),
    max_steps=100,
)

# Train and forecast
nf = NeuralForecast(models=[nbeats, nhits], freq="D")
nf.fit(df=data)
forecasts = nf.predict()
print(forecasts.head())
# Columns: unique_id, ds, NBEATS, NHITS`}
      />

      <NoteBlock type="note" title="N-BEATS vs N-HiTS Trade-offs">
        <p>
          N-BEATS works best for short-to-medium horizons with its uniform architecture.
          N-HiTS excels at long horizons by allowing different blocks to focus on different
          temporal scales, reducing computation by up to 50x while matching or exceeding
          N-BEATS accuracy.
        </p>
      </NoteBlock>
    </div>
  )
}
