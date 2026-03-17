import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function WindowingVisualizer() {
  const [windowSize, setWindowSize] = useState(4)
  const [horizon, setHorizon] = useState(2)
  const data = [2.1, 3.5, 1.8, 4.2, 3.9, 5.1, 2.7, 4.8, 3.3, 5.5, 4.1, 6.0]
  const cellW = 34, cellH = 28, gap = 2

  const windows = []
  for (let i = 0; i <= data.length - windowSize - horizon; i++) {
    windows.push({ start: i, inputEnd: i + windowSize, targetEnd: i + windowSize + horizon })
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Sliding Window Visualization</h3>
      <div className="flex flex-wrap gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Lookback: {windowSize}
          <input type="range" min={2} max={6} step={1} value={windowSize} onChange={e => setWindowSize(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Horizon: {horizon}
          <input type="range" min={1} max={4} step={1} value={horizon} onChange={e => setHorizon(parseInt(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="overflow-x-auto">
        <svg width={data.length * (cellW + gap) + 10} height={windows.length * (cellH + gap) + cellH + 10}>
          {data.map((v, i) => (
            <text key={`h-${i}`} x={i * (cellW + gap) + cellW / 2} y={14} textAnchor="middle" className="text-[10px] fill-gray-500">{v}</text>
          ))}
          {windows.map((w, wi) => (
            <g key={wi} transform={`translate(0, ${wi * (cellH + gap) + 22})`}>
              {data.map((v, i) => {
                const isInput = i >= w.start && i < w.inputEnd
                const isTarget = i >= w.inputEnd && i < w.targetEnd
                return (
                  <rect key={i} x={i * (cellW + gap)} y={0} width={cellW} height={cellH} rx={3}
                    fill={isInput ? '#8b5cf6' : isTarget ? '#f97316' : '#f3f4f6'}
                    opacity={isInput || isTarget ? 0.85 : 0.25} />
                )
              })}
            </g>
          ))}
        </svg>
      </div>
      <div className="mt-2 flex gap-4 text-xs text-gray-500 dark:text-gray-400">
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-violet-500" /> Input window</span>
        <span className="flex items-center gap-1"><span className="inline-block w-3 h-3 rounded bg-orange-500" /> Forecast horizon</span>
        <span className="ml-auto">{windows.length} samples generated</span>
      </div>
    </div>
  )
}

export default function WindowingFeatureEngineering() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Transforming a raw time series into supervised learning samples requires careful
        windowing. The choice of lookback length and forecast horizon directly affects
        what the model can learn and predict.
      </p>

      <DefinitionBlock title="Sliding Window Transformation">
        <p>Given a time series <InlineMath math="\{x_1, \ldots, x_T\}" />, lookback <InlineMath math="L" />, and horizon <InlineMath math="H" />:</p>
        <BlockMath math="\text{Input: } \mathbf{x}_{t-L:t} = [x_{t-L}, \ldots, x_{t-1}] \;\;\to\;\; \text{Target: } \mathbf{y}_{t:t+H} = [x_t, \ldots, x_{t+H-1}]" />
        <p className="mt-2">This produces <InlineMath math="T - L - H + 1" /> training samples from a series of length <InlineMath math="T" />.</p>
      </DefinitionBlock>

      <WindowingVisualizer />

      <ExampleBlock title="Lag Features">
        <p>Lag features augment each timestep with previous values as additional input dimensions:</p>
        <BlockMath math="\mathbf{f}_t = [x_t, x_{t-1}, x_{t-7}, x_{t-14}, \ldots]" />
        <p>Calendar features (day-of-week, month, holiday flags) provide exogenous context for seasonal patterns.</p>
      </ExampleBlock>

      <WarningBlock title="Data Leakage in Time Series">
        <p>
          Never shuffle time series data randomly for train/test splits. Always use a
          temporal cutoff: train on <InlineMath math="[1, T_{\text{train}}]" />, validate on
          <InlineMath math="(T_{\text{train}}, T_{\text{val}}]" />, test on <InlineMath math="(T_{\text{val}}, T]" />.
          Normalization statistics must be computed <strong>only from the training set</strong>.
        </p>
      </WarningBlock>

      <PythonCode
        title="Creating Sliding Windows in PyTorch"
        code={`import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, series, lookback=24, horizon=6):
        self.X, self.Y = [], []
        for i in range(len(series) - lookback - horizon + 1):
            self.X.append(series[i:i+lookback])
            self.Y.append(series[i+lookback:i+lookback+horizon])
        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx]

# Example: 1000-step series
series = torch.sin(torch.linspace(0, 20*3.14159, 1000)) + torch.randn(1000)*0.1
ds = TimeSeriesDataset(series, lookback=24, horizon=6)
loader = DataLoader(ds, batch_size=32, shuffle=False)  # no shuffle!
print(f"Samples: {len(ds)}, batch X: {next(iter(loader))[0].shape}")`}
      />

      <NoteBlock type="note" title="Stride and Multi-Scale Windows">
        <p>
          Using a stride greater than 1 reduces overlapping samples, which can speed up
          training. Multi-scale windowing — combining short and long lookback periods —
          helps models capture both short-term dynamics and long-range dependencies simultaneously.
        </p>
      </NoteBlock>
    </div>
  )
}
