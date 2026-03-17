import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import ExerciseBlock from '../../../components/content/ExerciseBlock.jsx'

function DTWPathViz() {
  const [warpFactor, setWarpFactor] = useState(1.5)
  const N = 12
  const seriesA = Array.from({ length: N }, (_, i) => Math.sin(i * 0.5) * 2)
  const seriesB = Array.from({ length: N }, (_, i) => Math.sin((i * 0.5) / warpFactor) * 2)

  const W = 380, H = 140, topH = 50, botY = 90
  const xStep = W / (N + 1)
  const toTop = (i, v) => ({ x: (i + 1) * xStep, y: topH / 2 - v * 10 + 10 })
  const toBot = (i, v) => ({ x: (i + 1) * xStep, y: botY + topH / 2 - v * 10 })

  const warpPairs = seriesA.map((_, i) => {
    const j = Math.min(N - 1, Math.round(i / warpFactor))
    return { i, j }
  })

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Dynamic Time Warping Alignment</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Time warp: {warpFactor.toFixed(1)}x
        <input type="range" min={0.6} max={2.5} step={0.1} value={warpFactor} onChange={e => setWarpFactor(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        {warpPairs.map((p, idx) => {
          const a = toTop(p.i, seriesA[p.i])
          const b = toBot(p.j, seriesB[p.j])
          return <line key={idx} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="#8b5cf6" strokeWidth={0.8} opacity={0.3} />
        })}
        {seriesA.map((v, i) => {
          const p = toTop(i, v)
          return <circle key={`a-${i}`} cx={p.x} cy={p.y} r={3} fill="#8b5cf6" />
        })}
        {seriesB.map((v, i) => {
          const p = toBot(i, v)
          return <circle key={`b-${i}`} cx={p.x} cy={p.y} r={3} fill="#f97316" />
        })}
        <text x={8} y={topH / 2 + 10} className="text-[9px] fill-violet-500">A</text>
        <text x={8} y={botY + topH / 2} className="text-[9px] fill-orange-500">B</text>
      </svg>
      <p className="mt-1 text-center text-xs text-gray-500 dark:text-gray-400">
        DTW finds the optimal alignment between time-warped versions of the same pattern
      </p>
    </div>
  )
}

export default function DTWShapeletFeatures() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Dynamic Time Warping (DTW) and shapelets are classical time series methods that
        deep learning has begun to incorporate. DTW provides a time-invariant distance
        measure, while shapelets identify discriminative local patterns.
      </p>

      <DefinitionBlock title="Dynamic Time Warping (DTW)">
        <p>DTW finds the minimum-cost alignment between two sequences <InlineMath math="\mathbf{a}" /> and <InlineMath math="\mathbf{b}" /> via dynamic programming:</p>
        <BlockMath math="D(i, j) = d(a_i, b_j) + \min\{D(i-1, j),\; D(i, j-1),\; D(i-1, j-1)\}" />
        <p className="mt-2">The DTW distance is <InlineMath math="D(M, N)" />, allowing one-to-many point matchings to handle temporal distortions.</p>
      </DefinitionBlock>

      <DTWPathViz />

      <TheoremBlock title="Soft-DTW: Differentiable DTW" id="soft-dtw">
        <p>Soft-DTW replaces the hard <InlineMath math="\min" /> with a smooth minimum for gradient-based learning:</p>
        <BlockMath math="\text{min}^{\gamma}(a_1, \ldots, a_n) = -\gamma \log \sum_i e^{-a_i/\gamma}" />
        <BlockMath math="D^\gamma(i, j) = d(a_i, b_j) + \text{min}^{\gamma}\{D^\gamma(i-1,j),\; D^\gamma(i,j-1),\; D^\gamma(i-1,j-1)\}" />
        <p>As <InlineMath math="\gamma \to 0" />, soft-DTW recovers exact DTW.</p>
      </TheoremBlock>

      <DefinitionBlock title="Shapelets">
        <p>A shapelet <InlineMath math="\mathbf{s} \in \mathbb{R}^l" /> (<InlineMath math="l \ll T" />) is a subsequence pattern that is maximally discriminative between classes:</p>
        <BlockMath math="d_{\text{shapelet}}(\mathbf{x}, \mathbf{s}) = \min_{t} \|\mathbf{x}_{t:t+l} - \mathbf{s}\|_2" />
        <p className="mt-2">Learned shapelets are initialized randomly and optimized end-to-end via gradient descent.</p>
      </DefinitionBlock>

      <ExampleBlock title="Learned Shapelets as Soft Convolutions">
        <p>
          A learned shapelet layer computes soft minimum distances using a smooth approximation.
          This is equivalent to a special type of convolutional layer where the kernel
          represents a prototype pattern and the output measures similarity rather than
          linear correlation.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Soft-DTW & Learned Shapelets"
        code={`import torch
import torch.nn as nn

def soft_dtw(x, y, gamma=0.1):
    """Differentiable DTW distance (simplified)."""
    M, N = x.size(0), y.size(0)
    D = torch.full((M + 1, N + 1), float('inf'))
    D[0, 0] = 0
    for i in range(1, M + 1):
        for j in range(1, N + 1):
            cost = (x[i-1] - y[j-1])**2
            neighbors = torch.stack([D[i-1,j], D[i,j-1], D[i-1,j-1]])
            D[i, j] = cost + (-gamma * torch.logsumexp(-neighbors / gamma, dim=0))
    return D[M, N]

class LearnedShapelets(nn.Module):
    def __init__(self, n_shapelets=10, shapelet_len=15, n_classes=5):
        super().__init__()
        self.shapelets = nn.Parameter(torch.randn(n_shapelets, shapelet_len))
        self.fc = nn.Linear(n_shapelets, n_classes)

    def forward(self, x):  # x: (B, T)
        B, T = x.shape
        L = self.shapelets.size(1)
        dists = []
        for s in self.shapelets:
            # Sliding distance to each shapelet
            d = torch.stack([((x[:, t:t+L] - s)**2).sum(-1) for t in range(T - L + 1)], dim=1)
            dists.append(d.min(dim=1).values)  # min distance
        features = torch.stack(dists, dim=1)  # (B, n_shapelets)
        return self.fc(features)

model = LearnedShapelets()
x = torch.randn(8, 128)
print(f"Predictions: {model(x).shape}")  # (8, 5)`}
      />

      <ExerciseBlock title="Exercise: DTW Complexity">
        <p>
          Standard DTW has <InlineMath math="O(MN)" /> complexity. For a dataset of <InlineMath math="n" /> training series
          and 1-NN classification, the total cost is <InlineMath math="O(n \cdot M \cdot N)" /> per test query.
          How does the Sakoe-Chiba band constraint with width <InlineMath math="w" /> reduce this?
          What is the new complexity?
        </p>
      </ExerciseBlock>
    </div>
  )
}
