import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ScalingLawViz() {
  const [dataScale, setDataScale] = useState(3)
  const W = 380, H = 160

  const points = Array.from({ length: 8 }, (_, i) => {
    const logData = i + 1
    const loss = 2.5 * Math.pow(logData, -0.4) + 0.3
    return { x: logData, y: loss }
  })
  const xMax = 9, yMin = 0, yMax = 3
  const toSVG = (x, y) => `${(x / xMax) * W},${H - ((y - yMin) / (yMax - yMin)) * H}`
  const curvePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${toSVG(p.x, p.y)}`).join(' ')
  const currentLoss = 2.5 * Math.pow(dataScale, -0.4) + 0.3

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Foundation Model Scaling</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Training data (log scale): {dataScale}
        <input type="range" min={1} max={8} step={0.5} value={dataScale} onChange={e => setDataScale(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={0} y1={H} x2={W} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={0} y1={0} x2={0} y2={H} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={curvePath} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <circle cx={(dataScale / xMax) * W} cy={H - ((currentLoss - yMin) / (yMax - yMin)) * H} r={5} fill="#f97316" />
        <text x={(dataScale / xMax) * W + 8} y={H - ((currentLoss - yMin) / (yMax - yMin)) * H + 4} className="text-[10px] fill-orange-500">
          loss = {currentLoss.toFixed(2)}
        </text>
        <text x={W / 2} y={H - 4} textAnchor="middle" className="text-[9px] fill-gray-400">log(training data)</text>
        <text x={4} y={12} className="text-[9px] fill-gray-400">loss</text>
      </svg>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        Zero-shot forecasting loss follows power-law scaling with pre-training data volume
      </p>
    </div>
  )
}

export default function FoundationModelsTimeSeries() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Foundation models for time series aim to create pre-trained models that generalize
        across domains without task-specific training. Models like Chronos, TimeGPT, and
        Moirai demonstrate that LLM-style pre-training can transfer to temporal data.
      </p>

      <DefinitionBlock title="Chronos: Tokenized Time Series Language Model">
        <p>Chronos maps real-valued time series into discrete tokens via quantization:</p>
        <BlockMath math="x_t \xrightarrow{\text{scale}} \tilde{x}_t \xrightarrow{\text{bin}} b_t \in \{1, \ldots, B\}" />
        <p className="mt-2">A T5-based language model is trained autoregressively on these tokens, then generates probabilistic forecasts by sampling token sequences and de-quantizing.</p>
      </DefinitionBlock>

      <ScalingLawViz />

      <TheoremBlock title="Zero-Shot Forecasting" id="zero-shot-ts">
        <p>A foundation model <InlineMath math="f_\theta" /> pre-trained on corpus <InlineMath math="\mathcal{D}_{\text{pre}}" /> can forecast unseen series <InlineMath math="\mathbf{x}_{\text{new}}" />:</p>
        <BlockMath math="\hat{\mathbf{y}} = f_\theta(\mathbf{x}_{\text{new}}) \quad \text{without any gradient updates}" />
        <p>The quality depends on the diversity of <InlineMath math="\mathcal{D}_{\text{pre}}" /> and similarity to the target domain.</p>
      </TheoremBlock>

      <ExampleBlock title="Moirai: Any-Variable, Any-Frequency">
        <p>
          Moirai uses a mixture of parametric distributions and handles varying numbers of
          variates via a masked attention mechanism. It supports arbitrary prediction lengths
          and frequencies, making it the most flexible foundation model for time series to date.
        </p>
      </ExampleBlock>

      <WarningBlock title="Limitations of TS Foundation Models">
        <p>
          Current foundation models can underperform domain-specific models on specialized
          datasets (e.g., medical or financial data with unique patterns). They also struggle
          with very long contexts and extreme distribution shifts. Always benchmark against
          a fine-tuned baseline before deploying zero-shot.
        </p>
      </WarningBlock>

      <PythonCode
        title="Using Chronos for Zero-Shot Forecasting"
        code={`import torch
import numpy as np
# pip install chronos-forecasting
# from chronos import ChronosPipeline

# Example usage (requires GPU + model download):
# pipeline = ChronosPipeline.from_pretrained(
#     "amazon/chronos-t5-small",
#     device_map="auto",
#     torch_dtype=torch.float32,
# )

# Simulated example of the Chronos workflow
context = torch.tensor(np.sin(np.linspace(0, 8*np.pi, 96)))

# Chronos quantization concept (simplified)
n_bins = 4096
scaled = (context - context.mean()) / (context.std() + 1e-8)
bins = torch.linspace(-3, 3, n_bins)
tokens = torch.bucketize(scaled, bins)
print(f"Context length: {len(context)}, Token range: [{tokens.min()}, {tokens.max()}]")

# In practice: pipeline.predict(context, prediction_length=24, num_samples=20)
# Returns (20, 24) samples for probabilistic forecasting
print("Zero-shot forecast: sample multiple trajectories -> quantile intervals")`}
      />

      <NoteBlock type="note" title="The Debate: Are TS Foundation Models Needed?">
        <p>
          Unlike NLP where text has universal grammar, time series are highly domain-specific.
          Recent work shows simple baselines (linear models) can be competitive. Foundation
          models shine when labeled data is scarce, diverse series must be handled, or rapid
          prototyping is needed. The field is actively evolving.
        </p>
      </NoteBlock>
    </div>
  )
}
