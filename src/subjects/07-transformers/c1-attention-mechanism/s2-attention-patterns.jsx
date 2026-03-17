import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function AttentionHeatmap() {
  const [temp, setTemp] = useState(1.0)
  const tokens = ['The', 'cat', 'sat', 'on']
  const rawScores = [
    [1.0, 0.3, 0.1, 0.2],
    [0.2, 1.0, 0.5, 0.1],
    [0.1, 0.6, 1.0, 0.8],
    [0.3, 0.1, 0.7, 1.0],
  ]

  function softmaxRow(row, t) {
    const scaled = row.map(s => s / t)
    const mx = Math.max(...scaled)
    const exps = scaled.map(s => Math.exp(s - mx))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(e => e / sum)
  }

  const weights = rawScores.map(row => softmaxRow(row, temp))
  const cellSize = 52

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Attention Heatmap</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Temperature: {temp.toFixed(2)}
        <input type="range" min={0.1} max={3.0} step={0.05} value={temp} onChange={e => setTemp(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="overflow-x-auto flex justify-center">
        <table className="border-collapse">
          <thead>
            <tr>
              <td />
              {tokens.map(t => <th key={t} className="text-xs text-gray-500 dark:text-gray-400 px-1 pb-1 font-mono">{t}</th>)}
            </tr>
          </thead>
          <tbody>
            {tokens.map((rowTok, i) => (
              <tr key={rowTok}>
                <td className="text-xs text-gray-500 dark:text-gray-400 pr-2 font-mono">{rowTok}</td>
                {weights[i].map((w, j) => (
                  <td key={j} style={{ width: cellSize, height: cellSize, backgroundColor: `rgba(139, 92, 246, ${w})` }} className="text-center text-xs font-mono text-gray-800 dark:text-gray-100 border border-gray-200 dark:border-gray-700">
                    {w.toFixed(2)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function AttentionPatterns() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Attention score distributions determine how the model allocates focus across input positions.
        Understanding how temperature and score magnitude shape these distributions is critical for
        diagnosing and tuning transformer behavior.
      </p>

      <DefinitionBlock title="Softmax Temperature">
        <BlockMath math="\text{softmax}(z_i / \tau) = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}}" />
        <p className="mt-2">
          As <InlineMath math="\tau \to 0" />, the distribution approaches a one-hot vector (hard attention).
          As <InlineMath math="\tau \to \infty" />, it approaches a uniform distribution.
        </p>
      </DefinitionBlock>

      <AttentionHeatmap />

      <ExampleBlock title="Entropy of Attention Weights">
        <p>The entropy of the attention distribution measures how spread the focus is:</p>
        <BlockMath math="H(\alpha) = -\sum_i \alpha_i \log \alpha_i" />
        <p>
          Uniform over <InlineMath math="n" /> tokens gives <InlineMath math="H = \log n" /> (maximum entropy).
          A peaked distribution gives <InlineMath math="H \approx 0" />.
        </p>
      </ExampleBlock>

      <WarningBlock title="Attention Collapse">
        <p>
          When attention weights become too peaked (low entropy), the model may ignore
          useful context — a phenomenon called <em>attention collapse</em>. This can happen
          when the model overfits or when temperature scaling is improperly tuned.
        </p>
      </WarningBlock>

      <PythonCode
        title="Visualizing Attention Score Distributions"
        code={`import torch
import torch.nn.functional as F

scores = torch.tensor([1.0, 2.5, 0.8, 3.1])

# Effect of temperature on attention distribution
for temp in [0.5, 1.0, 2.0, 5.0]:
    weights = F.softmax(scores / temp, dim=-1)
    entropy = -(weights * weights.log()).sum().item()
    print(f"T={temp:.1f}: weights={weights.numpy().round(3)}, H={entropy:.3f}")

# Output shows sharper distributions at low temp,
# more uniform at high temp`}
      />

      <NoteBlock type="note" title="Common Attention Patterns">
        <p>
          Trained transformers exhibit recurring patterns: <strong>diagonal</strong> (attending to same position),
          <strong>vertical stripes</strong> (attending to specific tokens like [CLS] or punctuation),
          and <strong>broad</strong> (roughly uniform). Different heads learn different patterns,
          enabling the model to capture diverse relationships.
        </p>
      </NoteBlock>
    </div>
  )
}
