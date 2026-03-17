import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function DataMixtureExplorer() {
  const [web, setWeb] = useState(50)
  const [code, setCode] = useState(20)
  const [books, setBooks] = useState(15)
  const total = web + code + books
  const other = Math.max(0, 100 - total)
  const sources = [
    { name: 'Web Text', pct: web, color: 'bg-violet-400' },
    { name: 'Code', pct: code, color: 'bg-violet-600' },
    { name: 'Books/Papers', pct: books, color: 'bg-violet-300' },
    { name: 'Other (wiki, etc.)', pct: other, color: 'bg-gray-300' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Training Data Mixture</h3>
      <div className="space-y-2 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Web: {web}% <input type="range" min={0} max={80} value={web} onChange={e => setWeb(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Code: {code}% <input type="range" min={0} max={40} value={code} onChange={e => setCode(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Books: {books}% <input type="range" min={0} max={30} value={books} onChange={e => setBooks(parseInt(e.target.value))} className="w-32 accent-violet-500" />
        </label>
      </div>
      <div className="h-6 flex rounded overflow-hidden">
        {sources.map((s, i) => s.pct > 0 && <div key={i} className={`${s.color} transition-all`} style={{ width: `${s.pct}%` }} />)}
      </div>
      <div className="flex gap-3 mt-2 flex-wrap text-xs text-gray-500">
        {sources.map((s, i) => <span key={i} className="flex items-center gap-1"><span className={`w-2 h-2 rounded ${s.color}`} />{s.name}: {s.pct}%</span>)}
      </div>
      {total > 100 && <p className="text-xs text-red-500 mt-1">Total exceeds 100% — reduce one category</p>}
    </div>
  )
}

export default function DataScalingQuality() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Data quality, diversity, and mixture composition have emerged as critical factors in LLM
        training — often more impactful than raw data volume. Research into data curation, filtering,
        and synthetic data generation is reshaping how models are trained.
      </p>

      <DefinitionBlock title="Data-Constrained Scaling">
        <p>When unique data <InlineMath math="D_u" /> is limited, repeating data has diminishing returns. The effective data follows:</p>
        <BlockMath math="D_{\text{eff}}(R, D_u) = D_u \cdot (1 - e^{-R})" />
        <p className="mt-2">where <InlineMath math="R = D_{\text{total}} / D_u" /> is the number of epochs. After ~4 epochs, additional repetitions contribute minimally. The loss modification becomes:</p>
        <BlockMath math="L(N, D_u, R) = E + \frac{A}{N^{\alpha}} + \frac{B}{D_{\text{eff}}(R, D_u)^{\beta}}" />
      </DefinitionBlock>

      <DataMixtureExplorer />

      <ExampleBlock title="Data Quality Interventions">
        <p>Key data curation techniques and their impact:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>Deduplication:</strong> Removing near-duplicates improves perplexity by 0.1-0.3 nats and reduces memorization</li>
          <li><strong>Quality filtering:</strong> Perplexity-based or classifier-based filtering yields 2-5x data efficiency gains</li>
          <li><strong>Domain upsampling:</strong> Over-representing high-quality domains (Wikipedia, textbooks) improves downstream tasks</li>
          <li><strong>Synthetic data:</strong> LLM-generated data can supplement scarce domains (math, reasoning)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Data Quality Filtering Pipeline"
        code={`import math

def perplexity_filter(documents, reference_model_ppl, threshold_low=10, threshold_high=1000):
    """Filter documents by perplexity from a reference model.

    - Too low perplexity: likely repetitive/template text
    - Too high perplexity: likely noise/gibberish
    - Medium perplexity: natural, informative text
    """
    filtered = []
    stats = {"kept": 0, "too_low": 0, "too_high": 0}
    for doc, ppl in zip(documents, reference_model_ppl):
        if ppl < threshold_low:
            stats["too_low"] += 1
        elif ppl > threshold_high:
            stats["too_high"] += 1
        else:
            filtered.append(doc)
            stats["kept"] += 1
    return filtered, stats

def effective_data(unique_tokens, epochs):
    """Compute effective training data with repetition."""
    return unique_tokens * (1 - math.exp(-epochs))

# Demonstrate diminishing returns of data repetition
unique = 1e12  # 1T unique tokens
print("Epochs | Total Tokens | Effective Tokens | Efficiency")
print("-" * 55)
for e in [1, 2, 4, 8, 16]:
    total = unique * e
    eff = effective_data(unique, e)
    efficiency = eff / total * 100
    print(f"  {e:>4}  | {total/1e12:.1f}T          | {eff/1e12:.2f}T            | {efficiency:.1f}%")
print("\\nAfter ~4 epochs, >98% of effective data has been captured")`}
      />

      <WarningBlock title="The Data Wall">
        <p>
          Estimates suggest only 1-10T tokens of high-quality natural text exist on the internet.
          With leading models already consuming 15T+ tokens (with repetition), the field is approaching
          a "data wall." Solutions include synthetic data generation, multimodal data, and more
          data-efficient training methods. This is one of the most pressing challenges for continued
          scaling.
        </p>
      </WarningBlock>
    </div>
  )
}
