import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExerciseBlock from '../../../components/content/ExerciseBlock.jsx'

function BLEUCalculator() {
  const [reference, setReference] = useState('The cat sat on the mat')
  const [hypothesis, setHypothesis] = useState('The cat is on the mat')

  const refTokens = reference.toLowerCase().split(/\s+/)
  const hypTokens = hypothesis.toLowerCase().split(/\s+/)

  const ngramPrecision = (n) => {
    const refNgrams = {}
    for (let i = 0; i <= refTokens.length - n; i++) {
      const ng = refTokens.slice(i, i + n).join(' ')
      refNgrams[ng] = (refNgrams[ng] || 0) + 1
    }
    let matches = 0, total = 0
    const used = { ...refNgrams }
    for (let i = 0; i <= hypTokens.length - n; i++) {
      const ng = hypTokens.slice(i, i + n).join(' ')
      total++
      if (used[ng] && used[ng] > 0) { matches++; used[ng]-- }
    }
    return total > 0 ? matches / total : 0
  }

  const p1 = ngramPrecision(1), p2 = ngramPrecision(2)
  const p3 = ngramPrecision(3), p4 = ngramPrecision(4)
  const bp = hypTokens.length >= refTokens.length ? 1 : Math.exp(1 - refTokens.length / hypTokens.length)
  const logAvg = [p1, p2, p3, p4].filter(p => p > 0)
  const bleu = logAvg.length > 0 ? bp * Math.exp(logAvg.reduce((s, p) => s + Math.log(p), 0) / 4) * 100 : 0

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">BLEU Score Calculator</h3>
      <div className="space-y-2 mb-4">
        <label className="block text-xs text-gray-600 dark:text-gray-400">
          Reference:
          <input type="text" value={reference} onChange={e => setReference(e.target.value)}
            className="mt-1 block w-full rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-gray-300" />
        </label>
        <label className="block text-xs text-gray-600 dark:text-gray-400">
          Hypothesis:
          <input type="text" value={hypothesis} onChange={e => setHypothesis(e.target.value)}
            className="mt-1 block w-full rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-gray-300" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-xs">
        <div className="text-gray-600 dark:text-gray-400">1-gram: <span className="font-bold text-violet-600">{(p1 * 100).toFixed(1)}%</span></div>
        <div className="text-gray-600 dark:text-gray-400">2-gram: <span className="font-bold text-violet-600">{(p2 * 100).toFixed(1)}%</span></div>
        <div className="text-gray-600 dark:text-gray-400">3-gram: <span className="font-bold text-violet-600">{(p3 * 100).toFixed(1)}%</span></div>
        <div className="text-gray-600 dark:text-gray-400">4-gram: <span className="font-bold text-violet-600">{(p4 * 100).toFixed(1)}%</span></div>
        <div className="text-gray-600 dark:text-gray-400">BP: <span className="font-bold text-violet-600">{bp.toFixed(3)}</span></div>
        <div className="text-gray-600 dark:text-gray-400">BLEU: <span className="font-bold text-violet-600">{bleu.toFixed(2)}</span></div>
      </div>
    </div>
  )
}

export default function MTEvaluation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Evaluating machine translation quality is challenging because many valid translations
        exist for any source sentence. Automatic metrics attempt to approximate human judgment
        by comparing system output against reference translations.
      </p>

      <DefinitionBlock title="BLEU (Bilingual Evaluation Understudy)">
        <p>BLEU computes modified n-gram precision with a brevity penalty:</p>
        <BlockMath math="\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)" />
        <p className="mt-2">where <InlineMath math="p_n" /> is the clipped n-gram precision and the brevity penalty is:</p>
        <BlockMath math="\text{BP} = \begin{cases} 1 & \text{if } c > r \\ e^{1 - r/c} & \text{if } c \leq r \end{cases}" />
        <p className="mt-1">with <InlineMath math="c" /> = hypothesis length and <InlineMath math="r" /> = reference length. Typically <InlineMath math="N = 4" /> with uniform weights.</p>
      </DefinitionBlock>

      <BLEUCalculator />

      <TheoremBlock title="COMET: Learned MT Evaluation" id="comet">
        <p>
          COMET uses a cross-lingual encoder (XLM-R) to score translations by comparing
          source, hypothesis, and reference embeddings:
        </p>
        <BlockMath math="\text{COMET}(s, h, r) = f_\theta(\mathbf{h}_s, \mathbf{h}_h, \mathbf{h}_r, \mathbf{h}_s \odot \mathbf{h}_h, \mathbf{h}_h \odot \mathbf{h}_r)" />
        <p className="mt-2">
          Trained on human quality judgments (DA scores), COMET achieves much higher correlation
          with human evaluation than BLEU.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Metric Comparison">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>BLEU:</strong> Fast, interpretable, but poor at sentence level, misses meaning</li>
          <li><strong>chrF:</strong> Character-level F-score, better for morphologically rich languages</li>
          <li><strong>BERTScore:</strong> Token-level cosine similarity with BERT embeddings</li>
          <li><strong>COMET:</strong> Trained on human judgments, highest human correlation</li>
          <li><strong>BLEURT:</strong> Fine-tuned BERT for quality estimation</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="MT Evaluation Metrics"
        code={`import sacrebleu
from comet import download_model, load_from_checkpoint

# BLEU with SacreBLEU (standardized implementation)
refs = [["The cat is on the mat."]]
hyps = ["The cat sits on the mat."]
bleu = sacrebleu.corpus_bleu(hyps, refs)
print(f"BLEU: {bleu.score:.2f}")
print(f"Signature: {bleu}")  # includes tokenization details

# chrF score
chrf = sacrebleu.corpus_chrf(hyps, refs)
print(f"chrF: {chrf.score:.2f}")

# COMET (neural metric, much higher human correlation)
model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)

data = [{
    "src": "Le chat est sur le tapis.",
    "mt": "The cat sits on the mat.",
    "ref": "The cat is on the mat."
}]
scores = model.predict(data, batch_size=8)
print(f"COMET score: {scores.system_score:.4f}")`}
      />

      <ExerciseBlock title="Exercise: BLEU Limitations">
        <p>
          Consider these reference-hypothesis pairs. Which pair would you rate higher as a human,
          and which would BLEU rate higher? Explain the discrepancy.
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Ref: "It is raining heavily" / Hyp A: "It is raining a lot" / Hyp B: "It raining is heavily"</li>
        </ul>
      </ExerciseBlock>

      <NoteBlock type="note" title="Human Evaluation Remains the Gold Standard">
        <p>
          No automatic metric perfectly correlates with human judgment. Best practice is to
          use COMET or BERTScore for development and always include human evaluation for
          final assessment. The WMT shared tasks standardize human evaluation protocols including
          Direct Assessment (DA) and Multidimensional Quality Metrics (MQM).
        </p>
      </NoteBlock>
    </div>
  )
}
