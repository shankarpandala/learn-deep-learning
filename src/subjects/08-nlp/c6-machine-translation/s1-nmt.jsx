import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function AttentionHeatmap() {
  const [highlighted, setHighlighted] = useState(null)
  const src = ['The', 'cat', 'sat', 'on', 'the', 'mat']
  const tgt = ['Le', 'chat', 'assis', 'sur', 'le', 'tapis']
  const weights = [
    [0.8, 0.05, 0.02, 0.03, 0.05, 0.05],
    [0.05, 0.85, 0.02, 0.02, 0.03, 0.03],
    [0.02, 0.05, 0.82, 0.05, 0.03, 0.03],
    [0.03, 0.02, 0.05, 0.80, 0.05, 0.05],
    [0.05, 0.03, 0.02, 0.05, 0.80, 0.05],
    [0.02, 0.03, 0.03, 0.02, 0.05, 0.85],
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Attention Alignment Heatmap</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Hover over target words to see attention weights:</p>
      <div className="inline-grid gap-0.5" style={{ gridTemplateColumns: `80px repeat(${src.length}, 40px)` }}>
        <div />
        {src.map((s, i) => <div key={i} className="text-xs text-center text-gray-600 dark:text-gray-400 font-mono">{s}</div>)}
        {tgt.map((t, ti) => (
          <>
            <div key={`l-${ti}`} className="text-xs text-right pr-2 text-gray-600 dark:text-gray-400 font-mono leading-8 cursor-pointer"
              onMouseEnter={() => setHighlighted(ti)} onMouseLeave={() => setHighlighted(null)}>{t}</div>
            {src.map((_, si) => {
              const w = weights[ti][si]
              const opacity = highlighted === ti ? w : 0.2
              return <div key={`${ti}-${si}`} className="w-10 h-8 rounded-sm transition-opacity duration-150"
                style={{ backgroundColor: `rgba(139, 92, 246, ${opacity})` }} />
            })}
          </>
        ))}
      </div>
    </div>
  )
}

export default function NMT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Neural Machine Translation (NMT) uses encoder-decoder architectures to translate text
        between languages. The encoder processes the source sentence into a continuous
        representation, and the decoder generates the target sentence token by token.
      </p>

      <DefinitionBlock title="Encoder-Decoder Translation">
        <p>NMT models the conditional probability of a target sentence given a source sentence:</p>
        <BlockMath math="P(\mathbf{y} \mid \mathbf{x}) = \prod_{t=1}^{T_y} P(y_t \mid y_{<t}, \mathbf{x}; \theta)" />
        <p className="mt-2">The training objective maximizes log-likelihood over parallel corpora:</p>
        <BlockMath math="\mathcal{L}(\theta) = \sum_{(\mathbf{x}, \mathbf{y}) \in \mathcal{D}} \sum_{t=1}^{T_y} \log P(y_t \mid y_{<t}, \mathbf{x}; \theta)" />
      </DefinitionBlock>

      <AttentionHeatmap />

      <TheoremBlock title="Beam Search Decoding" id="beam-search">
        <p>Beam search maintains <InlineMath math="B" /> best partial hypotheses at each step:</p>
        <BlockMath math="\mathcal{H}_t = \text{top-}B\left\{(h \oplus w, s_h + \log P(w \mid h)) : h \in \mathcal{H}_{t-1}, w \in V\right\}" />
        <p className="mt-2">
          Length normalization prevents beam search from preferring shorter translations:
        </p>
        <BlockMath math="\text{score}(\mathbf{y}) = \frac{\log P(\mathbf{y} \mid \mathbf{x})}{|\mathbf{y}|^\alpha}" />
        <p className="mt-1">with <InlineMath math="\alpha \in [0.6, 0.7]" /> typically.</p>
      </TheoremBlock>

      <ExampleBlock title="Evolution of NMT Architectures">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Seq2Seq + Attention (2014-2016):</strong> LSTM/GRU encoder-decoder with Bahdanau attention</li>
          <li><strong>ConvS2S (2017):</strong> Fully convolutional with multi-step attention</li>
          <li><strong>Transformer (2017):</strong> Self-attention only, parallel training, dominant architecture</li>
          <li><strong>LLM-based (2023+):</strong> Large decoder-only models prompted for translation</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Neural Machine Translation with Hugging Face"
        code={`from transformers import MarianMTModel, MarianTokenizer

# Load English-to-French translation model
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Translate
texts = [
    "The weather is beautiful today.",
    "Machine translation has improved dramatically.",
]

inputs = tokenizer(texts, return_tensors="pt", padding=True)
translated = model.generate(**inputs, num_beams=4, max_length=64)

for src, tgt in zip(texts, translated):
    decoded = tokenizer.decode(tgt, skip_special_tokens=True)
    print(f"EN: {src}")
    print(f"FR: {decoded}\\n")`}
      />

      <WarningBlock title="Translation Challenges">
        <p>
          NMT still struggles with rare words, domain-specific terminology, very long sentences,
          and low-resource language pairs. Hallucination (generating fluent but unfaithful
          translations) is a critical safety concern, especially in medical or legal domains.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Transformer Dominance">
        <p>
          The Transformer architecture, introduced in "Attention Is All You Need" (2017), replaced
          RNN-based NMT models due to superior parallelization and handling of long-range
          dependencies. Modern NMT systems are almost exclusively Transformer-based, with
          multilingual models like NLLB-200 covering 200 languages in a single model.
        </p>
      </NoteBlock>
    </div>
  )
}
