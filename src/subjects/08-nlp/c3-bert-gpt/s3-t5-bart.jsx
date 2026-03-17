import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function TextToTextDemo() {
  const [task, setTask] = useState('translate')
  const examples = {
    translate: { input: 'translate English to German: The house is wonderful.', output: 'Das Haus ist wunderbar.' },
    summarize: { input: 'summarize: State authorities dispatched emergency crews Tuesday to survey the damage after a series of powerful storms.', output: 'Emergency crews were sent to survey storm damage.' },
    sentiment: { input: 'sst2 sentence: This movie was absolutely fantastic!', output: 'positive' },
    cola: { input: 'cola sentence: The boy was reminded his homework.', output: 'unacceptable' },
  }
  const ex = examples[task]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">T5 Text-to-Text Framework</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(examples).map(t => (
          <button key={t} onClick={() => setTask(t)}
            className={`rounded px-3 py-1 text-sm transition ${t === task ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {t}
          </button>
        ))}
      </div>
      <div className="space-y-2">
        <div className="rounded bg-gray-50 p-3 text-sm dark:bg-gray-800">
          <span className="text-xs font-semibold text-violet-600 dark:text-violet-400">Input: </span>
          <span className="text-gray-700 dark:text-gray-300">{ex.input}</span>
        </div>
        <div className="rounded bg-violet-50 p-3 text-sm dark:bg-violet-900/20">
          <span className="text-xs font-semibold text-violet-600 dark:text-violet-400">Output: </span>
          <span className="text-gray-700 dark:text-gray-300">{ex.output}</span>
        </div>
      </div>
    </div>
  )
}

export default function T5BART() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        T5 and BART are encoder-decoder Transformer models that unify NLP tasks through
        sequence-to-sequence formulations. T5 casts every task as text-to-text, while BART
        uses denoising autoencoding as its pretraining objective.
      </p>

      <DefinitionBlock title="T5: Text-to-Text Transfer Transformer">
        <p>T5 converts every NLP task into a text-to-text format with task-specific prefixes:</p>
        <BlockMath math="\text{Input: prefix + input text} \rightarrow \text{Output: target text}" />
        <p className="mt-2">
          Pretrained with a span corruption objective: random spans of tokens are replaced with
          sentinel tokens, and the model learns to generate the missing spans.
        </p>
      </DefinitionBlock>

      <TextToTextDemo />

      <DefinitionBlock title="BART: Denoising Autoencoder">
        <p>BART corrupts text with arbitrary noise and trains to reconstruct the original. Noise types include:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Token masking (like BERT)</li>
          <li>Token deletion (model must decide what is missing)</li>
          <li>Text infilling (span replaced with single mask)</li>
          <li>Sentence permutation (shuffle sentence order)</li>
          <li>Document rotation (shift start position)</li>
        </ul>
        <BlockMath math="\mathcal{L}_{\text{BART}} = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, \text{corrupt}(\mathbf{x}))" />
      </DefinitionBlock>

      <TheoremBlock title="Span Corruption (T5 Pretraining)" id="span-corruption">
        <p>T5 randomly selects spans of mean length 3 covering 15% of tokens and replaces each span with a unique sentinel:</p>
        <BlockMath math="\text{Original: } w_1 w_2 \underline{w_3 w_4 w_5} w_6 \underline{w_7} w_8" />
        <BlockMath math="\text{Input: } w_1 w_2 \langle s_1 \rangle w_6 \langle s_2 \rangle w_8" />
        <BlockMath math="\text{Target: } \langle s_1 \rangle w_3 w_4 w_5 \langle s_2 \rangle w_7" />
      </TheoremBlock>

      <ExampleBlock title="Architecture Comparison">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>T5-Base:</strong> 220M params, encoder-decoder, relative position bias</li>
          <li><strong>T5-Large:</strong> 770M params, 24 layers each for encoder and decoder</li>
          <li><strong>BART-Base:</strong> 140M params, 6+6 layers, absolute position embeddings</li>
          <li><strong>BART-Large:</strong> 400M params, 12+12 layers</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Using T5 and BART for Summarization"
        code={`from transformers import pipeline

# T5 summarization
t5_summarizer = pipeline("summarization", model="t5-small")
text = """
The tower is 324 metres tall, about the same height as an 81-storey building.
It was the tallest man-made structure in the world until the Chrysler Building
in New York City was finished in 1930. It is now the tallest structure in Paris.
"""
summary = t5_summarizer(text, max_length=40, min_length=10)
print(f"T5 summary: {summary[0]['summary_text']}")

# BART summarization
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
summary = bart_summarizer(text, max_length=40, min_length=10)
print(f"BART summary: {summary[0]['summary_text']}")

# T5 for translation (text-to-text format)
from transformers import T5ForConditionalGeneration, T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")

input_text = "translate English to French: How are you?"
ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(ids, max_new_tokens=20)
print(f"Translation: {tokenizer.decode(output[0], skip_special_tokens=True)}")`}
      />

      <NoteBlock type="note" title="Encoder-Decoder vs Decoder-Only">
        <p>
          T5 and BART use the full encoder-decoder architecture, making them particularly strong
          for conditional generation tasks (summarization, translation). In contrast, decoder-only
          models like GPT handle these via autoregressive prompting. Recent work (e.g., Flan-T5)
          showed that instruction-tuned encoder-decoder models can be surprisingly competitive
          with much larger decoder-only models.
        </p>
      </NoteBlock>
    </div>
  )
}
