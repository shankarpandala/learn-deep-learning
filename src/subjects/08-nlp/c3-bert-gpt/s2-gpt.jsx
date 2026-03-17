import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function TemperatureSampling() {
  const [temp, setTemp] = useState(1.0)
  const rawLogits = [2.0, 1.5, 0.8, 0.3, -0.5]
  const words = ['the', 'a', 'one', 'this', 'some']

  const scaled = rawLogits.map(l => l / temp)
  const maxScaled = Math.max(...scaled)
  const exps = scaled.map(s => Math.exp(s - maxScaled))
  const sumExp = exps.reduce((a, b) => a + b, 0)
  const probs = exps.map(e => e / sumExp)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Temperature Sampling</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
        Temperature: {temp.toFixed(2)}
        <input type="range" min={0.1} max={3.0} step={0.1} value={temp}
          onChange={e => setTemp(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="space-y-2">
        {words.map((w, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="w-12 text-xs font-mono text-gray-600 dark:text-gray-400">{w}</span>
            <div className="flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded overflow-hidden">
              <div className="h-full bg-violet-500 rounded transition-all duration-200" style={{ width: `${probs[i] * 100}%` }} />
            </div>
            <span className="w-14 text-xs text-gray-500 text-right">{(probs[i] * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
        {temp < 0.5 ? 'Low temperature: peaked distribution (greedy)' : temp > 1.5 ? 'High temperature: flat distribution (creative)' : 'Moderate temperature: balanced sampling'}
      </p>
    </div>
  )
}

export default function GPT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The GPT (Generative Pre-trained Transformer) series uses autoregressive language modeling
        with a decoder-only Transformer. From GPT-1 to GPT-4, the series demonstrated that
        scaling model size, data, and compute leads to emergent capabilities including in-context learning.
      </p>

      <DefinitionBlock title="Autoregressive Language Model Objective">
        <p>GPT is trained to predict the next token given all preceding tokens:</p>
        <BlockMath math="\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_1, \ldots, x_{t-1}; \theta)" />
        <p className="mt-2">The causal (left-to-right) attention mask ensures each position can only attend to previous positions:</p>
        <BlockMath math="\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V" />
        <p className="mt-1">where <InlineMath math="M_{ij} = -\infty" /> if <InlineMath math="i < j" />, else 0.</p>
      </DefinitionBlock>

      <TemperatureSampling />

      <TheoremBlock title="In-Context Learning" id="in-context-learning">
        <p>
          GPT-3 showed that large language models can perform tasks by conditioning on
          demonstrations in the prompt without any gradient updates:
        </p>
        <BlockMath math="P(y \mid x, \{(x_1, y_1), \ldots, (x_k, y_k)\})" />
        <p className="mt-2">
          This is called <InlineMath math="k" />-shot learning. When <InlineMath math="k = 0" />,
          the model relies solely on task instructions (zero-shot).
        </p>
      </TheoremBlock>

      <ExampleBlock title="GPT Model Evolution">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>GPT-1 (2018):</strong> 117M params, 12 layers, pretrain + fine-tune</li>
          <li><strong>GPT-2 (2019):</strong> 1.5B params, 48 layers, zero-shot task transfer</li>
          <li><strong>GPT-3 (2020):</strong> 175B params, 96 layers, in-context learning</li>
          <li><strong>GPT-4 (2023):</strong> multimodal, RLHF-aligned, state-of-the-art</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Text Generation with GPT-2"
        code={`from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode prompt
prompt = "The future of artificial intelligence"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate with different strategies
# Greedy
greedy = model.generate(input_ids, max_new_tokens=30)
print("Greedy:", tokenizer.decode(greedy[0]))

# Nucleus (top-p) sampling
sampled = model.generate(
    input_ids, max_new_tokens=30,
    do_sample=True, top_p=0.9, temperature=0.8
)
print("Top-p:", tokenizer.decode(sampled[0]))`}
      />

      <WarningBlock title="Autoregressive vs Bidirectional">
        <p>
          GPT can only attend to left context, while BERT sees both directions. This makes GPT
          naturally suited for generation but potentially weaker for understanding tasks that
          benefit from full bidirectional context. BERT excels at classification and extraction,
          while GPT excels at generation and few-shot prompting.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Scaling Laws">
        <p>
          Kaplan et al. (2020) showed that language model performance follows predictable power
          laws with respect to model size (<InlineMath math="N" />), dataset size (<InlineMath math="D" />),
          and compute (<InlineMath math="C" />): <InlineMath math="L(N) \propto N^{-0.076}" />.
          This motivated the push toward ever-larger models in the GPT series.
        </p>
      </NoteBlock>
    </div>
  )
}
