import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function NgramExplorer() {
  const [n, setN] = useState(2)
  const sentence = 'the cat sat on the mat'
  const tokens = sentence.split(' ')
  const ngrams = []
  for (let i = 0; i <= tokens.length - n; i++) {
    ngrams.push(tokens.slice(i, i + n).join(' '))
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">N-gram Explorer</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="text-sm text-gray-600 dark:text-gray-400">
          n = {n}
          <input type="range" min={1} max={4} value={n} onChange={e => setN(parseInt(e.target.value))}
            className="ml-2 w-28 accent-violet-500" />
        </label>
        <span className="text-xs text-gray-500 dark:text-gray-400">({ngrams.length} {n}-grams)</span>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">Sentence: "<span className="italic">{sentence}</span>"</p>
      <div className="flex flex-wrap gap-2">
        {ngrams.map((g, i) => (
          <span key={i} className="rounded bg-violet-100 px-2 py-1 text-xs font-mono text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">
            {g}
          </span>
        ))}
      </div>
    </div>
  )
}

export default function NgramNeural() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Language models assign probabilities to sequences of words. The evolution from n-gram
        models to neural language models represents a fundamental shift in how we model language,
        moving from count-based statistics to learned continuous representations.
      </p>

      <DefinitionBlock title="N-gram Language Model">
        <p>An n-gram model approximates the probability of a word given its full history using only the previous <InlineMath math="n-1" /> words:</p>
        <BlockMath math="P(w_t \mid w_1, \ldots, w_{t-1}) \approx P(w_t \mid w_{t-n+1}, \ldots, w_{t-1})" />
        <p className="mt-2">Estimated via maximum likelihood from counts:</p>
        <BlockMath math="P(w_t \mid w_{t-n+1:t-1}) = \frac{C(w_{t-n+1}, \ldots, w_t)}{C(w_{t-n+1}, \ldots, w_{t-1})}" />
      </DefinitionBlock>

      <NgramExplorer />

      <DefinitionBlock title="Neural Language Model (Bengio et al., 2003)">
        <p>The first neural LM maps each word to a learned embedding, concatenates the context window, and passes through a hidden layer:</p>
        <BlockMath math="P(w_t \mid w_{t-n+1:t-1}) = \text{softmax}\left(\mathbf{W}_2 \tanh(\mathbf{W}_1 [\mathbf{e}_{t-n+1}; \ldots; \mathbf{e}_{t-1}] + \mathbf{b}_1) + \mathbf{b}_2\right)" />
      </DefinitionBlock>

      <TheoremBlock title="Chain Rule of Probability" id="chain-rule-lm">
        <p>Any language model decomposes the joint probability of a sequence via the chain rule:</p>
        <BlockMath math="P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_1, \ldots, w_{t-1})" />
        <p className="mt-2">N-gram models truncate this conditioning to a fixed window; neural models can in principle condition on the entire history.</p>
      </TheoremBlock>

      <ExampleBlock title="Bigram Probability Calculation">
        <p>Given corpus: "the cat sat on the mat"</p>
        <BlockMath math="P(\text{cat} \mid \text{the}) = \frac{C(\text{the cat})}{C(\text{the})} = \frac{1}{2} = 0.5" />
        <p><InlineMath math="P(\text{mat} \mid \text{the}) = 1/2 = 0.5" /> as well, since "the" appears twice.</p>
      </ExampleBlock>

      <PythonCode
        title="Simple Neural Language Model in PyTorch"
        code={`import torch
import torch.nn as nn

class NeuralLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, context_size, hidden_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.fc1 = nn.Linear(context_size * embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # x: (batch, context_size) word indices
        embeds = self.embeddings(x).view(x.size(0), -1)
        h = torch.tanh(self.fc1(embeds))
        logits = self.fc2(h)
        return logits  # (batch, vocab_size)

# Example: bigram model (context_size=1)
model = NeuralLM(vocab_size=10000, embed_dim=128,
                 context_size=2, hidden_dim=256)
x = torch.randint(0, 10000, (32, 2))  # batch of bigrams
logits = model(x)
print(f"Output shape: {logits.shape}")  # (32, 10000)`}
      />

      <WarningBlock title="Sparsity Problem in N-grams">
        <p>
          N-gram models suffer from data sparsity: most possible n-grams never appear in the
          training corpus. Smoothing techniques (Laplace, Kneser-Ney) help but cannot solve the
          fundamental issue. Neural LMs address this by generalizing through continuous embeddings.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Historical Significance">
        <p>
          Bengio's 2003 neural LM was ahead of its time but too slow for practical use. It took
          a decade for RNN-based LMs and later Transformer LMs to make neural language modeling
          the dominant paradigm, culminating in models like GPT and BERT.
        </p>
      </NoteBlock>
    </div>
  )
}
