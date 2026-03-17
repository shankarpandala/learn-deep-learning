import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function AnalogyExplorer() {
  const [wordA, setWordA] = useState('king')
  const [wordB, setWordB] = useState('man')
  const [wordC, setWordC] = useState('woman')

  const analogies = {
    'king-man+woman': 'queen',
    'paris-france+germany': 'berlin',
    'walked-walk+swim': 'swam',
  }

  const key = `${wordA}-${wordB}+${wordC}`
  const result = analogies[key] || '???'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Word Analogy Explorer</h3>
      <div className="flex flex-wrap items-center gap-3 mb-4">
        <select value={wordA} onChange={e => setWordA(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
          <option value="king">king</option><option value="paris">paris</option><option value="walked">walked</option>
        </select>
        <span className="text-violet-600 font-bold">-</span>
        <select value={wordB} onChange={e => setWordB(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
          <option value="man">man</option><option value="france">france</option><option value="walk">walk</option>
        </select>
        <span className="text-violet-600 font-bold">+</span>
        <select value={wordC} onChange={e => setWordC(e.target.value)} className="rounded border px-2 py-1 text-sm dark:bg-gray-800 dark:border-gray-600">
          <option value="woman">woman</option><option value="germany">germany</option><option value="swim">swim</option>
        </select>
        <span className="text-violet-600 font-bold">=</span>
        <span className="rounded bg-violet-100 px-3 py-1 font-bold text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">{result}</span>
      </div>
      <p className="text-xs text-gray-500 dark:text-gray-400">Select matching triplets to see the analogy result.</p>
    </div>
  )
}

export default function Word2Vec() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Word2Vec learns dense vector representations of words by training shallow neural networks on
        word co-occurrence patterns. Two architectures were proposed: Skip-gram and CBOW (Continuous
        Bag of Words), both producing embeddings that capture semantic relationships.
      </p>

      <DefinitionBlock title="Skip-gram Objective">
        <p>Given a center word <InlineMath math="w_t" />, predict surrounding context words within a window of size <InlineMath math="m" />:</p>
        <BlockMath math="J(\theta) = -\frac{1}{T}\sum_{t=1}^{T}\sum_{\substack{-m \leq j \leq m \\ j \neq 0}} \log P(w_{t+j} \mid w_t)" />
        <p className="mt-2">where <InlineMath math="P(o \mid c) = \frac{\exp(\mathbf{u}_o^\top \mathbf{v}_c)}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \mathbf{v}_c)}" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="CBOW (Continuous Bag of Words)">
        <p>CBOW predicts the center word from the average of its context word vectors:</p>
        <BlockMath math="P(w_t \mid w_{t-m}, \ldots, w_{t+m}) = \frac{\exp(\mathbf{u}_{w_t}^\top \bar{\mathbf{v}})}{\sum_{w \in V} \exp(\mathbf{u}_w^\top \bar{\mathbf{v}})}" />
        <p className="mt-2">where <InlineMath math="\bar{\mathbf{v}} = \frac{1}{2m}\sum_{\substack{-m \leq j \leq m, j \neq 0}} \mathbf{v}_{w_{t+j}}" /></p>
      </DefinitionBlock>

      <DefinitionBlock title="Negative Sampling">
        <p>The full softmax is expensive. Negative sampling approximates it by contrasting true pairs against <InlineMath math="k" /> randomly drawn negative samples:</p>
        <BlockMath math="J_{\text{NEG}} = -\log \sigma(\mathbf{u}_o^\top \mathbf{v}_c) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)}[\log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_c)]" />
        <p className="mt-2">Noise distribution: <InlineMath math="P_n(w) \propto f(w)^{3/4}" /></p>
      </DefinitionBlock>

      <AnalogyExplorer />

      <ExampleBlock title="Word Analogy via Vector Arithmetic">
        <p>The famous analogy relationship in embedding space:</p>
        <BlockMath math="\mathbf{v}_{\text{king}} - \mathbf{v}_{\text{man}} + \mathbf{v}_{\text{woman}} \approx \mathbf{v}_{\text{queen}}" />
        <p>This works because the gender direction is encoded consistently across word pairs.</p>
      </ExampleBlock>

      <PythonCode
        title="Training Word2Vec with Gensim"
        code={`from gensim.models import Word2Vec

# Example corpus (list of tokenized sentences)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "lay", "on", "the", "rug"],
    ["cats", "and", "dogs", "are", "pets"],
]

# Train Skip-gram model (sg=1), CBOW would be sg=0
model = Word2Vec(
    sentences, vector_size=100, window=5,
    min_count=1, sg=1, negative=5, epochs=100
)

# Get word vector
vec = model.wv["cat"]
print(f"Vector shape: {vec.shape}")  # (100,)

# Find most similar words
similar = model.wv.most_similar("cat", topn=3)
print(f"Similar to 'cat': {similar}")`}
      />

      <WarningBlock title="Limitations of Static Embeddings">
        <p>
          Word2Vec assigns a single vector per word regardless of context. The word "bank" gets
          the same representation whether it refers to a river bank or a financial bank. This
          limitation motivates contextual embeddings like ELMo and BERT.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Skip-gram vs CBOW">
        <p>
          <strong>Skip-gram</strong> works better for rare words and smaller datasets since each
          word gets more training signal. <strong>CBOW</strong> is faster to train and performs
          slightly better on frequent words. In practice, skip-gram with negative sampling is
          the more widely used variant.
        </p>
      </NoteBlock>
    </div>
  )
}
