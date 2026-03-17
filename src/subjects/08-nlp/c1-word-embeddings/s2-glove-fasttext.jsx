import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function SubwordDemo() {
  const [word, setWord] = useState('unhappiness')
  const ngrams = (w, n = 3) => {
    const padded = `<${w}>`
    const grams = []
    for (let i = 0; i <= padded.length - n; i++) grams.push(padded.slice(i, i + n))
    return grams
  }

  const grams3 = ngrams(word, 3)
  const grams4 = ngrams(word, 4)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">FastText Subword Decomposition</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Word:
        <input type="text" value={word} onChange={e => setWord(e.target.value.toLowerCase())}
          className="rounded border px-2 py-1 dark:bg-gray-800 dark:border-gray-600 w-40" />
      </label>
      <div className="mb-2">
        <span className="text-xs font-semibold text-violet-600 dark:text-violet-400">3-grams: </span>
        <span className="text-xs text-gray-600 dark:text-gray-400">{grams3.join(', ')}</span>
      </div>
      <div>
        <span className="text-xs font-semibold text-violet-600 dark:text-violet-400">4-grams: </span>
        <span className="text-xs text-gray-600 dark:text-gray-400">{grams4.join(', ')}</span>
      </div>
      <p className="mt-2 text-xs text-gray-500">Total: {grams3.length + grams4.length} subword n-grams (3 and 4)</p>
    </div>
  )
}

export default function GloveFastText() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        GloVe and FastText are two influential extensions of word embeddings. GloVe combines
        global co-occurrence statistics with local context window methods, while FastText enriches
        word vectors with subword (character n-gram) information.
      </p>

      <DefinitionBlock title="GloVe: Global Vectors for Word Representation">
        <p>GloVe directly factorizes the log co-occurrence matrix. Let <InlineMath math="X_{ij}" /> be the count of word <InlineMath math="j" /> appearing in context of word <InlineMath math="i" />:</p>
        <BlockMath math="J = \sum_{i,j=1}^{|V|} f(X_{ij})\left(\mathbf{w}_i^\top \tilde{\mathbf{w}}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2" />
        <p className="mt-2">where <InlineMath math="f(x)" /> is a weighting function that caps very frequent co-occurrences.</p>
      </DefinitionBlock>

      <TheoremBlock title="GloVe Weighting Function" id="glove-weighting">
        <p>The weighting function prevents extremely common pairs from dominating:</p>
        <BlockMath math="f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}" />
        <p className="mt-2">Typically <InlineMath math="\alpha = 0.75" /> and <InlineMath math="x_{\max} = 100" />.</p>
      </TheoremBlock>

      <DefinitionBlock title="FastText: Subword Embeddings">
        <p>FastText represents each word as the sum of its character n-gram vectors plus the word vector itself:</p>
        <BlockMath math="\mathbf{v}_w = \mathbf{z}_w + \sum_{g \in G_w} \mathbf{z}_g" />
        <p className="mt-2">
          where <InlineMath math="G_w" /> is the set of character n-grams (typically 3 to 6 characters)
          of the word <InlineMath math="w" /> with boundary markers.
        </p>
      </DefinitionBlock>

      <SubwordDemo />

      <ExampleBlock title="Handling Out-of-Vocabulary Words">
        <p>
          For an unseen word like "unfriendliness", FastText computes its embedding from subword
          n-grams shared with training vocabulary words like "unfriendly", "friendliness", and "kindness".
          GloVe and Word2Vec would return no embedding for this word.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Loading GloVe and FastText Embeddings"
        code={`import numpy as np
import gensim.downloader as api

# Load pretrained GloVe (converted to Word2Vec format)
glove = api.load("glove-wiki-gigaword-100")
print(f"GloVe vocab size: {len(glove)}")
print(f"'king' vector shape: {glove['king'].shape}")

# GloVe analogy: king - man + woman ≈ queen
result = glove.most_similar(
    positive=["king", "woman"],
    negative=["man"], topn=1
)
print(f"king - man + woman = {result[0][0]}")  # queen

# Load pretrained FastText
import fasttext
ft = fasttext.load_model("cc.en.300.bin")

# FastText handles OOV words via subwords
oov_vec = ft.get_word_vector("unfriendliness")
print(f"OOV vector norm: {np.linalg.norm(oov_vec):.3f}")`}
      />

      <NoteBlock type="note" title="When to Choose Which">
        <p>
          <strong>GloVe</strong> excels when you have large corpora and want high-quality static
          embeddings quickly. <strong>FastText</strong> is preferred for morphologically rich
          languages (German, Turkish, Finnish) and when dealing with misspellings, rare words,
          or domain-specific vocabulary not seen during training.
        </p>
      </NoteBlock>
    </div>
  )
}
