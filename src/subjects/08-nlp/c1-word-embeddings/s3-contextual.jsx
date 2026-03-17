import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ContextDemo() {
  const [sentIdx, setSentIdx] = useState(0)
  const examples = [
    { sentence: 'I went to the river bank to fish.', word: 'bank', meaning: 'edge of a river' },
    { sentence: 'I deposited money at the bank.', word: 'bank', meaning: 'financial institution' },
    { sentence: 'The bank approved my loan application.', word: 'bank', meaning: 'financial institution' },
  ]
  const ex = examples[sentIdx]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Context-Dependent Representations</h3>
      <div className="flex gap-2 mb-4">
        {examples.map((e, i) => (
          <button key={i} onClick={() => setSentIdx(i)}
            className={`rounded px-3 py-1 text-sm transition ${i === sentIdx ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            Sentence {i + 1}
          </button>
        ))}
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300 mb-2">
        {ex.sentence.split(ex.word).map((part, i, arr) => (
          <span key={i}>{part}{i < arr.length - 1 && <span className="font-bold text-violet-600 dark:text-violet-400">{ex.word}</span>}</span>
        ))}
      </p>
      <p className="text-xs text-gray-500 dark:text-gray-400">
        ELMo representation of "<span className="font-semibold">{ex.word}</span>" here encodes: <span className="font-semibold text-violet-600 dark:text-violet-400">{ex.meaning}</span>
      </p>
    </div>
  )
}

export default function ContextualEmbeddings() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Unlike static embeddings that assign one vector per word, contextual embeddings produce
        different representations for the same word depending on its surrounding context. ELMo
        (Embeddings from Language Models) pioneered this approach using deep bidirectional LSTMs.
      </p>

      <DefinitionBlock title="ELMo Architecture">
        <p>ELMo runs a forward and backward LSTM language model over the input, then combines representations from all layers:</p>
        <BlockMath math="\text{ELMo}_k^{task} = \gamma^{task} \sum_{j=0}^{L} s_j^{task} \mathbf{h}_{k,j}" />
        <p className="mt-2">
          where <InlineMath math="\mathbf{h}_{k,j}" /> is the hidden state at position <InlineMath math="k" /> from
          layer <InlineMath math="j" />, <InlineMath math="s_j" /> are softmax-normalized layer weights, and <InlineMath math="\gamma" /> is a task-specific scalar.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Bidirectional Language Model Objective" id="bilm-objective">
        <p>ELMo jointly maximizes the forward and backward log-likelihoods:</p>
        <BlockMath math="\mathcal{L} = \sum_{k=1}^{N}\left(\log P(t_k \mid t_1, \ldots, t_{k-1}) + \log P(t_k \mid t_{k+1}, \ldots, t_N)\right)" />
        <p className="mt-2">The forward and backward LSTMs share the token embedding and softmax layers but have separate LSTM parameters.</p>
      </TheoremBlock>

      <ContextDemo />

      <ExampleBlock title="Layer-wise Representations">
        <p>Different ELMo layers capture different linguistic information:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>Layer 0 (character CNN):</strong> morphology, word shape</li>
          <li><strong>Layer 1 (first biLSTM):</strong> syntax, POS tags</li>
          <li><strong>Layer 2 (second biLSTM):</strong> semantics, word sense</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Using ELMo Embeddings with AllenNLP"
        code={`from allennlp.modules.elmo import Elmo, batch_to_ids

# Load pretrained ELMo (2 layers, 1024-dim)
options_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, num_output_representations=1)

# Two sentences with "bank" in different contexts
sentences = [
    ["I", "went", "to", "the", "river", "bank"],
    ["I", "deposited", "money", "at", "the", "bank"],
]
char_ids = batch_to_ids(sentences)

# Get contextual embeddings
embeddings = elmo(char_ids)
elmo_repr = embeddings["elmo_representations"][0]
print(f"Shape: {elmo_repr.shape}")  # (2, 6, 256)

# "bank" in sentence 1 vs sentence 2 will differ
import torch
cos_sim = torch.nn.functional.cosine_similarity(
    elmo_repr[0, 5], elmo_repr[1, 5], dim=0
)
print(f"Cosine sim of 'bank' in two contexts: {cos_sim:.4f}")`}
      />

      <NoteBlock type="note" title="From ELMo to Transformers">
        <p>
          ELMo demonstrated that contextual representations dramatically improve downstream NLP
          tasks. However, its biLSTM architecture limits parallelization and long-range dependencies.
          BERT and GPT later replaced LSTMs with Transformers, achieving even stronger contextual
          representations. ELMo remains historically important as the bridge between static embeddings
          and modern pretrained language models.
        </p>
      </NoteBlock>
    </div>
  )
}
