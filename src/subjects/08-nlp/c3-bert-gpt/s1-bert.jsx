import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function MaskedLMDemo() {
  const [maskIdx, setMaskIdx] = useState(3)
  const words = ['The', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
  const predictions = {
    0: ['The', 'A', 'One'],
    1: ['quick', 'slow', 'fast'],
    2: ['brown', 'red', 'big'],
    3: ['fox', 'cat', 'dog'],
    4: ['jumps', 'runs', 'leaps'],
    5: ['over', 'across', 'onto'],
    6: ['the', 'a', 'this'],
    7: ['lazy', 'sleepy', 'old'],
    8: ['dog', 'cat', 'hound'],
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Masked Language Model Demo</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">Click a word to mask it and see BERT-style predictions:</p>
      <div className="flex flex-wrap gap-2 mb-4">
        {words.map((w, i) => (
          <button key={i} onClick={() => setMaskIdx(i)}
            className={`rounded px-2 py-1 text-sm font-mono transition ${i === maskIdx ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-300'}`}>
            {i === maskIdx ? '[MASK]' : w}
          </button>
        ))}
      </div>
      <div className="text-sm text-gray-600 dark:text-gray-400">
        Top predictions: {predictions[maskIdx].map((p, i) => (
          <span key={i} className="ml-2 rounded bg-violet-100 px-2 py-0.5 text-xs text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">
            {p} ({(0.9 - i * 0.25).toFixed(2)})
          </span>
        ))}
      </div>
    </div>
  )
}

export default function BERT() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        BERT (Bidirectional Encoder Representations from Transformers) revolutionized NLP by
        pretraining a deep bidirectional Transformer using masked language modeling, then
        fine-tuning on downstream tasks with minimal architectural changes.
      </p>

      <DefinitionBlock title="Masked Language Modeling (MLM)">
        <p>BERT randomly masks 15% of input tokens and predicts them from bidirectional context:</p>
        <BlockMath math="\mathcal{L}_{\text{MLM}} = -\mathbb{E}\left[\sum_{i \in \mathcal{M}} \log P(x_i \mid \mathbf{x}_{\backslash \mathcal{M}})\right]" />
        <p className="mt-2">Of the selected 15% tokens: 80% replaced with [MASK], 10% with a random token, 10% kept unchanged.</p>
      </DefinitionBlock>

      <MaskedLMDemo />

      <DefinitionBlock title="Next Sentence Prediction (NSP)">
        <p>BERT is also trained to classify whether sentence B follows sentence A:</p>
        <BlockMath math="\mathcal{L}_{\text{NSP}} = -\left[y \log P(\text{IsNext}) + (1-y)\log P(\text{NotNext})\right]" />
        <p className="mt-2">Input format: <code>[CLS] sentence A [SEP] sentence B [SEP]</code></p>
      </DefinitionBlock>

      <TheoremBlock title="BERT Input Representation" id="bert-input">
        <p>Each input token representation is the sum of three embeddings:</p>
        <BlockMath math="\mathbf{e}_i = \mathbf{E}_{\text{token}}(x_i) + \mathbf{E}_{\text{segment}}(s_i) + \mathbf{E}_{\text{position}}(i)" />
        <p className="mt-2">
          Token embeddings use WordPiece, segment embeddings distinguish sentence A from B,
          and position embeddings are learned (not sinusoidal).
        </p>
      </TheoremBlock>

      <ExampleBlock title="BERT Model Sizes">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>BERT-Base:</strong> 12 layers, 768 hidden, 12 heads, 110M parameters</li>
          <li><strong>BERT-Large:</strong> 24 layers, 1024 hidden, 16 heads, 340M parameters</li>
        </ul>
        <p className="mt-2">Both trained on BookCorpus + English Wikipedia (~3.3B words).</p>
      </ExampleBlock>

      <PythonCode
        title="Fine-tuning BERT for Classification"
        code={`from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

# Load pretrained BERT with a classification head
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Tokenize input
inputs = tokenizer(
    "BERT is a powerful model for NLP tasks.",
    return_tensors="pt", padding=True, truncation=True
)
print(f"Input IDs shape: {inputs['input_ids'].shape}")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    pred = torch.argmax(logits, dim=-1)
    print(f"Predicted class: {pred.item()}")`}
      />

      <NoteBlock type="note" title="BERT's Legacy">
        <p>
          BERT established the pretrain-then-fine-tune paradigm that dominates modern NLP.
          Subsequent models like RoBERTa (more data, no NSP), ALBERT (parameter sharing),
          and DeBERTa (disentangled attention) improved on BERT's recipe while keeping
          its core masked language modeling approach.
        </p>
      </NoteBlock>
    </div>
  )
}
