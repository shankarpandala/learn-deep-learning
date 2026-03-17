import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function BIOTagger() {
  const [sentIdx, setSentIdx] = useState(0)
  const examples = [
    {
      tokens: ['Barack', 'Obama', 'visited', 'Paris', 'last', 'Friday'],
      tags: ['B-PER', 'I-PER', 'O', 'B-LOC', 'O', 'B-DATE'],
    },
    {
      tokens: ['Apple', 'Inc', 'released', 'the', 'iPhone', 'in', 'San', 'Francisco'],
      tags: ['B-ORG', 'I-ORG', 'O', 'O', 'B-PROD', 'O', 'B-LOC', 'I-LOC'],
    },
  ]
  const ex = examples[sentIdx]

  const tagColor = (tag) => {
    if (tag.includes('PER')) return 'bg-violet-100 text-violet-700 dark:bg-violet-900/30 dark:text-violet-300'
    if (tag.includes('LOC')) return 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300'
    if (tag.includes('ORG')) return 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300'
    if (tag.includes('DATE') || tag.includes('PROD')) return 'bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-300'
    return 'bg-gray-100 text-gray-500 dark:bg-gray-800 dark:text-gray-400'
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">BIO Tagging Visualizer</h3>
      <div className="flex gap-2 mb-4">
        {examples.map((_, i) => (
          <button key={i} onClick={() => setSentIdx(i)}
            className={`rounded px-3 py-1 text-sm transition ${i === sentIdx ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            Sentence {i + 1}
          </button>
        ))}
      </div>
      <div className="flex flex-wrap gap-2">
        {ex.tokens.map((tok, i) => (
          <div key={i} className="text-center">
            <div className="text-sm font-mono text-gray-700 dark:text-gray-300 mb-1">{tok}</div>
            <div className={`rounded px-2 py-0.5 text-xs font-mono ${tagColor(ex.tags[i])}`}>{ex.tags[i]}</div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function NER() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Named Entity Recognition (NER) identifies and classifies named entities (persons,
        organizations, locations, etc.) in text. It is formulated as a token classification
        task where each token receives a label from a BIO or BIOES tagging scheme.
      </p>

      <DefinitionBlock title="BIO Tagging Scheme">
        <p>Each token is labeled with one of:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>B-TYPE:</strong> Beginning of an entity of the given type</li>
          <li><strong>I-TYPE:</strong> Inside (continuation) of an entity</li>
          <li><strong>O:</strong> Outside any entity</li>
        </ul>
        <p className="mt-2">
          The BIOES variant adds <strong>E-TYPE</strong> (end of entity) and <strong>S-TYPE</strong> (single-token entity)
          for finer boundary detection.
        </p>
      </DefinitionBlock>

      <BIOTagger />

      <TheoremBlock title="Token Classification with CRF" id="ner-crf">
        <p>A CRF layer on top of a neural encoder models label dependencies via transition scores:</p>
        <BlockMath math="P(\mathbf{y} \mid \mathbf{x}) = \frac{\exp\left(\sum_{t=1}^{T} \phi(y_t, \mathbf{x}, t) + \psi(y_{t-1}, y_t)\right)}{\sum_{\mathbf{y}'} \exp\left(\sum_{t=1}^{T} \phi(y'_t, \mathbf{x}, t) + \psi(y'_{t-1}, y'_t)\right)}" />
        <p className="mt-2">
          where <InlineMath math="\phi" /> are emission scores from the encoder and <InlineMath math="\psi" /> are
          learned transition scores (e.g., I-PER cannot follow B-LOC).
        </p>
      </TheoremBlock>

      <ExampleBlock title="Common Entity Types (CoNLL-2003)">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>PER:</strong> Person names (Barack Obama, Marie Curie)</li>
          <li><strong>ORG:</strong> Organizations (Apple Inc, United Nations)</li>
          <li><strong>LOC:</strong> Locations (Paris, Mount Everest)</li>
          <li><strong>MISC:</strong> Miscellaneous (English, Nobel Prize)</li>
        </ul>
        <p className="mt-2">Evaluation uses span-level F1: a predicted entity must match both the span boundaries and the type.</p>
      </ExampleBlock>

      <PythonCode
        title="NER with Hugging Face Transformers"
        code={`from transformers import pipeline, AutoModelForTokenClassification
from transformers import AutoTokenizer

# Quick NER with pipeline
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
               aggregation_strategy="simple")

text = "Barack Obama visited the United Nations in New York."
entities = ner(text)
for ent in entities:
    print(f"  {ent['word']:20s} {ent['entity_group']:6s} "
          f"({ent['score']:.3f})")

# Manual token classification
tokenizer = AutoTokenizer.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english")
model = AutoModelForTokenClassification.from_pretrained(
    "dbmdz/bert-large-cased-finetuned-conll03-english")

import torch
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predictions = torch.argmax(logits, dim=-1)[0]
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
for tok, pred in zip(tokens, predictions):
    label = model.config.id2label[pred.item()]
    if label != "O":
        print(f"  {tok}: {label}")`}
      />

      <NoteBlock type="note" title="Modern NER Approaches">
        <p>
          While token classification with BIO tags remains the dominant approach, recent work
          explores span-based methods (predicting entity spans directly), generative NER (using
          seq2seq models to output entities as text), and few-shot NER via prompting. Nested NER,
          where entities can overlap, requires specialized architectures beyond flat BIO tagging.
        </p>
      </NoteBlock>
    </div>
  )
}
