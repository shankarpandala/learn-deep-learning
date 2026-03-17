import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExerciseBlock from '../../../components/content/ExerciseBlock.jsx'

function RelationViewer() {
  const [sentIdx, setSentIdx] = useState(0)
  const examples = [
    { text: 'Albert Einstein was born in Ulm, Germany in 1879.', entities: ['Albert Einstein', 'Ulm', 'Germany', '1879'], relations: [{ subj: 0, obj: 1, rel: 'born_in_city' }, { subj: 0, obj: 2, rel: 'nationality' }, { subj: 0, obj: 3, rel: 'birth_year' }] },
    { text: 'Tim Cook is the CEO of Apple Inc, headquartered in Cupertino.', entities: ['Tim Cook', 'Apple Inc', 'Cupertino'], relations: [{ subj: 0, obj: 1, rel: 'CEO_of' }, { subj: 1, obj: 2, rel: 'headquartered_in' }] },
  ]
  const ex = examples[sentIdx]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Relation Extraction Viewer</h3>
      <div className="flex gap-2 mb-4">
        {examples.map((_, i) => (
          <button key={i} onClick={() => setSentIdx(i)}
            className={`rounded px-3 py-1 text-sm transition ${i === sentIdx ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            Example {i + 1}
          </button>
        ))}
      </div>
      <p className="text-sm text-gray-700 dark:text-gray-300 mb-3 italic">"{ex.text}"</p>
      <div className="space-y-2">
        {ex.relations.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="rounded bg-violet-100 px-2 py-0.5 font-semibold text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">{ex.entities[r.subj]}</span>
            <span className="text-gray-400">--</span>
            <span className="rounded bg-gray-100 px-2 py-0.5 font-mono text-gray-600 dark:bg-gray-800 dark:text-gray-400">{r.rel}</span>
            <span className="text-gray-400">--&gt;</span>
            <span className="rounded bg-blue-100 px-2 py-0.5 font-semibold text-blue-700 dark:bg-blue-900/30 dark:text-blue-300">{ex.entities[r.obj]}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function RelationExtraction() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Relation extraction identifies semantic relationships between entities in text, producing
        structured knowledge triples of the form (subject, relation, object). It is essential
        for building knowledge graphs and information extraction pipelines.
      </p>

      <DefinitionBlock title="Relation Classification">
        <p>
          Given a sentence <InlineMath math="\mathbf{x}" /> with marked entities <InlineMath math="e_1" /> and <InlineMath math="e_2" />,
          classify the relation <InlineMath math="r" /> between them:
        </p>
        <BlockMath math="P(r \mid \mathbf{x}, e_1, e_2) = \text{softmax}(\mathbf{W}[\mathbf{h}_{e_1}; \mathbf{h}_{e_2}; \mathbf{h}_{e_1} \odot \mathbf{h}_{e_2}])" />
        <p className="mt-2">
          where <InlineMath math="\mathbf{h}_{e_1}, \mathbf{h}_{e_2}" /> are the entity representations from
          a pretrained encoder. The set of relations includes a special "no_relation" class.
        </p>
      </DefinitionBlock>

      <RelationViewer />

      <TheoremBlock title="Distant Supervision" id="distant-supervision">
        <p>
          Distant supervision automatically generates training data by aligning a knowledge base
          with text. If a KB contains triple <InlineMath math="(e_1, r, e_2)" />, then any sentence
          mentioning both <InlineMath math="e_1" /> and <InlineMath math="e_2" /> is assumed to express <InlineMath math="r" />:
        </p>
        <BlockMath math="\{(x_i, r) : e_1 \in x_i \wedge e_2 \in x_i \wedge (e_1, r, e_2) \in \text{KB}\}" />
        <p className="mt-2">This assumption is noisy, requiring multi-instance learning or denoising strategies.</p>
      </TheoremBlock>

      <ExampleBlock title="Common Relation Types (TACRED)">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>per:born_in</strong> - Person born in a location</li>
          <li><strong>org:founded_by</strong> - Organization founded by person</li>
          <li><strong>per:employee_of</strong> - Person works for organization</li>
          <li><strong>org:headquarters</strong> - Organization located in city</li>
        </ul>
        <p className="mt-2">TACRED has 42 relation types plus "no_relation", with 106K examples.</p>
      </ExampleBlock>

      <PythonCode
        title="Relation Extraction with Transformers"
        code={`from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Using a BERT model fine-tuned for relation classification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Mark entity spans with special tokens
sentence = (
    "The [E1] Albert Einstein [/E1] was born in "
    "[E2] Ulm [/E2], Germany."
)

# Add special tokens to tokenizer
special_tokens = {"additional_special_tokens": [
    "[E1]", "[/E1]", "[E2]", "[/E2]"
]}
tokenizer.add_special_tokens(special_tokens)

inputs = tokenizer(sentence, return_tensors="pt")
print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")

# In a full pipeline, you would:
# 1. Run NER to find entities
# 2. For each entity pair, create a relation classification input
# 3. Classify the relation type
# 4. Filter out "no_relation" predictions

# Example output structure
relations = [
    ("Albert Einstein", "born_in_city", "Ulm"),
    ("Albert Einstein", "nationality", "Germany"),
]
for subj, rel, obj in relations:
    print(f"  ({subj}, {rel}, {obj})")`}
      />

      <ExerciseBlock title="Exercise: Joint vs Pipeline Approach">
        <p>
          Explain the tradeoff between pipeline NER + RE (first extract entities, then classify
          relations) versus joint entity and relation extraction. Consider error propagation,
          computational cost, and model complexity.
        </p>
      </ExerciseBlock>

      <NoteBlock type="note" title="Modern Approaches">
        <p>
          Recent work uses generative approaches where a large language model outputs structured
          triples directly from text. Document-level relation extraction handles relations spanning
          multiple sentences, and few-shot RE uses prompting to identify novel relation types
          without extensive labeled data.
        </p>
      </NoteBlock>
    </div>
  )
}
