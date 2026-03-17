import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ExtractiveQADemo() {
  const [qIdx, setQIdx] = useState(0)
  const context = 'The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World Fair. The tower is 330 metres tall and was the tallest man-made structure for 41 years.'

  const qas = [
    { question: 'Where is the Eiffel Tower located?', start: 62, end: 76, answer: 'Paris, France' },
    { question: 'When was it constructed?', start: 98, end: 114, answer: '1887 to 1889' },
    { question: 'How tall is the tower?', start: 172, end: 188, answer: '330 metres tall' },
  ]
  const qa = qas[qIdx]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Extractive QA Demo</h3>
      <div className="flex flex-wrap gap-2 mb-3">
        {qas.map((q, i) => (
          <button key={i} onClick={() => setQIdx(i)}
            className={`rounded px-3 py-1 text-xs transition ${i === qIdx ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            Q{i + 1}
          </button>
        ))}
      </div>
      <p className="text-sm text-violet-600 dark:text-violet-400 font-semibold mb-2">{qa.question}</p>
      <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed">
        {context.slice(0, qa.start)}
        <span className="bg-violet-200 dark:bg-violet-800/50 rounded px-0.5 font-semibold">{context.slice(qa.start, qa.end)}</span>
        {context.slice(qa.end)}
      </p>
      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">Answer span: "{qa.answer}"</p>
    </div>
  )
}

export default function QuestionAnswering() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Question Answering (QA) systems find answers to natural language questions. Extractive QA
        selects a span from a given context passage, while generative QA produces the answer
        as free-form text using encoder-decoder or decoder-only models.
      </p>

      <DefinitionBlock title="Extractive QA">
        <p>
          Given context <InlineMath math="C = (c_1, \ldots, c_n)" /> and question <InlineMath math="Q" />,
          predict start and end positions of the answer span:
        </p>
        <BlockMath math="P_{\text{start}}(i) = \frac{\exp(\mathbf{w}_s^\top \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_s^\top \mathbf{h}_j)}, \quad P_{\text{end}}(i) = \frac{\exp(\mathbf{w}_e^\top \mathbf{h}_i)}{\sum_j \exp(\mathbf{w}_e^\top \mathbf{h}_j)}" />
        <BlockMath math="\hat{a} = \arg\max_{i \leq j} P_{\text{start}}(i) \cdot P_{\text{end}}(j)" />
      </DefinitionBlock>

      <ExtractiveQADemo />

      <DefinitionBlock title="Generative QA">
        <p>Generative QA formulates answering as conditional text generation:</p>
        <BlockMath math="P(a \mid Q, C) = \prod_{t=1}^{|a|} P(a_t \mid a_{<t}, Q, C; \theta)" />
        <p className="mt-2">
          This handles questions requiring synthesis, multi-hop reasoning, or answers not
          present as exact spans in the context.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Retrieval-Augmented Generation (RAG)" id="rag">
        <p>RAG combines a retriever and generator for open-domain QA:</p>
        <BlockMath math="P(a \mid Q) = \sum_{d \in \text{top-}k} P_{\text{ret}}(d \mid Q) \cdot P_{\text{gen}}(a \mid Q, d)" />
        <p className="mt-2">
          The retriever fetches relevant documents, and the generator produces the answer
          conditioned on both the question and retrieved context.
        </p>
      </TheoremBlock>

      <ExampleBlock title="QA Benchmarks">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>SQuAD 2.0:</strong> Extractive QA with unanswerable questions</li>
          <li><strong>Natural Questions:</strong> Real Google search queries with Wikipedia answers</li>
          <li><strong>TriviaQA:</strong> Trivia questions with multiple evidence documents</li>
          <li><strong>HotpotQA:</strong> Multi-hop reasoning across multiple paragraphs</li>
        </ul>
        <p className="mt-2">Metrics: Exact Match (EM) and token-level F1 score.</p>
      </ExampleBlock>

      <PythonCode
        title="Extractive and Generative QA"
        code={`from transformers import pipeline

# Extractive QA with BERT
extractive_qa = pipeline("question-answering",
    model="deepset/roberta-base-squad2")

context = """
The Amazon rainforest produces approximately 20% of the world's
oxygen. It spans across nine countries in South America, with
60% of the forest located in Brazil.
"""
result = extractive_qa(
    question="How much oxygen does the Amazon produce?",
    context=context
)
print(f"Answer: {result['answer']}")
print(f"Score: {result['score']:.4f}")
print(f"Span: [{result['start']}:{result['end']}]")

# Generative QA with T5
gen_qa = pipeline("text2text-generation", model="google/flan-t5-base")
answer = gen_qa(
    "Answer the question based on the context. "
    f"Context: {context} "
    "Question: Why is the Amazon rainforest important?"
)
print(f"Generated answer: {answer[0]['generated_text']}")`}
      />

      <NoteBlock type="note" title="From Extractive to Generative">
        <p>
          Extractive QA is simpler and more interpretable (the answer is always a verbatim span),
          but it cannot handle abstractive answers or multi-hop reasoning. Modern systems
          increasingly use generative approaches, especially with large language models that can
          synthesize information from multiple passages and provide well-formed answers.
        </p>
      </NoteBlock>
    </div>
  )
}
