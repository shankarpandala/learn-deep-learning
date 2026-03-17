import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function BPEDemo() {
  const [step, setStep] = useState(0)
  const steps = [
    { vocab: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', '_'], corpus: 'l o w _ (5), l o w e r _ (2), n e w e s t _ (6), w i d e s t _ (3)', merge: 'Start: character-level vocabulary' },
    { vocab: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', '_', 'es'], corpus: 'l o w _ (5), l o w e r _ (2), n e w es t _ (6), w i d es t _ (3)', merge: 'Merge: e + s -> es (frequency: 9)' },
    { vocab: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', '_', 'es', 'est'], corpus: 'l o w _ (5), l o w e r _ (2), n e w est _ (6), w i d est _ (3)', merge: 'Merge: es + t -> est (frequency: 9)' },
    { vocab: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', '_', 'es', 'est', 'lo'], corpus: 'lo w _ (5), lo w e r _ (2), n e w est _ (6), w i d est _ (3)', merge: 'Merge: l + o -> lo (frequency: 7)' },
    { vocab: ['l', 'o', 'w', 'e', 'r', 'n', 's', 't', 'i', 'd', '_', 'es', 'est', 'lo', 'low'], corpus: 'low _ (5), low e r _ (2), n e w est _ (6), w i d est _ (3)', merge: 'Merge: lo + w -> low (frequency: 7)' },
  ]
  const s = steps[step]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">BPE Merge Visualization</h3>
      <div className="flex items-center gap-3 mb-3">
        <button onClick={() => setStep(Math.max(0, step - 1))} disabled={step === 0}
          className="rounded px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-40 dark:bg-gray-800 dark:hover:bg-gray-700">Prev</button>
        <span className="text-sm text-gray-600 dark:text-gray-400">Step {step}/{steps.length - 1}</span>
        <button onClick={() => setStep(Math.min(steps.length - 1, step + 1))} disabled={step === steps.length - 1}
          className="rounded px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 disabled:opacity-40 dark:bg-gray-800 dark:hover:bg-gray-700">Next</button>
      </div>
      <p className="text-xs font-semibold text-violet-600 dark:text-violet-400 mb-2">{s.merge}</p>
      <p className="text-xs text-gray-600 dark:text-gray-400 mb-1"><strong>Corpus:</strong> {s.corpus}</p>
      <div className="flex flex-wrap gap-1 mt-2">
        {s.vocab.map((v, i) => (
          <span key={i} className="rounded bg-violet-100 px-1.5 py-0.5 text-xs font-mono text-violet-700 dark:bg-violet-900/30 dark:text-violet-300">{v}</span>
        ))}
      </div>
    </div>
  )
}

export default function Tokenization() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Modern NLP models rely on subword tokenization to balance vocabulary size with the ability
        to represent any text. BPE, WordPiece, and SentencePiece are the three dominant approaches
        used by GPT, BERT, and T5 respectively.
      </p>

      <DefinitionBlock title="Byte Pair Encoding (BPE)">
        <p>
          BPE starts with a character-level vocabulary and iteratively merges the most frequent
          adjacent pair of tokens. After <InlineMath math="k" /> merges, the vocabulary
          has <InlineMath math="|V_{\text{base}}| + k" /> entries. Used by GPT-2/3/4.
        </p>
      </DefinitionBlock>

      <BPEDemo />

      <DefinitionBlock title="WordPiece Tokenization">
        <p>
          WordPiece (used by BERT) is similar to BPE but selects merges that maximize the
          likelihood of the training corpus rather than raw frequency:
        </p>
        <BlockMath math="\text{score}(x, y) = \frac{\text{freq}(xy)}{\text{freq}(x) \cdot \text{freq}(y)}" />
        <p className="mt-2">Subword tokens that are not word-initial are prefixed with <code>##</code> (e.g., "playing" becomes ["play", "##ing"]).</p>
      </DefinitionBlock>

      <DefinitionBlock title="SentencePiece / Unigram Model">
        <p>
          SentencePiece (used by T5, XLNet) treats tokenization as a probabilistic model. The
          unigram model starts with a large vocabulary and iteratively removes tokens whose
          removal least increases the corpus loss:
        </p>
        <BlockMath math="P(\mathbf{x}) = \prod_{i=1}^{M} P(x_i), \quad \text{where } \sum_{x \in V} P(x) = 1" />
      </DefinitionBlock>

      <ExampleBlock title="Tokenization Comparison">
        <p>Input: "unhappiness"</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>BPE (GPT-2):</strong> ["un", "h", "app", "iness"]</li>
          <li><strong>WordPiece (BERT):</strong> ["un", "##ha", "##pp", "##iness"]</li>
          <li><strong>SentencePiece (T5):</strong> ["_un", "happiness"]</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Tokenization with Hugging Face"
        code={`from transformers import AutoTokenizer

# BPE tokenizer (GPT-2)
gpt2_tok = AutoTokenizer.from_pretrained("gpt2")
tokens = gpt2_tok.tokenize("unhappiness is temporary")
print(f"GPT-2 BPE: {tokens}")

# WordPiece tokenizer (BERT)
bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = bert_tok.tokenize("unhappiness is temporary")
print(f"BERT WordPiece: {tokens}")

# SentencePiece tokenizer (T5)
t5_tok = AutoTokenizer.from_pretrained("t5-small")
tokens = t5_tok.tokenize("unhappiness is temporary")
print(f"T5 SentencePiece: {tokens}")

# Token IDs and decoding
ids = gpt2_tok.encode("Hello, world!")
print(f"Token IDs: {ids}")
print(f"Decoded: {gpt2_tok.decode(ids)}")`}
      />

      <WarningBlock title="Tokenization Affects Everything">
        <p>
          Tokenization choices deeply impact model behavior. Arithmetic is hard for LLMs partly
          because numbers are split unpredictably (e.g., "12345" might become ["123", "45"]).
          Perplexity scores are not comparable across different tokenizers. Always ensure your
          tokenizer matches the pretrained model.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Byte-Level BPE">
        <p>
          GPT-2 introduced byte-level BPE, which operates on raw bytes instead of Unicode
          characters. This guarantees any text can be tokenized without unknown tokens, at the
          cost of longer sequences for non-ASCII text. This approach has become the standard
          for most modern large language models.
        </p>
      </NoteBlock>
    </div>
  )
}
