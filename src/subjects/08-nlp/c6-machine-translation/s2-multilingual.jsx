import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function CrossLingualDemo() {
  const [task, setTask] = useState('zeroshot')
  const scenarios = {
    zeroshot: {
      title: 'Zero-Shot Cross-Lingual Transfer',
      train: 'Train on English NER data',
      eval: 'Evaluate on German NER (no German training data)',
      result: 'F1: ~70-80% (vs ~85% with German training data)',
    },
    translate_train: {
      title: 'Translate-Train',
      train: 'Machine-translate English training data to German, then train',
      eval: 'Evaluate on German NER',
      result: 'F1: ~75-82% (depends on translation quality)',
    },
    translate_test: {
      title: 'Translate-Test',
      train: 'Train on English NER data',
      eval: 'Translate German test data to English, then predict',
      result: 'F1: ~72-78% (error propagation from MT)',
    },
  }
  const s = scenarios[task]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Cross-Lingual Transfer Strategies</h3>
      <div className="flex flex-wrap gap-2 mb-4">
        {Object.keys(scenarios).map(k => (
          <button key={k} onClick={() => setTask(k)}
            className={`rounded px-3 py-1 text-xs transition ${k === task ? 'bg-violet-600 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {scenarios[k].title}
          </button>
        ))}
      </div>
      <div className="space-y-2 text-sm">
        <p className="text-gray-600 dark:text-gray-400"><strong>Training:</strong> {s.train}</p>
        <p className="text-gray-600 dark:text-gray-400"><strong>Evaluation:</strong> {s.eval}</p>
        <p className="text-violet-600 dark:text-violet-400 font-semibold">{s.result}</p>
      </div>
    </div>
  )
}

export default function MultilingualModels() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Multilingual models like mBERT and XLM-R learn shared representations across many
        languages from a single model. This enables cross-lingual transfer: training on data
        in one language and applying the model to another without additional training.
      </p>

      <DefinitionBlock title="Multilingual BERT (mBERT)">
        <p>
          mBERT is trained on Wikipedia text from 104 languages using the standard BERT
          masked language modeling objective. Despite no explicit cross-lingual signal, it
          learns aligned representations across languages through shared subword vocabulary
          and co-occurring multilingual patterns.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="XLM-R (Cross-Lingual Language Model - RoBERTa)">
        <p>
          XLM-R improves on mBERT by training on 2.5TB of CommonCrawl data in 100 languages
          with the RoBERTa recipe (more data, longer training, no NSP):
        </p>
        <BlockMath math="\mathcal{L}_{\text{XLM-R}} = -\sum_{i \in \mathcal{M}} \log P(x_i \mid \mathbf{x}_{\backslash \mathcal{M}})" />
        <p className="mt-2">
          Key improvement: SentencePiece tokenizer with 250K vocabulary shared across all languages,
          and exponential smoothing of language sampling to prevent high-resource language dominance.
        </p>
      </DefinitionBlock>

      <TheoremBlock title="Language Sampling Distribution" id="lang-sampling">
        <p>To balance high-resource and low-resource languages, XLM-R uses smoothed sampling:</p>
        <BlockMath math="p_l = \frac{n_l^\alpha}{\sum_{l'} n_{l'}^\alpha}" />
        <p className="mt-2">
          where <InlineMath math="n_l" /> is the number of sentences in language <InlineMath math="l" /> and
          <InlineMath math="\alpha = 0.7" />. This upsamples low-resource languages relative to their natural frequency.
        </p>
      </TheoremBlock>

      <CrossLingualDemo />

      <ExampleBlock title="Model Comparison">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>mBERT:</strong> 110M params, 104 languages, 110K vocab, Wikipedia</li>
          <li><strong>XLM-R Base:</strong> 270M params, 100 languages, 250K vocab, CC-100</li>
          <li><strong>XLM-R Large:</strong> 550M params, better on low-resource languages</li>
          <li><strong>NLLB:</strong> Translation-focused, 200 languages, encoder-decoder</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Cross-Lingual NER with XLM-R"
        code={`from transformers import (
    AutoModelForTokenClassification, AutoTokenizer, pipeline
)

# XLM-R fine-tuned on English CoNLL NER
model_name = "xlm-roberta-large-finetuned-conll03-english"
ner = pipeline("ner", model=model_name, aggregation_strategy="simple")

# Test on English (training language)
en_result = ner("Barack Obama visited Paris last Friday.")
print("English NER:")
for ent in en_result:
    print(f"  {ent['word']:20s} -> {ent['entity_group']}")

# Zero-shot transfer to German (no German training data!)
de_result = ner("Angela Merkel besuchte Berlin am Freitag.")
print("\\nGerman NER (zero-shot):")
for ent in de_result:
    print(f"  {ent['word']:20s} -> {ent['entity_group']}")

# Zero-shot transfer to Chinese
zh_result = ner("习近平访问了北京大学。")
print("\\nChinese NER (zero-shot):")
for ent in zh_result:
    print(f"  {ent['word']:20s} -> {ent['entity_group']}")`}
      />

      <WarningBlock title="The Curse of Multilinguality">
        <p>
          Adding more languages to a fixed-capacity model can degrade performance on each
          individual language (negative transfer). This is especially pronounced for
          typologically distant languages. XLM-R mitigates this through increased model
          capacity, but the tradeoff between language coverage and per-language quality remains.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="How Does Cross-Lingual Transfer Work?">
        <p>
          Multilingual models develop language-agnostic representations despite no explicit
          alignment objective. This likely arises from shared subwords across related languages,
          similar word order patterns, and anchor points in multilingual text. However, the
          representations are not perfectly aligned, and performance drops for distant language pairs.
        </p>
      </NoteBlock>
    </div>
  )
}
