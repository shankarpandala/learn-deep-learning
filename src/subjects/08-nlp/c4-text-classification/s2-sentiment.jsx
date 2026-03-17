import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import ExerciseBlock from '../../../components/content/ExerciseBlock.jsx'

function SentimentDemo() {
  const [text, setText] = useState('The movie was absolutely brilliant and I loved every moment of it!')
  const positiveWords = ['brilliant', 'loved', 'great', 'wonderful', 'excellent', 'amazing', 'fantastic', 'good', 'best']
  const negativeWords = ['terrible', 'awful', 'worst', 'boring', 'bad', 'horrible', 'disappointing', 'poor', 'hate']

  const words = text.toLowerCase().split(/\s+/)
  const posCount = words.filter(w => positiveWords.some(p => w.includes(p))).length
  const negCount = words.filter(w => negativeWords.some(n => w.includes(n))).length
  const score = posCount - negCount
  const sentiment = score > 0 ? 'Positive' : score < 0 ? 'Negative' : 'Neutral'
  const color = score > 0 ? 'text-green-600' : score < 0 ? 'text-red-600' : 'text-gray-600'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Simple Lexicon-Based Sentiment</h3>
      <textarea value={text} onChange={e => setText(e.target.value)} rows={2}
        className="w-full rounded border p-2 text-sm dark:bg-gray-800 dark:border-gray-600 dark:text-gray-300 mb-3" />
      <div className="flex items-center gap-4 text-sm">
        <span className="text-gray-600 dark:text-gray-400">Positive words: <strong className="text-green-600">{posCount}</strong></span>
        <span className="text-gray-600 dark:text-gray-400">Negative words: <strong className="text-red-600">{negCount}</strong></span>
        <span className={`font-bold ${color}`}>{sentiment}</span>
      </div>
      <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">This naive approach fails on negation, sarcasm, and nuance. Modern models handle these through contextual understanding.</p>
    </div>
  )
}

export default function SentimentAnalysis() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Sentiment analysis classifies text by expressed opinion or emotion. It ranges from
        simple binary positive/negative classification to fine-grained aspect-based sentiment
        analysis that identifies opinions about specific entities or features.
      </p>

      <DefinitionBlock title="Sentiment Classification">
        <p>Given text <InlineMath math="\mathbf{x}" />, predict a sentiment label:</p>
        <BlockMath math="\hat{y} = \arg\max_{c \in \mathcal{C}} P(c \mid \mathbf{x}; \theta)" />
        <p className="mt-2">
          Labels can be binary (positive/negative), ternary (adding neutral), or fine-grained
          (1-5 star ratings mapped to ordinal classes).
        </p>
      </DefinitionBlock>

      <SentimentDemo />

      <DefinitionBlock title="Aspect-Based Sentiment Analysis (ABSA)">
        <p>ABSA identifies the sentiment toward specific aspects or features within a text:</p>
        <BlockMath math="P(s_a \mid \mathbf{x}, a) \text{ for each aspect } a \in \mathcal{A}" />
        <p className="mt-2">
          Example: "The food was excellent but the service was terrible" has positive
          sentiment for food and negative for service.
        </p>
      </DefinitionBlock>

      <ExampleBlock title="Challenges in Sentiment Analysis">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Negation:</strong> "not bad" is actually positive</li>
          <li><strong>Sarcasm:</strong> "Oh great, another meeting" is negative</li>
          <li><strong>Implicit sentiment:</strong> "The battery lasted 2 hours" (negative, no sentiment words)</li>
          <li><strong>Domain shift:</strong> "sick beats" is positive in music reviews</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Sentiment Analysis with Transformers"
        code={`from transformers import pipeline
import torch

# Pre-trained sentiment pipeline
sentiment = pipeline("sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english")

texts = [
    "I absolutely loved this movie!",
    "The plot was confusing and boring.",
    "It was okay, nothing special.",
]
results = sentiment(texts)
for text, result in zip(texts, results):
    print(f"{result['label']} ({result['score']:.3f}): {text}")

# Aspect-Based Sentiment with a classifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("yangheng/deberta-v3-base-absa-v1.1")
model = AutoModelForSequenceClassification.from_pretrained(
    "yangheng/deberta-v3-base-absa-v1.1"
)

text = "Great food but terrible service."
for aspect in ["food", "service"]:
    inputs = tokenizer(f"[CLS] {text} [SEP] {aspect} [SEP]",
                       return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    pred = ["negative", "neutral", "positive"][logits.argmax().item()]
    print(f"Aspect '{aspect}': {pred}")`}
      />

      <ExerciseBlock title="Exercise: Sentiment on Negated Sentences">
        <p>
          Consider these sentences and predict what a simple lexicon-based method vs a
          Transformer model would classify:
        </p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>"This movie is not bad at all"</li>
          <li>"I wouldn't say this restaurant is good"</li>
          <li>"It's hard to find a worse experience"</li>
        </ul>
        <p className="mt-2 text-sm text-gray-500">Why do contextual models handle these better?</p>
      </ExerciseBlock>

      <NoteBlock type="note" title="Beyond Binary Sentiment">
        <p>
          Modern sentiment systems go beyond polarity to detect emotions (joy, anger, fear),
          stance (agree, disagree), and toxicity. Multitask learning across these related
          tasks often improves overall performance. Models like RoBERTa fine-tuned on
          multi-domain data achieve near-human accuracy on standard benchmarks.
        </p>
      </NoteBlock>
    </div>
  )
}
