import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function PromptBuilder() {
  const [numShots, setNumShots] = useState(2)
  const [testInput, setTestInput] = useState('This restaurant has amazing pasta.')

  const examples = [
    { text: 'The food was delicious and service was great.', label: 'positive' },
    { text: 'Terrible experience, cold food and rude staff.', label: 'negative' },
    { text: 'Average meal, nothing memorable.', label: 'neutral' },
  ]

  const prompt = [
    'Classify the sentiment of the following review.\n',
    ...examples.slice(0, numShots).map(e => `Review: "${e.text}"\nSentiment: ${e.label}\n`),
    `Review: "${testInput}"\nSentiment:`
  ].join('\n')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Few-Shot Prompt Builder</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="text-sm text-gray-600 dark:text-gray-400">
          Shots: {numShots}
          <input type="range" min={0} max={3} value={numShots}
            onChange={e => setNumShots(parseInt(e.target.value))} className="ml-2 w-24 accent-violet-500" />
        </label>
        <span className="text-xs text-violet-600 dark:text-violet-400 font-semibold">
          {numShots === 0 ? 'Zero-shot' : `${numShots}-shot`}
        </span>
      </div>
      <input type="text" value={testInput} onChange={e => setTestInput(e.target.value)}
        className="w-full rounded border px-2 py-1 text-sm mb-3 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-300"
        placeholder="Enter test review..." />
      <pre className="rounded bg-gray-50 p-3 text-xs overflow-x-auto dark:bg-gray-800 dark:text-gray-300 whitespace-pre-wrap">{prompt}</pre>
    </div>
  )
}

export default function FewShotClassification() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Few-shot and zero-shot classification leverage the knowledge stored in large pretrained
        models to classify text with minimal or no labeled examples. Instead of fine-tuning,
        the model is guided through carefully designed prompts.
      </p>

      <DefinitionBlock title="In-Context Learning">
        <p>A language model performs classification by conditioning on demonstrations in the prompt:</p>
        <BlockMath math="P(y \mid x) = P_{\text{LM}}(y \mid [d_1, d_2, \ldots, d_k, x])" />
        <p className="mt-2">
          where <InlineMath math="d_i = (x_i, y_i)" /> are demonstration examples and
          <InlineMath math="k" /> is the number of shots. When <InlineMath math="k = 0" />, this
          is zero-shot classification.
        </p>
      </DefinitionBlock>

      <PromptBuilder />

      <TheoremBlock title="Calibration for Few-Shot" id="calibration">
        <p>
          Language models have prior biases toward certain labels. Calibration adjusts for this
          by estimating the content-free prior <InlineMath math="\hat{P}(y)" /> using a null input:
        </p>
        <BlockMath math="P_{\text{calibrated}}(y \mid x) = \frac{P(y \mid x) / \hat{P}(y)}{\sum_{y'} P(y' \mid x) / \hat{P}(y')}" />
        <p className="mt-2">The null input (e.g., "N/A") reveals the model's label bias without content signal.</p>
      </TheoremBlock>

      <ExampleBlock title="Zero-Shot via Natural Language Inference">
        <p>Zero-shot classification can reuse NLI models by framing labels as hypotheses:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>Premise:</strong> "The new iPhone has a stunning camera."</li>
          <li><strong>Hypothesis:</strong> "This text is about technology."</li>
          <li>If the NLI model predicts <em>entailment</em>, the label "technology" applies.</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Zero-Shot and Few-Shot Classification"
        code={`from transformers import pipeline

# Zero-shot classification via NLI
classifier = pipeline("zero-shot-classification",
    model="facebook/bart-large-mnli")

text = "The stock market crashed after the announcement."
labels = ["politics", "finance", "sports", "technology"]
result = classifier(text, candidate_labels=labels)
for label, score in zip(result["labels"], result["scores"]):
    print(f"  {label}: {score:.3f}")

# Few-shot with a generative model
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
model = AutoModelForCausalLM.from_pretrained("gpt2-large")

prompt = """Classify the sentiment:
Text: "I love this product!" -> positive
Text: "Worst purchase ever." -> negative
Text: "The quality exceeded expectations." ->"""

inputs = tokenizer(prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=3)
print(tokenizer.decode(output[0], skip_special_tokens=True))`}
      />

      <WarningBlock title="Sensitivity to Prompt Design">
        <p>
          Few-shot performance is highly sensitive to prompt formatting, example order, and
          label verbalizers. Small changes like using "positive" vs "good" as label words, or
          reordering demonstrations, can cause accuracy swings of 10-30%. Prompt engineering
          and selection strategies are essential for reliable results.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="When to Use Few-Shot vs Fine-Tuning">
        <p>
          <strong>Few-shot prompting</strong> excels when labeled data is extremely scarce (fewer than
          ~100 examples) or when rapid prototyping is needed. <strong>Fine-tuning</strong> typically
          outperforms few-shot when hundreds or thousands of labeled examples are available. The
          crossover point depends on model size: larger models need fewer examples for effective prompting.
        </p>
      </NoteBlock>
    </div>
  )
}
