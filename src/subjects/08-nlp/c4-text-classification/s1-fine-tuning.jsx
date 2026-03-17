import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function LearningRateSchedule() {
  const [warmup, setWarmup] = useState(10)
  const totalSteps = 100
  const W = 400, H = 180, pad = 40
  const peakLR = 5e-5

  const lr = (step) => {
    if (step < warmup) return peakLR * (step / warmup)
    return peakLR * (1 - (step - warmup) / (totalSteps - warmup))
  }

  const points = Array.from({ length: totalSteps }, (_, i) => i)
  const path = points.map((s, i) => {
    const x = pad + (s / totalSteps) * (W - 2 * pad)
    const y = H - pad - (lr(s) / peakLR) * (H - 2 * pad)
    return `${i === 0 ? 'M' : 'L'}${x},${y}`
  }).join(' ')

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Fine-tuning Learning Rate Schedule</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Warmup steps: {warmup}
        <input type="range" min={0} max={30} value={warmup}
          onChange={e => setWarmup(parseInt(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <svg width={W} height={H} className="mx-auto block">
        <line x1={pad} y1={H - pad} x2={W - pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        <line x1={pad} y1={pad} x2={pad} y2={H - pad} stroke="#d1d5db" strokeWidth={0.5} />
        <path d={path} fill="none" stroke="#8b5cf6" strokeWidth={2.5} />
        <text x={W / 2} y={H - 5} textAnchor="middle" className="fill-gray-400 text-[10px]">Training steps</text>
        <text x={10} y={H / 2} textAnchor="middle" transform={`rotate(-90, 10, ${H / 2})`} className="fill-gray-400 text-[10px]">LR</text>
      </svg>
    </div>
  )
}

export default function FineTuning() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Fine-tuning adapts a pretrained language model to a specific downstream task by
        continuing training on task-specific labeled data. This transfer learning approach
        requires far less data than training from scratch and achieves superior performance.
      </p>

      <DefinitionBlock title="Fine-tuning for Classification">
        <p>
          A classification head is added on top of the pretrained encoder. For BERT, the [CLS]
          token representation is projected to the label space:
        </p>
        <BlockMath math="P(y \mid \mathbf{x}) = \text{softmax}(\mathbf{W} \cdot \mathbf{h}_{\text{[CLS]}} + \mathbf{b})" />
        <p className="mt-2">All parameters (including the pretrained ones) are updated with a small learning rate.</p>
      </DefinitionBlock>

      <LearningRateSchedule />

      <TheoremBlock title="Discriminative Fine-tuning" id="discriminative-ft">
        <p>Different layers may need different learning rates. ULMFiT proposed:</p>
        <BlockMath math="\eta^l = \eta^L \cdot \xi^{L-l}" />
        <p className="mt-2">
          where <InlineMath math="\eta^L" /> is the base learning rate for the last layer,
          <InlineMath math="l" /> is the layer index, and <InlineMath math="\xi < 1" /> is the decay factor.
          Lower layers (closer to input) get smaller learning rates to preserve general features.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Recommended Hyperparameters">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Learning rate:</strong> 2e-5 to 5e-5 (much smaller than pretraining)</li>
          <li><strong>Batch size:</strong> 16 or 32</li>
          <li><strong>Epochs:</strong> 2-4 (overfitting risk with more)</li>
          <li><strong>Warmup:</strong> 6-10% of total training steps</li>
          <li><strong>Weight decay:</strong> 0.01</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Fine-tuning BERT for Text Classification"
        code={`from transformers import (
    BertForSequenceClassification, BertTokenizer,
    Trainer, TrainingArguments
)
from datasets import load_dataset

# Load dataset and tokenizer
dataset = load_dataset("imdb")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize(examples):
    return tokenizer(examples["text"], truncation=True,
                     padding="max_length", max_length=256)

tokenized = dataset.map(tokenize, batched=True)

# Load model with classification head
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=2
)

# Training arguments with warmup + linear decay
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

trainer = Trainer(model=model, args=args,
                  train_dataset=tokenized["train"],
                  eval_dataset=tokenized["test"])
trainer.train()`}
      />

      <WarningBlock title="Catastrophic Forgetting">
        <p>
          Aggressive fine-tuning can cause the model to forget useful pretrained knowledge.
          Mitigation strategies include: low learning rates, gradual unfreezing (training only
          the head first, then progressively unfreezing lower layers), and regularization
          toward pretrained weights.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Parameter-Efficient Fine-tuning">
        <p>
          Full fine-tuning updates all model parameters, which is expensive for large models.
          Alternatives like LoRA (Low-Rank Adaptation), adapters, and prefix tuning update
          only a small fraction of parameters (0.1-1%) while achieving comparable performance.
        </p>
      </NoteBlock>
    </div>
  )
}
