import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TaskTypeDiagram() {
  const [task, setTask] = useState('many-to-one')
  const W = 400, H = 140
  const configs = {
    'many-to-one': { inputs: [0, 1, 2, 3], outputs: [3], label: 'Sequence Classification' },
    'one-to-many': { inputs: [0], outputs: [0, 1, 2, 3], label: 'Sequence Generation' },
    'many-to-many': { inputs: [0, 1, 2, 3], outputs: [0, 1, 2, 3], label: 'Language Modeling' },
  }
  const cfg = configs[task]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">{cfg.label}</h3>
      <div className="flex items-center gap-3 mb-3 flex-wrap">
        {Object.keys(configs).map(k => (
          <button key={k} onClick={() => setTask(k)}
            className={`px-3 py-1 rounded-lg text-sm ${task === k ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {configs[k].label}
          </button>
        ))}
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {[0, 1, 2, 3].map(i => {
          const cx = 60 + i * 90
          return (
            <g key={i}>
              <rect x={cx - 22} y={45} width={44} height={34} rx={5} fill="#8b5cf6" opacity={0.8} />
              <text x={cx} y={67} textAnchor="middle" fill="white" fontSize={11} fontWeight="bold">h{i}</text>
              {cfg.inputs.includes(i) && (
                <>
                  <text x={cx} y={130} textAnchor="middle" fill="#6b7280" fontSize={10}>x{i}</text>
                  <line x1={cx} y1={118} x2={cx} y2={79} stroke="#a78bfa" strokeWidth={1.5} />
                </>
              )}
              {cfg.outputs.includes(i) && (
                <>
                  <text x={cx} y={22} textAnchor="middle" fill="#6b7280" fontSize={10}>y{i}</text>
                  <line x1={cx} y1={45} x2={cx} y2={28} stroke="#a78bfa" strokeWidth={1.5} />
                </>
              )}
              {i < 3 && <line x1={cx + 22} y1={62} x2={cx + 68} y2={62} stroke="#c4b5fd" strokeWidth={1.5} />}
            </g>
          )
        })}
      </svg>
    </div>
  )
}

export default function RNNApplications() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        RNNs support a flexible set of input-output configurations, making them applicable to
        a wide range of sequence tasks including classification, generation, and language modeling.
      </p>

      <DefinitionBlock title="Sequence Classification (Many-to-One)">
        <p>
          The RNN processes an entire input sequence and produces a single output from the final
          hidden state:
        </p>
        <BlockMath math="\hat{y} = \text{softmax}(W_y \, h_T + b_y)" />
        <p className="mt-2">
          Common applications: sentiment analysis, spam detection, document classification.
        </p>
      </DefinitionBlock>

      <TaskTypeDiagram />

      <DefinitionBlock title="Language Modeling (Many-to-Many)">
        <p>
          At each time step the model predicts the next token given all previous tokens:
        </p>
        <BlockMath math="P(x_{t+1} | x_1, \ldots, x_t) = \text{softmax}(W_y \, h_t)" />
        <p className="mt-2">
          The training objective is to minimize the cross-entropy loss, which is equivalent to
          maximizing the log-likelihood of the data.
        </p>
        <BlockMath math="\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_{t+1} | x_{\le t})" />
      </DefinitionBlock>

      <ExampleBlock title="Perplexity">
        <p>Language model quality is measured by perplexity:</p>
        <BlockMath math="\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T}\log P(x_{t+1}|x_{\le t})\right)" />
        <p>A perplexity of 100 means the model is as uncertain as choosing uniformly among 100 tokens.</p>
      </ExampleBlock>

      <PythonCode
        title="Character-Level Language Model"
        code={`import torch
import torch.nn as nn

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h=None):
        e = self.embed(x)
        out, h = self.rnn(e, h)
        logits = self.head(out)
        return logits, h

# Example: vocab of 26 letters + space
model = CharRNN(vocab_size=27)
x = torch.randint(0, 27, (4, 50))  # batch=4, seq_len=50
logits, h = model(x)
print(f"Logits: {logits.shape}")  # (4, 50, 27)

# Greedy generation
def generate(model, seed, length=100):
    model.eval()
    tokens = [seed]
    h = None
    for _ in range(length):
        x = torch.tensor([[tokens[-1]]])
        logits, h = model(x, h)
        tokens.append(logits[0, -1].argmax().item())
    return tokens`}
      />

      <NoteBlock type="note" title="Sequence Generation (One-to-Many)">
        <p>
          In generation tasks like image captioning, a single input (e.g., a CNN feature vector) is
          fed as the initial hidden state, and the RNN autoregressively produces output tokens.
          <strong> Teacher forcing</strong> is commonly used during training: the ground-truth token at
          step <InlineMath math="t" /> is fed as input at step <InlineMath math="t+1" />, rather than
          the model's own prediction.
        </p>
      </NoteBlock>

      <ExampleBlock title="Practical Tip: Sampling Strategies">
        <p>
          During generation, greedy decoding always picks the most likely token. Alternatives include
          <strong> temperature scaling</strong> (<InlineMath math="p_i \propto \exp(\text{logit}_i / \tau)" />) and
          <strong> top-k sampling</strong>, which truncates the distribution to the <InlineMath math="k" /> most
          likely tokens before sampling.
        </p>
      </ExampleBlock>
    </div>
  )
}
