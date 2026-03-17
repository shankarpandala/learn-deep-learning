import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function ClippingDemo() {
  const [maxNorm, setMaxNorm] = useState(5.0)
  const gradNorms = [1.2, 3.5, 8.0, 15.0, 2.1, 50.0, 0.8, 7.3]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Gradient Clipping Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Max norm: {maxNorm.toFixed(1)}
        <input type="range" min={1} max={20} step={0.5} value={maxNorm} onChange={e => setMaxNorm(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex items-end gap-2 h-32">
        {gradNorms.map((g, i) => {
          const clipped = Math.min(g, maxNorm)
          const scale = 2.2
          return (
            <div key={i} className="flex flex-col items-center gap-1 flex-1">
              <span className="text-xs text-gray-500">{clipped.toFixed(1)}</span>
              <div className="w-full flex flex-col items-center">
                <div style={{ height: `${clipped * scale}px` }}
                  className={`w-6 rounded-t ${g > maxNorm ? 'bg-violet-400' : 'bg-violet-600'}`} />
                {g > maxNorm && (
                  <div style={{ height: `${(g - clipped) * scale}px` }}
                    className="w-6 bg-red-200 dark:bg-red-900/40 border-t border-dashed border-red-400" />
                )}
              </div>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-gray-500 mt-2 text-center">Violet = kept, red dashed = clipped away</p>
    </div>
  )
}

export default function LSTMTraining() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Training LSTMs effectively requires careful attention to gradient management,
        weight initialization, and hyperparameter choices. These practical techniques
        are essential for stable, fast convergence.
      </p>

      <DefinitionBlock title="Gradient Clipping">
        <p>Gradient clipping rescales the gradient when its norm exceeds a threshold <InlineMath math="\theta" />:</p>
        <BlockMath math="g \leftarrow \begin{cases} g & \text{if } \|g\| \leq \theta \\ \theta \cdot \frac{g}{\|g\|} & \text{if } \|g\| > \theta \end{cases}" />
        <p className="mt-2">
          This preserves gradient direction while bounding its magnitude, preventing
          the catastrophic parameter updates caused by exploding gradients.
        </p>
      </DefinitionBlock>

      <ClippingDemo />

      <TheoremBlock title="Forget Gate Bias Initialization" id="forget-bias-init">
        <p>
          Initializing the forget gate bias to a positive value (typically 1.0 or 2.0) ensures
          that <InlineMath math="f_t \approx 1" /> at the start of training, which allows
          gradients to flow through the cell state early in training:
        </p>
        <BlockMath math="b_f \leftarrow 1.0 \implies f_t = \sigma(Wh + Wx + 1.0) \approx 0.73" />
        <p>
          This simple trick, proposed by Gers et al. (2000), significantly improves LSTM
          training stability and is now standard practice.
        </p>
      </TheoremBlock>

      <PythonCode
        title="LSTM Training with Best Practices"
        code={`import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                           dropout=0.3, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Best practice: initialize forget gate bias to 1.0
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)  # forget gate bias
            elif 'weight' in name:
                nn.init.orthogonal_(param)  # orthogonal init for stability

    def forward(self, x):
        e = self.embed(x)
        out, (h_n, _) = self.lstm(e)
        return self.fc(self.dropout(h_n[-1]))

model = LSTMClassifier(10000, 128, 256, 5)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop with gradient clipping
for epoch in range(5):
    x = torch.randint(0, 10000, (32, 50))
    y = torch.randint(0, 5, (32,))
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")`}
      />

      <WarningBlock title="Common LSTM Training Pitfalls">
        <p>
          <strong>1. Forgetting to clip gradients</strong> leads to NaN losses.
          <strong> 2. Default zero bias</strong> for forget gates causes information loss early in training.
          <strong> 3. Too-high learning rate</strong> with Adam can destabilize LSTMs more than feedforward
          networks due to the recurrent dynamics.
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Learning Rate Scheduling">
        <p>
          LSTMs benefit from learning rate warmup (linearly increasing over the first ~1000 steps)
          followed by cosine or step decay. A common recipe: start with <InlineMath math="\text{lr} = 10^{-3}" /> for
          Adam or <InlineMath math="\text{lr} = 1.0" /> for SGD with gradient clipping at 0.25-5.0.
        </p>
      </NoteBlock>
    </div>
  )
}
