import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ClassificationArchCompare() {
  const [arch, setArch] = useState('cnn')
  const architectures = {
    cnn: { name: 'CNN (InceptionTime)', layers: ['Conv 1x10', 'Conv 1x20', 'Conv 1x40', 'MaxPool', 'GAP', 'FC'], color: '#8b5cf6' },
    rnn: { name: 'LSTM Classifier', layers: ['LSTM-1', 'LSTM-2', 'Last h_T', 'FC', 'Softmax', ''], color: '#f97316' },
    transformer: { name: 'TST Classifier', layers: ['Patch', 'Pos Enc', 'Encoder x3', 'CLS Token', 'FC', ''], color: '#06b6d4' },
  }
  const a = architectures[arch]
  const blockW = 60, blockH = 36, gap = 10
  const totalW = a.layers.filter(l => l).length * (blockW + gap) + 20

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Classification Architecture Comparison</h3>
      <div className="flex gap-2 mb-3">
        {Object.entries(architectures).map(([key, val]) => (
          <button key={key} onClick={() => setArch(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${arch === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <svg width={totalW} height={80} className="mx-auto block">
          {a.layers.filter(l => l).map((layer, i) => (
            <g key={i}>
              {i > 0 && <line x1={i * (blockW + gap) - gap + 5} y1={40} x2={i * (blockW + gap) + 5} y2={40} stroke="#d1d5db" strokeWidth={1.5} markerEnd="url(#arrowC)" />}
              <rect x={i * (blockW + gap) + 5} y={22} width={blockW} height={blockH} rx={6} fill={a.color} opacity={0.15 + (i / a.layers.length) * 0.3} stroke={a.color} strokeWidth={1.5} />
              <text x={i * (blockW + gap) + 5 + blockW / 2} y={44} textAnchor="middle" className="text-[9px] fill-gray-700 dark:fill-gray-300">{layer}</text>
            </g>
          ))}
          <defs>
            <marker id="arrowC" viewBox="0 0 10 10" refX={9} refY={5} markerWidth={4} markerHeight={4} orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#d1d5db" />
            </marker>
          </defs>
        </svg>
      </div>
      <p className="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">
        {arch === 'cnn' && 'Multi-scale convolutions capture patterns at different temporal resolutions'}
        {arch === 'rnn' && 'Sequential processing captures order-dependent features, uses final hidden state'}
        {arch === 'transformer' && 'Self-attention over patches with a learnable classification token'}
      </p>
    </div>
  )
}

export default function TSClassificationApproaches() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Time series classification assigns a label to an entire sequence based on its
        temporal patterns. Applications include ECG diagnosis, activity recognition, and
        industrial fault detection. Deep learning approaches have largely replaced
        traditional distance-based methods on standard benchmarks.
      </p>

      <DefinitionBlock title="Time Series Classification Task">
        <p>Given a labeled dataset <InlineMath math="\{(\mathbf{x}_i, y_i)\}_{i=1}^N" /> where <InlineMath math="\mathbf{x}_i \in \mathbb{R}^T" /> and <InlineMath math="y_i \in \{1, \ldots, K\}" />:</p>
        <BlockMath math="f_\theta : \mathbb{R}^T \to \Delta^K, \qquad \hat{y} = \arg\max_k f_\theta(\mathbf{x})_k" />
        <p className="mt-2">The model maps a variable-length time series to a probability distribution over <InlineMath math="K" /> classes.</p>
      </DefinitionBlock>

      <ClassificationArchCompare />

      <TheoremBlock title="InceptionTime Architecture" id="inceptiontime">
        <p>InceptionTime applies multiple parallel convolutions with different kernel sizes at each layer:</p>
        <BlockMath math="\mathbf{h}_\ell = \text{BN}\left(\sum_{k \in \{10,20,40\}} \text{Conv}_{1 \times k}(\mathbf{h}_{\ell-1}) + \text{MaxPool}_{3}(\mathbf{h}_{\ell-1})\right)" />
        <p>Global Average Pooling (GAP) aggregates temporal features: <InlineMath math="\bar{\mathbf{h}} = \frac{1}{T}\sum_t \mathbf{h}_t" />, followed by a linear classifier.</p>
      </TheoremBlock>

      <ExampleBlock title="ResNet Baseline">
        <p>
          A simple 1D ResNet with 3 residual blocks of (Conv-BN-ReLU) x 3 and GAP achieves
          competitive results on the UCR archive (128 datasets). It serves as the standard
          deep learning baseline, outperforming most non-DL methods with minimal tuning.
        </p>
      </ExampleBlock>

      <PythonCode
        title="InceptionTime Module in PyTorch"
        code={`import torch
import torch.nn as nn

class InceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch=32):
        super().__init__()
        self.conv10 = nn.Conv1d(in_ch, out_ch, kernel_size=10, padding=4)
        self.conv20 = nn.Conv1d(in_ch, out_ch, kernel_size=20, padding=9)
        self.conv40 = nn.Conv1d(in_ch, out_ch, kernel_size=40, padding=19)
        self.mp_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
        )
        self.bn = nn.BatchNorm1d(out_ch * 4)

    def forward(self, x):  # x: (B, C, T)
        c10 = self.conv10(x)[:, :, :x.size(2)]
        c20 = self.conv20(x)[:, :, :x.size(2)]
        c40 = self.conv40(x)[:, :, :x.size(2)]
        mp = self.mp_conv(x)
        return torch.relu(self.bn(torch.cat([c10, c20, c40, mp], dim=1)))

class InceptionTime(nn.Module):
    def __init__(self, in_ch=1, n_classes=5, depth=6):
        super().__init__()
        ch = 32 * 4  # output channels per inception block
        self.blocks = nn.ModuleList([InceptionBlock(in_ch if i == 0 else ch) for i in range(depth)])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(ch, n_classes)

    def forward(self, x):  # x: (B, 1, T)
        for block in self.blocks:
            x = block(x)
        return self.fc(self.gap(x).squeeze(-1))

model = InceptionTime(in_ch=1, n_classes=5)
x = torch.randn(8, 1, 128)
print(f"Predictions: {model(x).shape}")  # (8, 5)`}
      />

      <NoteBlock type="note" title="Ensembling for Robustness">
        <p>
          The original InceptionTime paper uses an ensemble of 5 models with different random
          initializations. This reduces variance significantly and is a common practice in
          time series classification where datasets are often small (tens to hundreds of samples).
        </p>
      </NoteBlock>
    </div>
  )
}
