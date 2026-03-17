import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DilatedConvVisualizer() {
  const [layer, setLayer] = useState(0)
  const dilation = Math.pow(2, layer)
  const receptiveField = Math.pow(2, layer + 1) - 1

  const W = 400, H = 160
  const nodeSpacing = W / 16
  const layers = 4
  const yStep = H / (layers + 1)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Causal Dilated Convolutions</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Layer: {layer} (dilation={dilation})
          <input type="range" min={0} max={3} step={1} value={layer} onChange={e => setLayer(Number(e.target.value))} className="w-32 accent-violet-500" />
        </label>
        <span className="text-sm text-violet-600 dark:text-violet-400 font-semibold">
          Receptive field: {receptiveField} samples
        </span>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        {Array.from({ length: 16 }).map((_, i) => (
          <g key={i}>
            <circle cx={i * nodeSpacing + nodeSpacing / 2} cy={H - yStep} r={4} fill="#d1d5db" />
            {i + dilation < 16 && (
              <line x1={i * nodeSpacing + nodeSpacing / 2} y1={H - yStep} x2={(i + dilation) * nodeSpacing + nodeSpacing / 2} y2={H - 2 * yStep} stroke="#8b5cf6" strokeWidth={1.5} opacity={0.6} />
            )}
            {i - dilation >= 0 && i < 16 && (
              <line x1={i * nodeSpacing + nodeSpacing / 2} y1={H - yStep} x2={i * nodeSpacing + nodeSpacing / 2} y2={H - 2 * yStep} stroke="#8b5cf6" strokeWidth={1.5} opacity={0.6} />
            )}
            <circle cx={i * nodeSpacing + nodeSpacing / 2} cy={H - 2 * yStep} r={4} fill="#8b5cf6" />
          </g>
        ))}
        <text x={5} y={H - yStep + 4} fontSize={10} fill="#9ca3af">Input</text>
        <text x={5} y={H - 2 * yStep + 4} fontSize={10} fill="#8b5cf6">Layer {layer}</text>
      </svg>
      <p className="text-xs text-center text-gray-500 mt-2">
        After {layers} layers with doubling dilation: receptive field = {Math.pow(2, layers + 1) - 1} samples
      </p>
    </div>
  )
}

export default function WaveNetAutoregressiveTTS() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        WaveNet generates raw audio waveforms sample-by-sample using causal dilated convolutions,
        producing speech quality that was indistinguishable from human recordings and
        revolutionizing both TTS and generative audio modeling.
      </p>

      <DefinitionBlock title="WaveNet Autoregressive Model">
        <p>WaveNet models the joint probability of a waveform as:</p>
        <BlockMath math="P(x) = \prod_{t=1}^{T} P(x_t | x_1, \ldots, x_{t-1})" />
        <p className="mt-2">
          Each sample is predicted using a stack of causal dilated convolutions with gated
          activations. At 16 kHz, this means generating 16,000 samples per second of audio.
        </p>
      </DefinitionBlock>

      <DilatedConvVisualizer />

      <TheoremBlock title="Exponential Receptive Field Growth" id="dilated-receptive-field">
        <p>
          With <InlineMath math="L" /> layers of dilation rates <InlineMath math="1, 2, 4, \ldots, 2^{L-1}" />,
          the receptive field grows exponentially:
        </p>
        <BlockMath math="R = 2^L - 1 + (k - 1)(2^L - 1)" />
        <p className="mt-1">
          where <InlineMath math="k" /> is the kernel size. With 10 layers repeated 3 times (30 layers total)
          and <InlineMath math="k=2" />, this covers <InlineMath math="\sim 300" /> ms at 16 kHz.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Gated Activation Unit">
        <p>Each WaveNet layer uses a gated activation inspired by LSTMs:</p>
        <BlockMath math="z = \tanh(W_f * x + V_f * h) \odot \sigma(W_g * x + V_g * h)" />
        <p className="mt-1">
          where <InlineMath math="*" /> denotes dilated convolution, <InlineMath math="h" /> is the
          conditioning input (e.g., mel spectrogram or speaker embedding), and <InlineMath math="\odot" /> is
          element-wise multiplication. Skip connections from each layer feed into the output.
        </p>
      </ExampleBlock>

      <PythonCode
        title="WaveNet Causal Dilated Conv Block"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetBlock(nn.Module):
    def __init__(self, channels=64, kernel_size=2, dilation=1):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            channels, 2 * channels, kernel_size,
            padding=dilation * (kernel_size - 1),  # causal padding
            dilation=dilation
        )
        self.cond_proj = nn.Conv1d(80, 2 * channels, 1)  # mel conditioning
        self.res_conv = nn.Conv1d(channels, channels, 1)
        self.skip_conv = nn.Conv1d(channels, channels, 1)

    def forward(self, x, cond):
        h = self.dilated_conv(x)[..., :x.size(-1)]  # causal trim
        h = h + self.cond_proj(cond)
        gate, filt = h.chunk(2, dim=1)
        h = torch.tanh(filt) * torch.sigmoid(gate)
        skip = self.skip_conv(h)
        res = self.res_conv(h) + x
        return res, skip

# Stack of dilated convolutions
channels = 64
blocks = nn.ModuleList([
    WaveNetBlock(channels, dilation=2**i) for i in range(10)
])

x = torch.randn(1, channels, 1000)
cond = torch.randn(1, 80, 1000)  # mel spectrogram
skip_sum = 0
for block in blocks:
    x, skip = block(x, cond)
    skip_sum = skip_sum + skip
print(f"Output: {skip_sum.shape}")  # [1, 64, 1000]`}
      />

      <NoteBlock type="note" title="From WaveNet to Real-Time Vocoders">
        <p>
          WaveNet's autoregressive generation is extremely slow (minutes per second of audio).
          Parallel WaveNet uses inverse autoregressive flows for real-time synthesis. Modern
          vocoders like <strong>HiFi-GAN</strong> and <strong>WaveGlow</strong> achieve real-time
          speeds with comparable quality using GAN training or flow-based methods.
        </p>
      </NoteBlock>
    </div>
  )
}
