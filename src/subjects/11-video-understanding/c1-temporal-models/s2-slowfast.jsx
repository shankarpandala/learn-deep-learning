import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function SlowFastVisualizer() {
  const [alpha, setAlpha] = useState(8)
  const [beta, setBeta] = useState(8)
  const totalFrames = 64

  const slowFrames = Math.floor(totalFrames / alpha)
  const fastFrames = totalFrames
  const slowChannels = 64
  const fastChannels = Math.floor(slowChannels / beta)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">SlowFast Dual-Pathway Design</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Alpha (temporal stride): {alpha}
          <input type="range" min={2} max={16} step={2} value={alpha} onChange={e => setAlpha(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Beta (channel ratio): {beta}
          <input type="range" min={2} max={16} step={2} value={beta} onChange={e => setBeta(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-2 gap-4">
        <div className="rounded-lg bg-violet-100 dark:bg-violet-900/30 p-4">
          <p className="font-bold text-violet-700 dark:text-violet-300 text-sm">Slow Pathway</p>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Frames: {slowFrames} (every {alpha}th frame)</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Channels: {slowChannels}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">FLOPs share: ~80%</p>
          <div className="flex gap-1 mt-2">
            {Array.from({ length: Math.min(slowFrames, 12) }).map((_, i) => (
              <div key={i} className="w-4 h-6 rounded bg-violet-500" />
            ))}
            {slowFrames > 12 && <span className="text-xs text-violet-500">+{slowFrames - 12}</span>}
          </div>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-4">
          <p className="font-bold text-violet-600 dark:text-violet-400 text-sm">Fast Pathway</p>
          <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Frames: {fastFrames} (all frames)</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">Channels: {fastChannels}</p>
          <p className="text-xs text-gray-600 dark:text-gray-400">FLOPs share: ~20%</p>
          <div className="flex gap-0.5 mt-2 flex-wrap">
            {Array.from({ length: Math.min(fastFrames, 24) }).map((_, i) => (
              <div key={i} className="w-2 h-4 rounded-sm bg-violet-400" />
            ))}
            {fastFrames > 24 && <span className="text-xs text-violet-400">+{fastFrames - 24}</span>}
          </div>
        </div>
      </div>
      <p className="text-xs text-gray-500 mt-2">
        Slow: high spatial detail, low temporal rate. Fast: fine temporal resolution, lightweight channels.
      </p>
    </div>
  )
}

export default function SlowFastNetworks() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        SlowFast Networks process video through two parallel pathways operating at different
        temporal rates, inspired by the primate visual system's distinction between sustained
        (parvocellular) and transient (magnocellular) processing.
      </p>

      <DefinitionBlock title="SlowFast Architecture">
        <p>Two pathways process video at different temporal resolutions:</p>
        <BlockMath math="\text{Slow}: T/\alpha \text{ frames}, \quad \text{Fast}: T \text{ frames with } C/\beta \text{ channels}" />
        <p className="mt-2">
          The Slow pathway operates at <InlineMath math="1/\alpha" /> temporal rate (e.g., 2 fps)
          with full channel capacity, capturing spatial semantics. The Fast pathway runs at full
          frame rate with <InlineMath math="1/\beta" /> channels (lightweight), capturing fine
          temporal patterns and motion.
        </p>
      </DefinitionBlock>

      <SlowFastVisualizer />

      <TheoremBlock title="Lateral Connections" id="lateral-connections">
        <p>
          Information flows from Fast to Slow pathway via lateral connections at each stage.
          Since frame rates differ, temporal resolution must be matched:
        </p>
        <BlockMath math="x_\text{slow}^{l+1} = f(x_\text{slow}^l, \text{Fuse}(x_\text{fast}^l))" />
        <p className="mt-1">
          The fusion uses either time-strided convolution (<InlineMath math="5 \times 1^2" /> with
          stride <InlineMath math="\alpha" />) or time-to-channel reshaping. This allows the Slow
          pathway to benefit from fine temporal information without processing all frames.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Computational Efficiency">
        <p>With typical settings (<InlineMath math="\alpha=8, \beta=8" />):</p>
        <BlockMath math="\frac{\text{FLOPs}_\text{Fast}}{\text{FLOPs}_\text{Slow}} \approx \frac{\alpha}{\beta^2} = \frac{8}{64} = 12.5\%" />
        <p className="mt-1">
          The Fast pathway adds only ~20% computational overhead while processing 8x more
          frames. This asymmetric design is key: temporal resolution is cheap when channels are few.
        </p>
      </ExampleBlock>

      <PythonCode
        title="SlowFast Network in PyTorch"
        code={`import torch
import torch.nn as nn

class SlowFastBlock(nn.Module):
    def __init__(self, slow_ch=64, fast_ch=8, alpha=8):
        super().__init__()
        self.alpha = alpha
        # Slow pathway: standard 3D conv
        self.slow_conv = nn.Conv3d(slow_ch, slow_ch, (1, 3, 3), padding=(0, 1, 1))
        # Fast pathway: lightweight 3D conv
        self.fast_conv = nn.Conv3d(fast_ch, fast_ch, (3, 3, 3), padding=(1, 1, 1))
        # Lateral connection: Fast -> Slow
        self.lateral = nn.Conv3d(fast_ch, slow_ch, (alpha, 1, 1), stride=(alpha, 1, 1))
        self.slow_bn = nn.BatchNorm3d(slow_ch)
        self.fast_bn = nn.BatchNorm3d(fast_ch)

    def forward(self, x_slow, x_fast):
        # Process each pathway
        slow_out = torch.relu(self.slow_bn(self.slow_conv(x_slow)))
        fast_out = torch.relu(self.fast_bn(self.fast_conv(x_fast)))
        # Fuse fast into slow via lateral connection
        lateral = self.lateral(fast_out)
        slow_out = slow_out + lateral
        return slow_out, fast_out

# Create SlowFast inputs
alpha, beta = 8, 8
T = 64  # total frames
slow_input = torch.randn(2, 64, T // alpha, 56, 56)  # [B, C, T/alpha, H, W]
fast_input = torch.randn(2, 64 // beta, T, 56, 56)    # [B, C/beta, T, H, W]

block = SlowFastBlock(slow_ch=64, fast_ch=64 // beta, alpha=alpha)
slow_out, fast_out = block(slow_input, fast_input)
print(f"Slow output: {slow_out.shape}")  # [2, 64, 8, 56, 56]
print(f"Fast output: {fast_out.shape}")  # [2, 8, 64, 56, 56]`}
      />

      <NoteBlock type="note" title="SlowFast for Detection">
        <p>
          SlowFast is widely used as the backbone for spatiotemporal action detection (e.g.,
          AVA dataset). Features from both pathways are RoI-pooled around person bounding boxes,
          concatenated, and classified per-actor. This achieved the first superhuman results
          on several action detection benchmarks.
        </p>
      </NoteBlock>
    </div>
  )
}
