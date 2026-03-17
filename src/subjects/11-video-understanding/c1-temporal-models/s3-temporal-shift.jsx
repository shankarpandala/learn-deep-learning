import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TemporalShiftDemo() {
  const [shiftRatio, setShiftRatio] = useState(0.25)
  const numChannels = 8
  const numFrames = 5
  const shiftCount = Math.round(numChannels * shiftRatio)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Temporal Shift Visualization</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
        Shift ratio: {(shiftRatio * 100).toFixed(0)}% ({shiftCount} forward + {shiftCount} backward of {numChannels} channels)
        <input type="range" min={0.125} max={0.5} step={0.125} value={shiftRatio} onChange={e => setShiftRatio(Number(e.target.value))} className="w-32 accent-violet-500" />
      </label>
      <div className="overflow-x-auto">
        <div className="grid gap-1" style={{ gridTemplateColumns: `60px repeat(${numFrames}, 1fr)` }}>
          <div className="text-xs text-gray-500" />
          {Array.from({ length: numFrames }).map((_, t) => (
            <div key={t} className="text-xs text-center text-gray-500 font-semibold">t={t}</div>
          ))}
          {Array.from({ length: numChannels }).map((_, c) => (
            <React.Fragment key={c}>
              <div className="text-xs text-gray-500 flex items-center">ch {c}</div>
              {Array.from({ length: numFrames }).map((_, t) => {
                let color = 'bg-gray-200 dark:bg-gray-700'
                let label = ''
                if (c < shiftCount) {
                  color = 'bg-violet-400 text-white'
                  label = t > 0 ? `t${t - 1}` : 'pad'
                } else if (c < 2 * shiftCount) {
                  color = 'bg-violet-600 text-white'
                  label = t < numFrames - 1 ? `t${t + 1}` : 'pad'
                } else {
                  color = 'bg-violet-100 dark:bg-violet-900/30'
                  label = `t${t}`
                }
                return <div key={t} className={`${color} rounded text-xs text-center py-1`}>{label}</div>
              })}
            </React.Fragment>
          ))}
        </div>
      </div>
      <div className="flex gap-4 mt-2 text-xs">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-violet-400" /> Forward shift</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-violet-600" /> Backward shift</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-violet-100 dark:bg-violet-900/30" /> No shift</span>
      </div>
    </div>
  )
}

export default function TemporalShiftTSM() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The Temporal Shift Module (TSM) enables temporal reasoning with 2D CNNs by simply
        shifting a portion of channels along the time dimension, achieving 3D CNN-level accuracy
        at 2D CNN computational cost.
      </p>

      <DefinitionBlock title="Temporal Shift Module">
        <p>TSM shifts a fraction of channels along the temporal axis:</p>
        <BlockMath math="X'_t[c] = \begin{cases} X_{t-1}[c] & \text{if } c < C/4 \text{ (forward shift)} \\ X_{t+1}[c] & \text{if } C/4 \leq c < C/2 \text{ (backward shift)} \\ X_t[c] & \text{otherwise (no shift)} \end{cases}" />
        <p className="mt-2">
          This zero-parameter, zero-FLOP operation enables temporal information exchange.
          When followed by a 2D convolution, the network effectively computes 3D features.
        </p>
      </DefinitionBlock>

      <TemporalShiftDemo />

      <TheoremBlock title="TSM Equivalence to 3D Convolution" id="tsm-equivalence">
        <p>
          A temporal shift followed by a <InlineMath math="1 \times 1" /> convolution across channels
          is equivalent to a <InlineMath math="3 \times 1 \times 1" /> depthwise-separable 3D convolution:
        </p>
        <BlockMath math="\text{Shift} + \text{Conv2D}(1{\times}1) \equiv \text{Conv3D}_\text{depthwise}(3{\times}1{\times}1)" />
        <p className="mt-1">
          Combined with spatial convolutions in a ResNet block, TSM captures full spatiotemporal
          patterns without any 3D convolution parameters.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Residual Shift for Stability">
        <p>
          In-place shifting can harm spatial feature learning. The residual shift variant
          adds the shifted features instead of replacing:
        </p>
        <BlockMath math="X'_t = X_t + \alpha \cdot \text{Shift}(X_t)" />
        <p className="mt-1">
          With <InlineMath math="\alpha" /> initialized small, this preserves the pre-trained 2D
          features while gradually learning temporal patterns. In practice, the partial shift
          (1/8 or 1/4 of channels) works best.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Temporal Shift Module Implementation"
        code={`import torch
import torch.nn as nn

class TemporalShift(nn.Module):
    def __init__(self, n_segment=8, shift_ratio=0.25):
        super().__init__()
        self.n_segment = n_segment
        self.shift_ratio = shift_ratio

    def forward(self, x):
        B, C, H, W = x.shape
        T = self.n_segment
        BT = B // T  # batch per clip
        x = x.view(BT, T, C, H, W)
        shift = int(C * self.shift_ratio)

        out = x.clone()
        # Forward shift: channels [0, shift) get frame t-1
        out[:, 1:, :shift] = x[:, :-1, :shift]
        out[:, 0, :shift] = 0  # zero-pad first frame
        # Backward shift: channels [shift, 2*shift) get frame t+1
        out[:, :-1, shift:2*shift] = x[:, 1:, shift:2*shift]
        out[:, -1, shift:2*shift] = 0  # zero-pad last frame
        # Remaining channels unchanged

        return out.view(B, C, H, W)

# Apply TSM to a ResNet backbone
class TSMResBlock(nn.Module):
    def __init__(self, channels=64, n_segment=8):
        super().__init__()
        self.tsm = TemporalShift(n_segment=n_segment)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = self.tsm(x)  # temporal shift (zero params!)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + identity)

T, B = 8, 2
x = torch.randn(B * T, 64, 56, 56)  # [BT, C, H, W]
block = TSMResBlock(64, n_segment=T)
out = block(x)
print(f"TSM output: {out.shape}")  # [16, 64, 56, 56]`}
      />

      <NoteBlock type="note" title="TSM for Online/Streaming Video">
        <p>
          TSM's unidirectional variant (forward shift only) enables online video understanding
          where future frames are unavailable. This is critical for real-time applications like
          autonomous driving, live sports analysis, and streaming action detection. The
          computational overhead is essentially zero compared to frame-by-frame 2D CNN processing.
        </p>
      </NoteBlock>
    </div>
  )
}
