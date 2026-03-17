import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function Conv3DVisualizer() {
  const [temporalKernel, setTemporalKernel] = useState(3)
  const [spatialKernel, setSpatialKernel] = useState(3)
  const [inputFrames, setInputFrames] = useState(16)

  const params3D = temporalKernel * spatialKernel * spatialKernel * 64 * 64
  const params2D = spatialKernel * spatialKernel * 64 * 64
  const paramsP3D = spatialKernel * spatialKernel * 64 * 64 + temporalKernel * 64 * 64

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">3D Convolution Parameter Explorer</h3>
      <div className="flex flex-wrap gap-4 mb-4">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Temporal kernel: {temporalKernel}
          <input type="range" min={1} max={7} step={2} value={temporalKernel} onChange={e => setTemporalKernel(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Spatial kernel: {spatialKernel}
          <input type="range" min={1} max={7} step={2} value={spatialKernel} onChange={e => setSpatialKernel(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          Input frames: {inputFrames}
          <input type="range" min={4} max={64} step={4} value={inputFrames} onChange={e => setInputFrames(Number(e.target.value))} className="w-24 accent-violet-500" />
        </label>
      </div>
      <div className="grid grid-cols-3 gap-3 text-sm">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">Full 3D Conv</p>
          <p className="text-lg font-bold text-violet-600">{(params3D / 1000).toFixed(1)}K params</p>
          <p className="text-xs text-gray-500">kernel: {temporalKernel}x{spatialKernel}x{spatialKernel}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">2D Conv (per frame)</p>
          <p className="text-lg font-bold text-violet-600">{(params2D / 1000).toFixed(1)}K params</p>
          <p className="text-xs text-gray-500">kernel: {spatialKernel}x{spatialKernel}</p>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/20 p-3">
          <p className="text-xs text-violet-600 dark:text-violet-400 font-semibold">(2+1)D Factorized</p>
          <p className="text-lg font-bold text-violet-600">{(paramsP3D / 1000).toFixed(1)}K params</p>
          <p className="text-xs text-gray-500">spatial + temporal</p>
        </div>
      </div>
      <p className="mt-2 text-xs text-gray-500">
        Input volume: [{inputFrames}, {64}, 224, 224] &mdash; Full 3D is {(params3D / paramsP3D).toFixed(1)}x more parameters than (2+1)D
      </p>
    </div>
  )
}

export default function ThreeDCNNs() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        3D convolutional networks extend image CNNs to video by learning spatiotemporal features
        jointly. From C3D to I3D, these architectures capture motion and temporal patterns
        that frame-level models miss entirely.
      </p>

      <DefinitionBlock title="3D Convolution">
        <p>A 3D convolution operates over time, height, and width simultaneously:</p>
        <BlockMath math="y[t,h,w] = \sum_{\tau,i,j} W[\tau,i,j] \cdot x[t+\tau, h+i, w+j] + b" />
        <p className="mt-2">
          The kernel has shape <InlineMath math="(k_t, k_h, k_w)" />, where <InlineMath math="k_t" /> is
          the temporal extent. This enables learning motion patterns, speed changes, and
          temporal textures directly from raw video clips.
        </p>
      </DefinitionBlock>

      <Conv3DVisualizer />

      <TheoremBlock title="Inflating 2D to 3D: I3D" id="i3d-inflation">
        <p>
          I3D (Inflated 3D ConvNets) inflates pre-trained 2D ImageNet weights into 3D by repeating
          along the temporal dimension and rescaling:
        </p>
        <BlockMath math="W_\text{3D}[\tau, i, j] = \frac{1}{k_t} W_\text{2D}[i, j] \quad \forall \tau \in \{1, \ldots, k_t\}" />
        <p className="mt-1">
          This preserves the spatial semantics learned on ImageNet while enabling temporal learning.
          I3D with Two-Stream (RGB + optical flow) achieved breakthrough results on Kinetics.
        </p>
      </TheoremBlock>

      <ExampleBlock title="(2+1)D Factorization">
        <p>R(2+1)D decomposes 3D convolution into spatial and temporal components:</p>
        <BlockMath math="3\text{D conv} \approx (1 \times k \times k) \text{ spatial} + (k_t \times 1 \times 1) \text{ temporal}" />
        <p className="mt-1">
          This doubles the number of nonlinearities (one after each sub-convolution) and reduces
          parameters. The factorization makes optimization easier while maintaining the ability
          to model spatiotemporal features.
        </p>
      </ExampleBlock>

      <PythonCode
        title="3D CNNs for Video Classification"
        code={`import torch
import torch.nn as nn

# Basic 3D convolution
conv3d = nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3))
video_clip = torch.randn(2, 3, 16, 224, 224)  # [B, C, T, H, W]
features = conv3d(video_clip)
print(f"3D conv output: {features.shape}")  # [2, 64, 16, 112, 112]

# (2+1)D factorized convolution
class R2Plus1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_kernel=3, s_kernel=3):
        super().__init__()
        mid_ch = (in_ch * out_ch * s_kernel**2 * t_kernel) // (in_ch * s_kernel**2 + out_ch * t_kernel)
        self.spatial = nn.Conv3d(in_ch, mid_ch, (1, s_kernel, s_kernel), padding=(0, s_kernel//2, s_kernel//2))
        self.temporal = nn.Conv3d(mid_ch, out_ch, (t_kernel, 1, 1), padding=(t_kernel//2, 0, 0))
        self.bn1 = nn.BatchNorm3d(mid_ch)
        self.bn2 = nn.BatchNorm3d(out_ch)

    def forward(self, x):
        x = torch.relu(self.bn1(self.spatial(x)))
        return torch.relu(self.bn2(self.temporal(x)))

block = R2Plus1DBlock(64, 64)
x = torch.randn(2, 64, 16, 56, 56)
out = block(x)
print(f"(2+1)D output: {out.shape}")  # [2, 64, 16, 56, 56]

# Full 3D params vs (2+1)D params
full_3d_params = 3 * 3 * 3 * 64 * 64
r21d_params = sum(p.numel() for p in block.parameters() if p.requires_grad)
print(f"Full 3D: {full_3d_params:,} vs (2+1)D: {r21d_params:,}")`}
      />

      <NoteBlock type="note" title="Two-Stream Hypothesis">
        <p>
          The two-stream architecture processes RGB frames (appearance) and optical flow
          (motion) through separate networks, fusing predictions at the end. While I3D showed
          optical flow helps significantly, modern approaches like SlowFast learn temporal
          patterns directly from RGB, reducing the need for expensive optical flow computation.
        </p>
      </NoteBlock>
    </div>
  )
}
