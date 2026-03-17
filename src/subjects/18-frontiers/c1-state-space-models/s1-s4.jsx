import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function SSMDualMode() {
  const [mode, setMode] = useState('recurrent')
  const modes = {
    recurrent: { name: 'Recurrent Mode (Inference)', complexity: 'O(L)', desc: 'Process one token at a time with hidden state. Sequential but constant memory.', formula: 'h_t = \\bar{A} h_{t-1} + \\bar{B} x_t, \\quad y_t = C h_t' },
    convolutional: { name: 'Convolutional Mode (Training)', complexity: 'O(L log L)', desc: 'Compute all outputs in parallel using a global convolution kernel via FFT.', formula: 'y = \\bar{K} * x, \\quad \\bar{K} = (C\\bar{B}, C\\bar{A}\\bar{B}, C\\bar{A}^2\\bar{B}, \\ldots)' },
  }
  const m = modes[mode]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">S4 Dual Computation Modes</h3>
      <div className="flex gap-2 mb-3">
        {Object.entries(modes).map(([key, val]) => (
          <button key={key} onClick={() => setMode(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${mode === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-2">
        <p className="text-gray-600 dark:text-gray-400">{m.desc}</p>
        <BlockMath math={m.formula} />
        <p className="text-xs text-gray-500">Complexity: {m.complexity}</p>
      </div>
    </div>
  )
}

export default function S4StructuredStateSpaces() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        S4 (Structured State Spaces for Sequence Modeling) provides an alternative to attention
        for long-range sequence modeling. By parameterizing a continuous-time linear system and
        discretizing it, S4 achieves efficient training via convolution and efficient inference
        via recurrence.
      </p>

      <DefinitionBlock title="Continuous-Time State Space Model">
        <p>A linear state space model maps input <InlineMath math="x(t)" /> to output <InlineMath math="y(t)" /> through a hidden state <InlineMath math="h(t)" />:</p>
        <BlockMath math="h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t) + Dx(t)" />
        <p className="mt-2">where <InlineMath math="A \in \mathbb{R}^{N \times N}" />, <InlineMath math="B \in \mathbb{R}^{N \times 1}" />, <InlineMath math="C \in \mathbb{R}^{1 \times N}" />. After discretization with step size <InlineMath math="\Delta" />:</p>
        <BlockMath math="\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B" />
      </DefinitionBlock>

      <SSMDualMode />

      <TheoremBlock title="HiPPO Initialization for Long-Range Memory" id="hippo">
        <p>The key to S4's long-range capability is the HiPPO (High-order Polynomial Projection Operators) initialization of matrix <InlineMath math="A" />:</p>
        <BlockMath math="A_{nk} = -\begin{cases} (2n+1)^{1/2}(2k+1)^{1/2} & \text{if } n > k \\ n+1 & \text{if } n = k \\ 0 & \text{if } n < k \end{cases}" />
        <p className="mt-2">This initialization ensures the state <InlineMath math="h(t)" /> optimally approximates the history of the input signal using Legendre polynomials, enabling memory over thousands of timesteps.</p>
      </TheoremBlock>

      <PythonCode
        title="Simplified S4 Convolution Kernel"
        code={`import torch
import torch.nn.functional as F

def s4_kernel(A, B, C, L, dt=1.0):
    """Compute the S4 convolution kernel of length L.

    Args:
        A: [N, N] state matrix
        B: [N, 1] input matrix
        C: [1, N] output matrix
        L: sequence length
        dt: discretization step size
    """
    N = A.shape[0]
    # Discretize (simplified zero-order hold)
    A_bar = torch.matrix_exp(A * dt)
    B_bar = torch.linalg.solve(A, (A_bar - torch.eye(N)) @ B)

    # Build kernel: K[i] = C @ A_bar^i @ B_bar
    kernel = torch.zeros(L)
    A_power = torch.eye(N)
    for i in range(L):
        kernel[i] = (C @ A_power @ B_bar).squeeze()
        A_power = A_power @ A_bar

    return kernel

def s4_convolve(x, kernel):
    """Apply S4 kernel as a global convolution using FFT."""
    L = x.shape[-1]
    # Pad for causal convolution
    K = F.pad(kernel, (0, L))
    X = F.pad(x, (0, kernel.shape[-1]))
    return torch.fft.irfft(torch.fft.rfft(X) * torch.fft.rfft(K))[:L]

# Example: 64-dim state, 1024-length sequence
N, L = 64, 1024
A = -torch.eye(N) + 0.1 * torch.randn(N, N)  # Stable A
B, C = torch.randn(N, 1), torch.randn(1, N)
kernel = s4_kernel(A, B, C, L)
print(f"Kernel shape: {kernel.shape}, decays: {kernel[:3]} ... {kernel[-3:]}")`}
      />

      <ExampleBlock title="S4 on Long Range Arena">
        <p>S4 achieved state-of-the-art on the Long Range Arena benchmark (sequences of 1K-16K tokens):</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Path-X (16K tokens): 96.4% — first model above random chance on this task</li>
          <li>Overall LRA average: 86.1% vs Transformer's 59.3%</li>
          <li>Key advantage: <InlineMath math="O(L \log L)" /> training vs <InlineMath math="O(L^2)" /> for attention</li>
        </ul>
      </ExampleBlock>

      <NoteBlock type="note" title="From S4 to Modern SSMs">
        <p>
          S4 spawned a family of models: S4D (diagonal approximation), S5 (parallel scan), H3
          (combining SSM with attention), and ultimately Mamba. Each simplification improved
          speed while maintaining the core benefit of efficient long-range modeling.
        </p>
      </NoteBlock>
    </div>
  )
}
