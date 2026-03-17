import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function LaplacianViz() {
  const [showNorm, setShowNorm] = useState(false)
  const A = [[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]]
  const D = A.map(row => row.reduce((a, b) => a + b, 0))
  const L = A.map((row, i) => row.map((v, j) => (i === j ? D[i] : 0) - v))
  const Lnorm = A.map((row, i) => row.map((v, j) => {
    if (i === j && D[i] > 0) return 1
    if (v === 1) return (-1 / Math.sqrt(D[i] * D[j])).toFixed(2)
    return 0
  }))
  const mat = showNorm ? Lnorm : L

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Graph Laplacian</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          <input type="checkbox" checked={showNorm} onChange={e => setShowNorm(e.target.checked)} className="accent-violet-500" />
          Normalized Laplacian
        </label>
      </div>
      <div className="flex items-center gap-8 justify-center">
        <div>
          <p className="text-xs text-gray-500 mb-1 text-center">A (adjacency)</p>
          <table className="text-xs font-mono border-collapse">
            <tbody>{A.map((row, i) => (
              <tr key={i}>{row.map((v, j) => <td key={j} className={`px-2 py-1 text-center ${v ? 'text-violet-600 font-bold' : 'text-gray-400'}`}>{v}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
        <span className="text-gray-400 text-lg">&#8594;</span>
        <div>
          <p className="text-xs text-gray-500 mb-1 text-center">{showNorm ? 'L_norm' : 'L = D - A'}</p>
          <table className="text-xs font-mono border-collapse">
            <tbody>{mat.map((row, i) => (
              <tr key={i}>{row.map((v, j) => <td key={j} className={`px-2 py-1 text-center ${i === j ? 'text-violet-700 dark:text-violet-400 font-bold bg-violet-50 dark:bg-violet-900/20' : parseFloat(v) < 0 ? 'text-orange-600' : 'text-gray-400'}`}>{v}</td>)}</tr>
            ))}</tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default function SpectralBasics() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Spectral graph theory studies graphs through the eigenvalues and eigenvectors of matrices
        associated with them, particularly the graph Laplacian. This forms the mathematical
        foundation for spectral graph convolutions.
      </p>

      <DefinitionBlock title="Graph Laplacian">
        <p>The combinatorial Laplacian:</p>
        <BlockMath math="L = D - A" />
        <p className="mt-2">The symmetric normalized Laplacian:</p>
        <BlockMath math="L_\text{sym} = D^{-1/2} L D^{-1/2} = I - D^{-1/2} A D^{-1/2}" />
        <p className="mt-2"><InlineMath math="L" /> is positive semi-definite with eigenvalues <InlineMath math="0 = \lambda_1 \le \lambda_2 \le \cdots \le \lambda_N" />.</p>
      </DefinitionBlock>

      <LaplacianViz />

      <TheoremBlock title="Spectral Decomposition" id="spectral-decomp">
        <p>The Laplacian admits an eigendecomposition:</p>
        <BlockMath math="L = U \Lambda U^\top, \quad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_N)" />
        <p className="mt-2">The eigenvectors <InlineMath math="U" /> form an orthonormal basis for signals
        on the graph. The <strong>Graph Fourier Transform</strong> of a signal <InlineMath math="\mathbf{x}" /> is:</p>
        <BlockMath math="\hat{\mathbf{x}} = U^\top \mathbf{x}" />
      </TheoremBlock>

      <ExampleBlock title="Eigenvalue Interpretation">
        <p><InlineMath math="\lambda_1 = 0" />: constant eigenvector. Number of zero eigenvalues = number of connected components.</p>
        <p><InlineMath math="\lambda_2" /> (Fiedler value): measures graph connectivity. Used for spectral clustering.</p>
        <p>Higher eigenvalues correspond to higher-frequency variations across the graph.</p>
      </ExampleBlock>

      <PythonCode
        title="Spectral Analysis with NumPy"
        code={`import numpy as np

# Define adjacency matrix
A = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 1, 1, 0, 1],
    [0, 0, 1, 1, 0]
])

D = np.diag(A.sum(axis=1))
L = D - A  # Combinatorial Laplacian

# Normalized Laplacian
D_inv_sqrt = np.diag(1.0 / np.sqrt(A.sum(axis=1)))
L_norm = np.eye(5) - D_inv_sqrt @ A @ D_inv_sqrt

# Eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
print("Eigenvalues:", eigenvalues.round(4))
print("Fiedler value (algebraic connectivity):", eigenvalues[1].round(4))
print("Fiedler vector:", eigenvectors[:, 1].round(3))

# Graph Fourier Transform of a signal
x = np.array([1.0, 0.5, -0.5, -1.0, 0.0])
x_hat = eigenvectors.T @ x  # spectral coefficients
print("Spectral coefficients:", x_hat.round(3))`}
      />

      <NoteBlock type="note" title="From Spectral to Spatial">
        <p>
          Spectral convolution (<InlineMath math="g_\theta \star x = U g_\theta(\Lambda) U^\top x" />) requires
          computing the full eigendecomposition (<InlineMath math="O(N^3)" />). ChebNet approximates the
          filter with Chebyshev polynomials, and GCN simplifies further to a first-order approximation,
          leading to the efficient spatial message-passing framework.
        </p>
      </NoteBlock>
    </div>
  )
}
