import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TrustRegionViz() {
  const [delta, setDelta] = useState(0.5)
  const W = 300, H = 200, cx = 150, cy = 100, scale = 80

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Trust Region Constraint</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          delta = {delta.toFixed(2)}
          <input type="range" min={0.05} max={1.5} step={0.05} value={delta} onChange={e => setDelta(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <svg width={W} height={H} className="mx-auto block">
        <circle cx={cx} cy={cy} r={delta * scale} fill="none" stroke="#7c3aed" strokeWidth={2} strokeDasharray="5,3" opacity={0.7} />
        <circle cx={cx} cy={cy} r={delta * scale} fill="#7c3aed" opacity={0.08} />
        <circle cx={cx} cy={cy} r={5} fill="#7c3aed" />
        <text x={cx + 8} y={cy - 8} fill="#7c3aed" fontSize={11}>theta_old</text>
        {delta > 0.3 && <circle cx={cx + delta * scale * 0.5} cy={cy - delta * scale * 0.4} r={4} fill="#f97316" />}
        {delta > 0.3 && <text x={cx + delta * scale * 0.5 + 8} y={cy - delta * scale * 0.4} fill="#f97316" fontSize={11}>theta_new</text>}
        <text x={cx + delta * scale + 4} y={cy + 4} fill="#7c3aed" fontSize={10} opacity={0.7}>KL &le; delta</text>
      </svg>
      <p className="text-center text-xs text-gray-500 mt-1">Larger delta allows bigger policy updates but risks instability</p>
    </div>
  )
}

export default function TRPO() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Trust Region Policy Optimization (TRPO) constrains each policy update to stay close
        to the current policy, guaranteeing monotonic improvement under certain conditions.
      </p>

      <TheoremBlock title="Surrogate Objective" id="surrogate-objective">
        <p>TRPO maximizes a lower bound on the true policy improvement:</p>
        <BlockMath math="\max_\theta \; \mathbb{E}_{s,a \sim \pi_{\theta_\text{old}}}\left[\frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} \hat{A}(s,a)\right]" />
        <BlockMath math="\text{subject to} \quad \mathbb{E}_s\left[D_\text{KL}\!\left(\pi_{\theta_\text{old}}(\cdot|s) \| \pi_\theta(\cdot|s)\right)\right] \le \delta" />
      </TheoremBlock>

      <DefinitionBlock title="Natural Policy Gradient">
        <p>The natural gradient preconditions with the Fisher information matrix:</p>
        <BlockMath math="\theta \leftarrow \theta + \alpha F^{-1} \nabla_\theta J(\theta)" />
        <p className="mt-2">where <InlineMath math="F = \mathbb{E}[\nabla \log \pi \cdot \nabla \log \pi^\top]" />.
        TRPO approximates this using conjugate gradients to avoid computing <InlineMath math="F^{-1}" /> directly.</p>
      </DefinitionBlock>

      <TrustRegionViz />

      <ExampleBlock title="TRPO vs Vanilla Policy Gradient">
        <p>On the Humanoid-v2 MuJoCo task (376-dim state, 17-dim action):</p>
        <p><strong>Vanilla PG</strong>: Training collapses after ~200 episodes due to large destructive updates.</p>
        <p><strong>TRPO</strong>: Monotonically improves, reaching 2000+ reward. The KL constraint prevents
        catastrophic policy changes.</p>
      </ExampleBlock>

      <PythonCode
        title="TRPO Core: Conjugate Gradient Step"
        code={`import torch

def conjugate_gradient(Fvp, b, n_steps=10, residual_tol=1e-10):
    """Solve Fx = b using conjugate gradient, where Fvp computes F@v."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = b.clone()
    rdotr = r.dot(r)
    for _ in range(n_steps):
        Fp = Fvp(p)
        alpha = rdotr / (p.dot(Fp) + 1e-8)
        x += alpha * p
        r -= alpha * Fp
        new_rdotr = r.dot(r)
        if new_rdotr < residual_tol:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr
    return x

def trpo_step(policy, get_loss, get_kl, max_kl=0.01):
    """One TRPO update step."""
    loss = get_loss()
    grads = torch.autograd.grad(loss, policy.parameters())
    flat_grad = torch.cat([g.view(-1) for g in grads])

    def Fvp(v):  # Fisher-vector product
        kl = get_kl()
        kl_grad = torch.autograd.grad(kl, policy.parameters(), create_graph=True)
        flat_kl_grad = torch.cat([g.view(-1) for g in kl_grad])
        return torch.autograd.grad(flat_kl_grad.dot(v), policy.parameters())

    step_dir = conjugate_gradient(Fvp, flat_grad)
    shs = 0.5 * step_dir.dot(Fvp(step_dir))
    step_size = torch.sqrt(2 * max_kl / (shs + 1e-8))
    return step_size * step_dir`}
      />

      <NoteBlock type="note" title="TRPO to PPO">
        <p>
          TRPO's constrained optimization with conjugate gradients is complex to implement and
          computationally expensive. PPO replaces the hard KL constraint with a simpler clipped
          surrogate objective, achieving similar or better performance with much simpler code.
        </p>
      </NoteBlock>
    </div>
  )
}
