import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import ProofBlock from '../../../components/content/ProofBlock.jsx'

function AdvantageViz() {
  const [vState, setVState] = useState(3.0)
  const qValues = [1.5, 3.0, 4.5, 2.0]
  const actions = ['Left', 'Stay', 'Right', 'Jump']
  const advantages = qValues.map(q => q - vState)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Advantage = Q - V</h3>
      <div className="flex items-center gap-4 mb-3">
        <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
          V(s) = {vState.toFixed(1)}
          <input type="range" min={0} max={6} step={0.1} value={vState} onChange={e => setVState(parseFloat(e.target.value))} className="w-28 accent-violet-500" />
        </label>
      </div>
      <div className="flex flex-wrap gap-2 justify-center">
        {actions.map((act, i) => (
          <div key={i} className={`px-4 py-2 rounded-lg text-center ${advantages[i] >= 0 ? 'bg-violet-100 dark:bg-violet-900/40 border border-violet-400' : 'bg-gray-100 dark:bg-gray-800 border border-gray-300 dark:border-gray-600'}`}>
            <div className="text-xs text-gray-500">{act}</div>
            <div className="text-xs">Q={qValues[i].toFixed(1)}</div>
            <div className={`font-bold text-sm ${advantages[i] >= 0 ? 'text-violet-700 dark:text-violet-400' : 'text-gray-500'}`}>
              A={advantages[i].toFixed(1)}
            </div>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Positive advantage = better than average for this state</p>
    </div>
  )
}

export default function BaselinesAdvantage() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Subtracting a baseline from the return reduces gradient variance without changing the
        expected gradient. The advantage function is the canonical baseline choice.
      </p>

      <DefinitionBlock title="Advantage Function">
        <BlockMath math="A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)" />
        <p className="mt-2">The advantage measures how much better action <InlineMath math="a" /> is compared
        to the average action under <InlineMath math="\pi" />. By construction, <InlineMath math="\mathbb{E}_{a \sim \pi}[A(s,a)] = 0" />.</p>
      </DefinitionBlock>

      <TheoremBlock title="Baseline Does Not Change Expected Gradient" id="baseline-unbiased">
        <p>For any baseline <InlineMath math="b(s)" /> that depends only on the state:</p>
        <BlockMath math="\mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot b(s)\right] = 0" />
      </TheoremBlock>

      <ProofBlock title="Proof: Baseline is Zero in Expectation">
        <BlockMath math="\mathbb{E}_{a \sim \pi}\left[\nabla \log \pi(a|s) \cdot b(s)\right] = b(s) \sum_a \nabla \pi(a|s) = b(s) \nabla \underbrace{\sum_a \pi(a|s)}_{=1} = 0" />
      </ProofBlock>

      <AdvantageViz />

      <DefinitionBlock title="Generalized Advantage Estimation (GAE)">
        <p>GAE interpolates between high-bias (1-step TD) and high-variance (MC) estimates:</p>
        <BlockMath math="\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}" />
        <p className="mt-2">where <InlineMath math="\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)" /> is the TD residual.
        <InlineMath math="\lambda=0" /> gives 1-step TD, <InlineMath math="\lambda=1" /> gives MC returns.</p>
      </DefinitionBlock>

      <ExampleBlock title="Choosing Lambda">
        <p>In practice, <InlineMath math="\lambda = 0.95" /> works well for most tasks. It provides
        a good bias-variance tradeoff: mostly low-variance TD estimates with some Monte Carlo
        contribution for faster credit assignment over longer horizons.</p>
      </ExampleBlock>

      <PythonCode
        title="GAE Implementation"
        code={`import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    T = len(rewards)
    advantages = torch.zeros(T)
    gae = 0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < len(values) else 0
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values[:T]
    return advantages, returns

# Example usage
rewards = torch.tensor([1.0, 0.0, 0.0, 10.0])
values = torch.tensor([2.0, 1.5, 1.0, 0.5, 0.0])
dones = torch.tensor([0, 0, 0, 1])
adv, ret = compute_gae(rewards, values, dones)
print(f"Advantages: {adv}")
print(f"Returns: {ret}")`}
      />

      <NoteBlock type="note" title="From Baselines to Actor-Critic">
        <p>
          When we learn both a policy <InlineMath math="\pi_\theta" /> (actor) and a value
          function <InlineMath math="V_\phi" /> (critic), we get an actor-critic algorithm.
          The critic provides the baseline, and both are trained simultaneously.
        </p>
      </NoteBlock>
    </div>
  )
}
