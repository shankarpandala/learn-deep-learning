import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function DPOvsRLHFViz() {
  const [method, setMethod] = useState('rlhf')

  const rlhfSteps = ['Collect preferences', 'Train reward model', 'Run PPO', 'Iterate']
  const dpoSteps = ['Collect preferences', 'Train policy directly', 'Done']

  const steps = method === 'rlhf' ? rlhfSteps : dpoSteps
  const color = method === 'rlhf' ? '#f97316' : '#7c3aed'

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">RLHF vs DPO Pipeline</h3>
      <div className="flex items-center gap-4 mb-3">
        <button onClick={() => setMethod('rlhf')} className={`px-3 py-1 rounded text-sm ${method === 'rlhf' ? 'bg-orange-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800'}`}>RLHF</button>
        <button onClick={() => setMethod('dpo')} className={`px-3 py-1 rounded text-sm ${method === 'dpo' ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800'}`}>DPO</button>
      </div>
      <div className="flex items-center gap-2 justify-center flex-wrap">
        {steps.map((step, i) => (
          <div key={i} className="flex items-center">
            <div className="px-3 py-2 rounded-lg text-xs font-medium text-center" style={{ backgroundColor: color + '22', border: `1.5px solid ${color}`, color }}>
              {step}
            </div>
            {i < steps.length - 1 && <span className="mx-1 text-gray-400">&#8594;</span>}
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">DPO eliminates the reward model and RL loop entirely</p>
    </div>
  )
}

export default function DPO() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Direct Preference Optimization (DPO) bypasses reward modeling and RL entirely, optimizing
        the policy directly on preference data. It is analytically equivalent to RLHF with a specific
        reward parameterization.
      </p>

      <TheoremBlock title="DPO Key Insight" id="dpo-derivation">
        <p>The optimal policy under the KL-constrained RLHF objective has a closed form:</p>
        <BlockMath math="\pi^*(y|x) = \frac{1}{Z(x)} \pi_\text{ref}(y|x) \exp\!\left(\frac{r(x,y)}{\beta}\right)" />
        <p className="mt-2">Inverting this gives the implicit reward:</p>
        <BlockMath math="r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)" />
      </TheoremBlock>

      <DefinitionBlock title="DPO Loss Function">
        <p>Substituting the implicit reward into the Bradley-Terry model, the partition function cancels:</p>
        <BlockMath math="\mathcal{L}_\text{DPO}(\theta) = -\mathbb{E}_{(x,y_w,y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}\right)\right]" />
        <p className="mt-2">This is a simple classification loss that can be optimized with standard SGD.</p>
      </DefinitionBlock>

      <DPOvsRLHFViz />

      <ExampleBlock title="DPO Advantages">
        <p><strong>Simplicity</strong>: No reward model training, no RL loop, no value function.</p>
        <p><strong>Stability</strong>: Standard supervised learning, no PPO hyperparameters to tune.</p>
        <p><strong>Efficiency</strong>: Only needs forward/backward passes through the policy model.</p>
        <p>DPO matches or exceeds RLHF performance on summarization and dialogue benchmarks.</p>
      </ExampleBlock>

      <PythonCode
        title="DPO Training Implementation"
        code={`import torch
import torch.nn.functional as F

def dpo_loss(policy, ref_policy, chosen_ids, rejected_ids,
             chosen_mask, rejected_mask, beta=0.1):
    # Compute log probabilities under both models
    pi_chosen = policy.log_probs(chosen_ids, chosen_mask)
    pi_rejected = policy.log_probs(rejected_ids, rejected_mask)

    with torch.no_grad():
        ref_chosen = ref_policy.log_probs(chosen_ids, chosen_mask)
        ref_rejected = ref_policy.log_probs(rejected_ids, rejected_mask)

    # DPO implicit reward difference
    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)

    # Bradley-Terry loss with implicit rewards
    loss = -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # Useful metrics
    with torch.no_grad():
        reward_margin = (chosen_reward - rejected_reward).mean()
        accuracy = (chosen_reward > rejected_reward).float().mean()
    return loss, {"margin": reward_margin.item(), "acc": accuracy.item()}`}
      />

      <NoteBlock type="note" title="Beyond DPO">
        <p>
          Variants include <strong>IPO</strong> (identity preference optimization, avoids overfitting),
          <strong>KTO</strong> (Kahneman-Tversky optimization, works with binary feedback instead of
          pairs), and <strong>ORPO</strong> (odds ratio preference optimization, combines SFT and
          alignment in one step). The field is evolving rapidly toward simpler alignment methods.
        </p>
      </NoteBlock>
    </div>
  )
}
