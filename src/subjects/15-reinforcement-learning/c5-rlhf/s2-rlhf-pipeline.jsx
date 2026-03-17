import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PipelineViz() {
  const [stage, setStage] = useState(0)
  const stages = [
    { name: 'Pre-training', desc: 'Train LLM on large text corpus', color: '#7c3aed' },
    { name: 'SFT', desc: 'Fine-tune on demonstrations', color: '#f97316' },
    { name: 'Reward Model', desc: 'Train RM from comparisons', color: '#10b981' },
    { name: 'PPO', desc: 'Optimize policy with RM signal', color: '#f43f5e' },
  ]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">RLHF Pipeline Stages</h3>
      <div className="flex items-center gap-2 mb-4">
        <label className="text-sm text-gray-600 dark:text-gray-400">Stage:</label>
        {stages.map((s, i) => (
          <button key={i} onClick={() => setStage(i)}
            className={`px-3 py-1 rounded text-xs font-medium transition-colors ${i === stage ? 'text-white' : 'text-gray-600 bg-gray-100 dark:bg-gray-800'}`}
            style={i === stage ? { backgroundColor: s.color } : {}}>
            {s.name}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-1 justify-center">
        {stages.map((s, i) => (
          <div key={i} className="flex items-center">
            <div className={`w-20 h-14 rounded-lg flex flex-col items-center justify-center text-xs transition-opacity ${i <= stage ? 'opacity-100' : 'opacity-30'}`}
              style={{ backgroundColor: s.color + '22', border: `2px solid ${s.color}` }}>
              <span style={{ color: s.color }} className="font-bold">{s.name}</span>
            </div>
            {i < 3 && <span className="mx-1 text-gray-400">&#8594;</span>}
          </div>
        ))}
      </div>
      <p className="text-center text-sm text-gray-600 dark:text-gray-400 mt-3 font-medium">{stages[stage].desc}</p>
    </div>
  )
}

export default function RLHFPipeline() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The full RLHF pipeline combines supervised fine-tuning, reward modeling, and PPO-based
        optimization to align language models with human preferences.
      </p>

      <PipelineViz />

      <DefinitionBlock title="KL-Penalized RL Objective">
        <p>The PPO stage optimizes:</p>
        <BlockMath math="\max_\pi \; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi}\left[r_\theta(x, y) - \beta D_\text{KL}(\pi(\cdot|x) \| \pi_\text{ref}(\cdot|x))\right]" />
        <p className="mt-2">The KL penalty prevents the policy from deviating too far from the
        SFT model <InlineMath math="\pi_\text{ref}" />, reducing reward hacking. <InlineMath math="\beta" /> is
        typically 0.01-0.2.</p>
      </DefinitionBlock>

      <TheoremBlock title="Per-Token KL Computation" id="per-token-kl">
        <p>In practice, KL divergence is computed token by token:</p>
        <BlockMath math="D_\text{KL} = \sum_{t=1}^T \left[\log \pi(y_t|x, y_{<t}) - \log \pi_\text{ref}(y_t|x, y_{<t})\right]" />
        <p className="mt-2">This per-token KL is added as a penalty to the reward at each token position,
        shaping the reward to discourage divergence throughout generation.</p>
      </TheoremBlock>

      <ExampleBlock title="InstructGPT Numbers">
        <p><strong>Pre-training</strong>: 175B params, ~300B tokens.</p>
        <p><strong>SFT</strong>: ~13K demonstrations from labelers.</p>
        <p><strong>RM</strong>: ~33K comparisons, trained for 1 epoch to avoid overfitting.</p>
        <p><strong>PPO</strong>: ~31K prompts, trained for a few epochs with KL penalty.</p>
        <p>Result: 1.3B RLHF model preferred over 175B SFT model by human raters.</p>
      </ExampleBlock>

      <PythonCode
        title="RLHF PPO Training Loop (Simplified)"
        code={`import torch

def rlhf_ppo_step(policy, ref_policy, reward_model, optimizer,
                   prompts, kl_coeff=0.1, clip_eps=0.2):
    # 1. Generate responses from current policy
    with torch.no_grad():
        responses, old_log_probs = policy.generate(prompts)
        ref_log_probs = ref_policy.log_probs(prompts, responses)

    # 2. Score with reward model
    with torch.no_grad():
        rewards = reward_model(prompts, responses)

    # 3. Compute per-token KL penalty
    with torch.no_grad():
        kl_penalty = old_log_probs - ref_log_probs  # per token
        shaped_rewards = rewards - kl_coeff * kl_penalty.sum(dim=-1)

    # 4. Compute advantages (simplified - use GAE in practice)
    advantages = shaped_rewards - shaped_rewards.mean()
    advantages = advantages / (advantages.std() + 1e-8)

    # 5. PPO clipped update
    new_log_probs = policy.log_probs(prompts, responses)
    ratio = torch.exp(new_log_probs.sum(-1) - old_log_probs.sum(-1))
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    loss = -torch.min(surr1, surr2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {"loss": loss.item(), "mean_reward": rewards.mean().item()}`}
      />

      <WarningBlock title="Training Instabilities">
        <p>
          RLHF training can be unstable. Common issues: reward model overfitting (use early stopping),
          KL divergence explosion (clip or increase <InlineMath math="\beta" />), mode collapse
          (monitor generation diversity), and reward hacking (verify with held-out evaluators).
        </p>
      </WarningBlock>

      <NoteBlock type="note" title="Beyond InstructGPT">
        <p>
          Modern RLHF pipelines often use iterative training: collect new preferences on the
          latest policy outputs, retrain the reward model, and run another round of PPO.
          Constitutional AI (Anthropic) uses AI feedback instead of human labels for some stages.
        </p>
      </NoteBlock>
    </div>
  )
}
