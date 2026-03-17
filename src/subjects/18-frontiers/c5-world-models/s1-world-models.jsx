import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function WorldModelLoop() {
  const [step, setStep] = useState(0)
  const steps = [
    { phase: 'Observe', desc: 'Encode current observation into latent state z_t', color: 'bg-violet-100 dark:bg-violet-900/20' },
    { phase: 'Imagine', desc: 'Use learned dynamics model to predict future: z_{t+1} = f(z_t, a_t)', color: 'bg-violet-200 dark:bg-violet-900/30' },
    { phase: 'Evaluate', desc: 'Predict reward from imagined state: r_{t+1} = R(z_{t+1})', color: 'bg-violet-300 dark:bg-violet-900/40' },
    { phase: 'Plan', desc: 'Search over action sequences in imagination to maximize expected reward', color: 'bg-violet-400 dark:bg-violet-800/40' },
    { phase: 'Act', desc: 'Execute best action from planning, observe real outcome, update model', color: 'bg-violet-500/20 dark:bg-violet-700/30' },
  ]
  const s = steps[step]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">World Model Planning Loop</h3>
      <div className="flex gap-1 mb-3">
        {steps.map((st, i) => (
          <button key={i} onClick={() => setStep(i)}
            className={`flex-1 px-2 py-1 rounded text-xs transition ${step === i ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            {st.phase}
          </button>
        ))}
      </div>
      <div className={`p-3 rounded-lg ${s.color} text-sm`}>
        <p className="font-medium text-gray-700 dark:text-gray-300">{s.phase}</p>
        <p className="text-gray-600 dark:text-gray-400 mt-1">{s.desc}</p>
      </div>
      <div className="flex mt-2 gap-1">
        {steps.map((_, i) => <div key={i} className={`h-1 flex-1 rounded ${i <= step ? 'bg-violet-500' : 'bg-gray-200 dark:bg-gray-700'}`} />)}
      </div>
    </div>
  )
}

export default function WorldModels() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        World models learn a compressed representation of environment dynamics, enabling agents
        to plan by "imagining" future states. This approach is sample-efficient because the
        agent can practice in its learned model rather than the real environment.
      </p>

      <DefinitionBlock title="World Model Components">
        <p>A world model consists of three learned components operating in latent space:</p>
        <BlockMath math="\text{Encoder: } z_t = q(o_t), \quad \text{Dynamics: } z_{t+1} = f(z_t, a_t), \quad \text{Reward: } r_t = R(z_t)" />
        <p className="mt-2">The agent can unroll the dynamics model to simulate trajectories:</p>
        <BlockMath math="\hat{\tau} = (z_t, a_t, \hat{z}_{t+1}, \hat{r}_{t+1}, a_{t+1}, \hat{z}_{t+2}, \hat{r}_{t+2}, \ldots)" />
        <p className="mt-1">Planning is then optimization over action sequences in this imagined trajectory.</p>
      </DefinitionBlock>

      <WorldModelLoop />

      <ExampleBlock title="DreamerV3: Universal World Model">
        <p>DreamerV3 (Hafner et al., 2023) learns world models that transfer across diverse domains:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>First algorithm to collect diamonds in Minecraft from scratch (no human data)</li>
          <li>Same hyperparameters work across Atari, DMC, DMLab, and Minecraft</li>
          <li>Uses a Recurrent State Space Model (RSSM) with discrete latent variables</li>
          <li>Trains actor and critic entirely in imagination (no real-env policy rollouts)</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Simplified World Model with Latent Dynamics"
        code={`import torch
import torch.nn as nn

class WorldModel(nn.Module):
    """Simplified world model with encoder, dynamics, and reward prediction."""
    def __init__(self, obs_dim=64, latent_dim=128, action_dim=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(), nn.Linear(256, latent_dim))
        self.dynamics = nn.GRUCell(action_dim, latent_dim)
        self.reward_head = nn.Linear(latent_dim, 1)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, obs_dim))

    def encode(self, obs):
        return self.encoder(obs)

    def imagine_step(self, z, action):
        z_next = self.dynamics(action, z)
        reward = self.reward_head(z_next)
        return z_next, reward

    def imagine_trajectory(self, z0, actions):
        """Unroll dynamics model for planning."""
        z = z0
        rewards = []
        states = [z]
        for a in actions:
            z, r = self.imagine_step(z, a)
            rewards.append(r)
            states.append(z)
        return torch.stack(states), torch.cat(rewards)

# Planning in imagination
model = WorldModel()
obs = torch.randn(1, 64)
z = model.encode(obs)

# Plan 10 steps ahead
actions = [torch.randn(1, 4) for _ in range(10)]
imagined_states, imagined_rewards = model.imagine_trajectory(z, actions)
print(f"Imagined {len(actions)} steps: states {imagined_states.shape}")
print(f"Expected return: {imagined_rewards.sum().item():.3f}")`}
      />

      <NoteBlock type="note" title="Video Generation as World Models">
        <p>
          Large video generation models (Sora, Genie) can be viewed as world models that learn
          physics and dynamics from internet video. They predict future visual frames conditioned
          on actions or text, potentially enabling robots and game agents to learn from vast
          video data without environment interaction. The boundary between generative models
          and world models is rapidly blurring.
        </p>
      </NoteBlock>
    </div>
  )
}
