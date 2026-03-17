import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PreferenceViz() {
  const [scores, setScores] = useState([3.2, 1.8, 4.1, 2.5])
  const responses = ['Response A', 'Response B', 'Response C', 'Response D']
  const maxScore = Math.max(...scores)
  const expScores = scores.map(s => Math.exp(s))
  const sumExp = expScores.reduce((a, b) => a + b, 0)
  const probs = expScores.map(e => e / sumExp)

  const updateScore = (i, val) => {
    const newScores = [...scores]
    newScores[i] = val
    setScores(newScores)
  }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Reward Model Scores</h3>
      <div className="space-y-2">
        {responses.map((resp, i) => (
          <div key={i} className="flex items-center gap-3">
            <span className="text-sm text-gray-600 dark:text-gray-400 w-24">{resp}</span>
            <input type="range" min={0} max={5} step={0.1} value={scores[i]} onChange={e => updateScore(i, parseFloat(e.target.value))} className="w-24 accent-violet-500" />
            <div className="w-32 h-5 bg-gray-200 dark:bg-gray-700 rounded overflow-hidden">
              <div className="h-full bg-violet-500 rounded" style={{ width: `${probs[i] * 100}%` }} />
            </div>
            <span className="text-xs text-violet-600 dark:text-violet-400 font-mono w-16">r={scores[i].toFixed(1)} ({(probs[i] * 100).toFixed(0)}%)</span>
          </div>
        ))}
      </div>
      <p className="text-center text-xs text-gray-500 mt-2">Bradley-Terry probabilities from reward scores</p>
    </div>
  )
}

export default function RewardModeling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Reward modeling learns a scalar reward function from human preference comparisons,
        enabling RL to optimize for objectives that are hard to specify programmatically.
      </p>

      <DefinitionBlock title="Bradley-Terry Preference Model">
        <p>Given two responses <InlineMath math="y_w" /> (preferred) and <InlineMath math="y_l" /> (dispreferred)
        to a prompt <InlineMath math="x" />, the probability of the observed preference is:</p>
        <BlockMath math="P(y_w \succ y_l | x) = \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)" />
        <p className="mt-2">where <InlineMath math="r_\theta" /> is the reward model and <InlineMath math="\sigma" /> is the sigmoid function.</p>
      </DefinitionBlock>

      <DefinitionBlock title="Reward Model Training Loss">
        <BlockMath math="\mathcal{L}(\theta) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\!\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]" />
        <p className="mt-2">This is equivalent to binary cross-entropy on pairwise comparisons.</p>
      </DefinitionBlock>

      <PreferenceViz />

      <TheoremBlock title="Reward Model Architecture" id="rm-architecture">
        <p>Typically, the reward model is a pretrained language model with the unembedding head
        replaced by a scalar projection:</p>
        <BlockMath math="r_\theta(x, y) = \text{Linear}\!\left(\text{LLM}_\theta([x; y])_{\text{last}}\right) \in \mathbb{R}" />
        <p className="mt-2">The model is initialized from the SFT checkpoint to preserve language understanding.</p>
      </TheoremBlock>

      <ExampleBlock title="Data Collection Pipeline">
        <p>1. Sample K responses from the SFT model for each prompt.</p>
        <p>2. Human annotators rank or compare pairs of responses.</p>
        <p>3. From K responses, extract <InlineMath math="\binom{K}{2}" /> pairwise comparisons.</p>
        <p>InstructGPT used K=4 responses per prompt, giving 6 pairs each.</p>
      </ExampleBlock>

      <PythonCode
        title="Reward Model Training"
        code={`import torch
import torch.nn as nn

class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model  # pretrained LLM
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]  # last token
        return self.reward_head(last_hidden).squeeze(-1)

def reward_model_loss(rm, chosen_ids, chosen_mask, rejected_ids, rejected_mask):
    r_chosen = rm(chosen_ids, chosen_mask)
    r_rejected = rm(rejected_ids, rejected_mask)
    loss = -torch.log(torch.sigmoid(r_chosen - r_rejected)).mean()
    accuracy = (r_chosen > r_rejected).float().mean()
    return loss, accuracy`}
      />

      <NoteBlock type="note" title="Reward Hacking">
        <p>
          The RL policy may find ways to achieve high reward scores that do not correspond
          to genuinely better outputs. This is <strong>reward hacking</strong>. Mitigations include
          KL penalties against the reference policy, reward model ensembles, and iterative
          data collection with updated policies.
        </p>
      </NoteBlock>
    </div>
  )
}
