import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import WarningBlock from '../../../components/content/WarningBlock.jsx'

function OpenProblemsExplorer() {
  const [problem, setProblem] = useState('alignment')
  const problems = {
    alignment: { name: 'AI Safety & Alignment', desc: 'Ensuring AI systems behave according to human values and intentions, especially as capabilities increase. Includes reward hacking, goal misgeneralization, deceptive alignment, and scalable oversight.', difficulty: 'Critical', timeframe: 'Urgent (needed before AGI)' },
    generalization: { name: 'Robust Generalization', desc: 'Current models fail on distributional shift, adversarial inputs, and novel combinations of known concepts. True out-of-distribution generalization remains elusive.', difficulty: 'Fundamental', timeframe: 'Decades-long research program' },
    reasoning: { name: 'Genuine Reasoning', desc: 'Do LLMs truly reason or pattern-match? Can they handle truly novel problems that require logical deduction rather than retrieval of similar training examples?', difficulty: 'Open debate', timeframe: 'Active research area' },
    efficiency: { name: 'Sample Efficiency', desc: 'Humans learn from far fewer examples than neural networks. Achieving human-level sample efficiency would transform the field and reduce compute requirements.', difficulty: 'Hard', timeframe: '5-15 years' },
    understanding: { name: 'Understanding DNNs', desc: 'Why do overparameterized networks generalize? What determines the inductive biases of different architectures? Can we formally characterize what networks learn?', difficulty: 'Theoretical', timeframe: 'Ongoing fundamental research' },
  }
  const p = problems[problem]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Open Problems in Deep Learning</h3>
      <div className="flex gap-1 mb-3 flex-wrap">
        {Object.entries(problems).map(([key, val]) => (
          <button key={key} onClick={() => setProblem(key)}
            className={`px-2 py-1 rounded-lg text-xs transition ${problem === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1">
        <p className="text-gray-600 dark:text-gray-400">{p.desc}</p>
        <div className="flex gap-4 mt-2 text-xs text-gray-500">
          <span>Difficulty: <strong>{p.difficulty}</strong></span>
          <span>Timeframe: <strong>{p.timeframe}</strong></span>
        </div>
      </div>
    </div>
  )
}

export default function OpenProblems() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Despite remarkable progress, deep learning faces fundamental unsolved challenges in
        safety, generalization, reasoning, and theoretical understanding. These open problems
        define the research frontier and will shape the future of the field.
      </p>

      <DefinitionBlock title="The Alignment Problem">
        <p>As AI systems become more capable, ensuring they act according to human intentions becomes critical. The alignment problem has several formal aspects:</p>
        <BlockMath math="\text{Outer alignment: } R_{\text{specified}} \approx R_{\text{intended}}" />
        <BlockMath math="\text{Inner alignment: } R_{\text{learned}} \approx R_{\text{specified}}" />
        <p className="mt-2">Outer alignment asks whether we can correctly specify what we want. Inner alignment asks whether the trained model actually optimizes for the specified objective, even in novel situations.</p>
      </DefinitionBlock>

      <OpenProblemsExplorer />

      <WarningBlock title="Scaling Alone May Not Suffice">
        <p>
          Several fundamental problems are unlikely to be solved by scaling alone: adversarial
          robustness (adversarial examples persist at all scales), formal reasoning (LLMs still
          fail at novel logic puzzles), and alignment (larger models may be harder to align).
          New paradigms, architectures, or training methods may be needed.
        </p>
      </WarningBlock>

      <ExampleBlock title="Key Research Directions">
        <ul className="list-disc list-inside space-y-1">
          <li><strong>Mechanistic interpretability:</strong> Understanding what networks compute, enabling debugging and alignment</li>
          <li><strong>Constitutional AI / RLHF:</strong> Scalable techniques for aligning model behavior with human values</li>
          <li><strong>Neurosymbolic methods:</strong> Combining neural networks with formal logic for reliable reasoning</li>
          <li><strong>Continual learning:</strong> Models that learn from new data without catastrophic forgetting</li>
          <li><strong>Energy efficiency:</strong> Neuromorphic computing, spiking networks, and analog hardware</li>
          <li><strong>Multimodal grounding:</strong> Connecting language to real-world experience and embodiment</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Measuring Alignment: Reward Hacking Detection"
        code={`import torch

def detect_reward_hacking(proxy_rewards, true_rewards, threshold=0.5):
    """Detect when optimizing a proxy reward diverges from true reward.

    Goodhart's Law: "When a measure becomes a target,
    it ceases to be a good measure."

    Args:
        proxy_rewards: rewards from the learned reward model
        true_rewards: rewards from human evaluation (expensive)
        threshold: correlation threshold for alarm
    """
    # Compute correlation between proxy and true rewards
    proxy = torch.tensor(proxy_rewards, dtype=torch.float)
    true = torch.tensor(true_rewards, dtype=torch.float)

    correlation = torch.corrcoef(torch.stack([proxy, true]))[0, 1]

    # Check for Goodhart's Law violation
    # High proxy reward but low true reward = reward hacking
    mean_proxy = proxy.mean().item()
    mean_true = true.mean().item()

    print(f"Proxy-True correlation: {correlation:.3f}")
    print(f"Mean proxy reward: {mean_proxy:.3f}")
    print(f"Mean true reward: {mean_true:.3f}")

    if correlation < threshold:
        print("WARNING: Low correlation — possible reward hacking!")
        print("The model may be exploiting proxy reward without achieving true goal.")
    else:
        print("Rewards appear aligned (but vigilance is still needed).")

# Simulate: model finds exploit in proxy reward
proxy_rewards = [0.9, 0.85, 0.92, 0.95, 0.88, 0.91]  # looks good
true_rewards =  [0.7, 0.3, 0.4, 0.2, 0.5, 0.3]        # actually bad
detect_reward_hacking(proxy_rewards, true_rewards)`}
      />

      <NoteBlock type="note" title="The Road Ahead">
        <p>
          Deep learning has achieved extraordinary results, but significant challenges remain.
          The field is at an inflection point — moving from "can we scale?" to "can we build
          safe, reliable, and efficient AI systems?" Progress on these open problems will
          determine whether AI fulfills its transformative potential responsibly.
        </p>
      </NoteBlock>
    </div>
  )
}
