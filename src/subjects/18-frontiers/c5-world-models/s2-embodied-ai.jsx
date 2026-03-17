import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function EmbodiedAITimeline() {
  const [era, setEra] = useState('foundation')
  const eras = {
    classical: { name: 'Classical (2015-2019)', approach: 'Task-specific RL policies trained in simulation', transfer: 'Sim-to-real gap is a major challenge', examples: 'OpenAI hand manipulation, locomotion policies' },
    language: { name: 'Language-Guided (2020-2023)', approach: 'Language models as planners for robot actions', transfer: 'Natural language bridges sim and real domains', examples: 'SayCan, Code-as-Policies, RT-2' },
    foundation: { name: 'Foundation Models (2023+)', approach: 'Pretrained on diverse robot data, fine-tune for specific tasks', transfer: 'Cross-embodiment generalization', examples: 'RT-X, Octo, pi0' },
  }
  const e = eras[era]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Embodied AI Evolution</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(eras).map(([key, val]) => (
          <button key={key} onClick={() => setEra(key)}
            className={`px-3 py-1 rounded-lg text-sm transition ${era === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {val.name}
          </button>
        ))}
      </div>
      <div className="p-3 rounded-lg bg-violet-50 dark:bg-violet-900/20 text-sm space-y-1">
        <p><strong>Approach:</strong> {e.approach}</p>
        <p><strong>Transfer:</strong> {e.transfer}</p>
        <p><strong>Examples:</strong> {e.examples}</p>
      </div>
    </div>
  )
}

export default function EmbodiedAIRobotics() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Foundation models are transforming robotics by providing robots with broad knowledge
        about the physical world, natural language understanding, and general-purpose reasoning.
        The goal is a single model that can control diverse robots across diverse tasks.
      </p>

      <DefinitionBlock title="Vision-Language-Action (VLA) Models">
        <p>VLA models extend multimodal LLMs to output robot actions, creating an end-to-end policy:</p>
        <BlockMath math="\pi(a_t | o_t, l) = \text{VLA}(\text{image}_t, \text{language\_instruction}, \text{robot\_state}_t)" />
        <p className="mt-2">where <InlineMath math="a_t" /> is the robot action (e.g., end-effector pose), <InlineMath math="o_t" /> is the visual observation, and <InlineMath math="l" /> is the task instruction. The model is typically a pretrained VLM fine-tuned on robot demonstrations with action tokens added to the vocabulary.</p>
      </DefinitionBlock>

      <EmbodiedAITimeline />

      <ExampleBlock title="RT-2: Vision-Language-Action Model">
        <p>RT-2 (Google DeepMind) fine-tunes a 55B VLM (PaLI-X) on robot data:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li>Actions represented as text tokens: "1 128 91 241 1 128 147" (7-DOF)</li>
          <li>Inherits reasoning from VLM pretraining (can handle "pick up the extinct animal")</li>
          <li>3x improvement on unseen objects compared to RT-1 (task-specific model)</li>
          <li>Emergent capabilities: multi-step reasoning about objects, spatial relationships</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Vision-Language-Action Policy (Simplified)"
        code={`import torch
import torch.nn as nn

class SimpleVLAPolicy(nn.Module):
    """Simplified VLA policy: image + language -> robot action."""
    def __init__(self, vision_dim=1024, lang_dim=768, action_dim=7):
        super().__init__()
        # Pretrained encoders (frozen in practice)
        self.vision_proj = nn.Linear(vision_dim, 512)
        self.lang_proj = nn.Linear(lang_dim, 512)
        # Action prediction head (trained on robot data)
        self.action_head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, action_dim),  # 7-DOF: xyz + rotation + gripper
            nn.Tanh(),  # Normalize actions to [-1, 1]
        )

    def forward(self, image_features, language_features):
        v = self.vision_proj(image_features)
        l = self.lang_proj(language_features)
        combined = torch.cat([v, l], dim=-1)
        return self.action_head(combined)

def collect_demonstration(env, policy, instruction):
    """Collect expert demonstration for VLA training."""
    obs = env.reset()
    trajectory = []
    for step in range(100):
        action = policy(obs, instruction)  # Expert or teleoperation
        next_obs, reward, done = env.step(action)
        trajectory.append({
            "image": obs["image"],
            "instruction": instruction,
            "action": action,
        })
        obs = next_obs
        if done:
            break
    return trajectory

# Simulate
vla = SimpleVLAPolicy()
image_feat = torch.randn(1, 1024)   # From ViT
lang_feat = torch.randn(1, 768)     # From language model
action = vla(image_feat, lang_feat)
print(f"Predicted action (7-DOF): {action.squeeze().tolist()[:4]}...")
print(f"Action dim meanings: [dx, dy, dz, rx, ry, rz, gripper]")`}
      />

      <NoteBlock type="note" title="The Data Challenge in Robotics">
        <p>
          Robot data is orders of magnitude scarcer than internet text/images. The Open X-Embodiment
          dataset (RT-X) aggregates data from 22 robots across 21 institutions — still only ~1M
          trajectories vs trillions of text tokens. Simulation, data augmentation, and cross-embodiment
          transfer learning are critical for bridging this data gap.
        </p>
      </NoteBlock>
    </div>
  )
}
