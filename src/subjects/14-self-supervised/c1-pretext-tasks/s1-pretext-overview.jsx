import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function PretextTaskGrid() {
  const [selectedTask, setSelectedTask] = useState('rotation')
  const tasks = {
    rotation: { label: 'Rotation Prediction', desc: 'Predict which of 4 rotations (0, 90, 180, 270) was applied. Forces understanding of object orientation and canonical poses.', classes: 4 },
    jigsaw: { label: 'Jigsaw Puzzle', desc: 'Predict the permutation of shuffled image patches. Learns spatial relationships between parts.', classes: '30-1000' },
    colorization: { label: 'Colorization', desc: 'Predict color channels from grayscale input. Requires semantic understanding (sky is blue, grass is green).', classes: 'continuous' },
    inpainting: { label: 'Inpainting', desc: 'Reconstruct missing image regions. Forces understanding of context and object structure.', classes: 'continuous' },
  }
  const task = tasks[selectedTask]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">Pretext Tasks</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {Object.entries(tasks).map(([key, t]) => (
          <button key={key} onClick={() => setSelectedTask(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${selectedTask === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
        <p className="text-sm text-gray-700 dark:text-gray-300">{task.desc}</p>
        <p className="text-xs text-violet-600 mt-1">Output classes: {task.classes}</p>
      </div>
    </div>
  )
}

export default function PretextOverview() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Self-supervised learning creates supervision signals from the data itself, eliminating the
        need for manual labels. Pretext tasks exploit the inherent structure of images to train
        representations that transfer to downstream tasks.
      </p>

      <DefinitionBlock title="Pretext Task">
        <p>
          A pretext task is a surrogate objective where labels are derived automatically from the input.
          The model learns representations <InlineMath math="f_\theta(\mathbf{x})" /> useful for downstream tasks
          by solving the pretext task:
        </p>
        <BlockMath math="\min_\theta \mathbb{E}_{\mathbf{x} \sim \mathcal{D}}\left[\mathcal{L}(g_\phi(f_\theta(T(\mathbf{x}))), y_T)\right]" />
        <p className="mt-2">
          where <InlineMath math="T" /> is a transformation, <InlineMath math="y_T" /> is the pseudo-label, and
          <InlineMath math="g_\phi" /> is a task-specific head (discarded after pre-training).
        </p>
      </DefinitionBlock>

      <PretextTaskGrid />

      <ExampleBlock title="Rotation Prediction (RotNet)">
        <p>
          Apply one of four rotations <InlineMath math="r \in \{0°, 90°, 180°, 270°\}" /> and train a
          classifier to predict which rotation was applied. This forces the network to understand
          object semantics — recognizing that a dog is upside-down requires knowing what a dog
          looks like right-side up.
        </p>
      </ExampleBlock>

      <PythonCode
        title="Rotation Prediction Pretext Task"
        code={`import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class RotNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # e.g., ResNet without final FC
        self.rotation_head = nn.Linear(512, 4)  # 4 rotation classes

    def forward(self, x):
        features = self.backbone(x)
        return self.rotation_head(features)

def create_rotation_batch(images):
    """Create self-supervised batch with 4 rotations per image."""
    rotated, labels = [], []
    for img in images:
        for r, angle in enumerate([0, 90, 180, 270]):
            rotated.append(TF.rotate(img, angle))
            labels.append(r)
    return torch.stack(rotated), torch.tensor(labels)

# Training
# images = next(dataloader)  # unlabeled images
# x_rot, y_rot = create_rotation_batch(images)
# logits = model(x_rot)
# loss = nn.CrossEntropyLoss()(logits, y_rot)

# After pre-training: discard rotation_head, fine-tune backbone
print("RotNet: 4-class classification on rotation angle")
print("Key insight: semantic understanding is needed to detect orientation")`}
      />

      <NoteBlock type="note" title="Jigsaw Puzzles (Noroozi & Favaro)">
        <p>
          Split an image into a 3x3 grid of patches, shuffle them, and train the network to predict
          the permutation. From 9! = 362,880 possible permutations, a subset of ~1000 maximally
          different permutations is selected. This teaches spatial reasoning and part-whole relationships.
        </p>
      </NoteBlock>

      <NoteBlock type="note" title="Limitations of Pretext Tasks">
        <p>
          Hand-designed pretext tasks have a fundamental limitation: the learned features may be
          biased toward solving the specific proxy task rather than learning general representations.
          Contrastive and masked modeling approaches (next sections) largely supersede these methods.
        </p>
      </NoteBlock>
    </div>
  )
}
