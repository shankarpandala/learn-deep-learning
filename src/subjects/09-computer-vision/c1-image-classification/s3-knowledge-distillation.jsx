import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function TemperatureDemo() {
  const [temp, setTemp] = useState(1)
  const logits = [5.0, 2.0, 0.5, -1.0]
  const softmax = (logits, T) => {
    const exps = logits.map(l => Math.exp(l / T))
    const sum = exps.reduce((a, b) => a + b, 0)
    return exps.map(e => e / sum)
  }
  const probs = softmax(logits, temp)
  const labels = ['Cat', 'Dog', 'Bird', 'Fish']
  const maxProb = Math.max(...probs)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-3 text-base font-bold text-gray-800 dark:text-gray-200">Temperature Scaling Effect</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-4">
        T = {temp.toFixed(1)}
        <input type="range" min={0.1} max={10} step={0.1} value={temp}
          onChange={e => setTemp(parseFloat(e.target.value))} className="w-48 accent-violet-500" />
      </label>
      <div className="flex items-end gap-3 justify-center h-32">
        {probs.map((p, i) => (
          <div key={i} className="flex flex-col items-center">
            <span className="text-xs text-gray-500 mb-1">{(p * 100).toFixed(1)}%</span>
            <div className="w-12 bg-violet-500 rounded-t" style={{ height: `${(p / maxProb) * 100}px` }} />
            <span className="text-xs mt-1 text-gray-600 dark:text-gray-400">{labels[i]}</span>
          </div>
        ))}
      </div>
      <p className="mt-2 text-center text-xs text-gray-500">Higher T produces softer probability distributions</p>
    </div>
  )
}

export default function KnowledgeDistillation() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Knowledge distillation trains a smaller student model to mimic a larger teacher model,
        transferring "dark knowledge" encoded in the teacher's soft probability outputs.
      </p>

      <DefinitionBlock title="Distillation Loss">
        <p>
          The student is trained with a combination of hard label loss and soft target loss:
        </p>
        <BlockMath math="\mathcal{L} = (1 - \alpha)\,\mathcal{L}_{\text{CE}}(y, \hat{y}_s) + \alpha\,T^2\,\text{KL}\!\left(\sigma\!\left(\frac{z_t}{T}\right) \| \sigma\!\left(\frac{z_s}{T}\right)\right)" />
        <p className="mt-2">
          where <InlineMath math="T" /> is the temperature, <InlineMath math="\alpha" /> balances
          the two losses, and <InlineMath math="z_t, z_s" /> are teacher/student logits.
        </p>
      </DefinitionBlock>

      <TemperatureDemo />

      <TheoremBlock title="Dark Knowledge" id="dark-knowledge">
        <p>
          The soft targets from the teacher encode inter-class similarities. At high temperature:
        </p>
        <BlockMath math="\frac{\partial}{\partial z_i}\sigma(z/T)_i \approx \frac{1}{T \cdot C}\left(1 + \frac{z_i}{T} - \frac{\bar{z}}{T}\right)" />
        <p className="mt-1">
          This reveals that soft targets carry gradient information proportional to logit
          differences, not just class labels.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Compression Ratios">
        <p>Typical distillation results on ImageNet:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li>Teacher: ResNet-152 (60M params, 78.3% top-1)</li>
          <li>Student: ResNet-18 (11M params, 71.5% alone, 73.2% distilled)</li>
          <li>5.5x compression with only 5.1% accuracy drop</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Knowledge Distillation in PyTorch"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, T=4.0, alpha=0.7):
        super().__init__()
        self.T = T
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Hard label loss
        hard_loss = self.ce(student_logits, labels)

        # Soft target loss (KL divergence)
        soft_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1)
        soft_loss = F.kl_div(soft_student, soft_teacher,
                             reduction='batchmean') * (self.T ** 2)

        return (1 - self.alpha) * hard_loss + self.alpha * soft_loss

# Training loop
teacher.eval()
criterion = DistillationLoss(T=4.0, alpha=0.7)
for images, labels in loader:
    with torch.no_grad():
        teacher_logits = teacher(images)
    student_logits = student(images)
    loss = criterion(student_logits, teacher_logits, labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()`}
      />

      <NoteBlock type="note" title="Feature Distillation">
        <p>
          Beyond logit-level distillation, intermediate feature maps can also be matched.
          Methods like FitNets align student hidden layers to teacher layers using
          <InlineMath math="\mathcal{L}_{\text{hint}} = \|W_s h_s - h_t\|^2" />.
          This provides richer supervision and often improves student performance further.
        </p>
      </NoteBlock>
    </div>
  )
}
