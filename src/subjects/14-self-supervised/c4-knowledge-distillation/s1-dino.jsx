import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function CenteringViz() {
  const [centerStrength, setCenterStrength] = useState(0.9)
  const dims = 8
  const rawValues = [0.6, 0.1, 0.05, 0.02, 0.02, 0.01, 0.01, 0.19]
  const center = rawValues.map(v => v * (1 - centerStrength) + (1 / dims) * centerStrength)
  const sum = center.reduce((a, b) => a + b, 0)
  const normalized = center.map(v => v / sum)
  const maxVal = Math.max(...normalized)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Centering + Sharpening Effect</h3>
      <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400 mb-3">
        Centering strength: {centerStrength.toFixed(2)}
        <input type="range" min={0} max={0.99} step={0.01} value={centerStrength} onChange={e => setCenterStrength(parseFloat(e.target.value))} className="w-40 accent-violet-500" />
      </label>
      <div className="flex gap-3 justify-center">
        <div>
          <p className="text-[10px] text-gray-500 text-center mb-1">Without centering</p>
          <div className="flex gap-1 items-end h-20">
            {rawValues.map((v, i) => (
              <div key={i} className="w-5 bg-gray-300 rounded-t" style={{ height: `${v / 0.6 * 70}px` }} />
            ))}
          </div>
        </div>
        <div>
          <p className="text-[10px] text-violet-500 text-center mb-1">With centering</p>
          <div className="flex gap-1 items-end h-20">
            {normalized.map((v, i) => (
              <div key={i} className="w-5 bg-violet-400 rounded-t" style={{ height: `${v / maxVal * 70}px` }} />
            ))}
          </div>
        </div>
      </div>
      <p className="text-xs text-gray-500 text-center mt-1">
        Centering prevents one dimension from dominating (collapse to uniform = prevented)
      </p>
    </div>
  )
}

export default function DINO() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        DINO (Self-Distillation with No Labels) learns visual features through self-distillation
        between student and teacher networks. Its attention maps exhibit remarkable emergent
        properties, capturing object boundaries without any segmentation supervision.
      </p>

      <DefinitionBlock title="DINO Framework">
        <p>Student and teacher networks produce probability distributions via softmax with temperatures:</p>
        <BlockMath math="P_s(x)^{(i)} = \frac{\exp(g_{\theta_s}(x)^{(i)} / \tau_s)}{\sum_k \exp(g_{\theta_s}(x)^{(k)} / \tau_s)}" />
        <p className="mt-2">The loss minimizes cross-entropy between student and (centered, sharpened) teacher:</p>
        <BlockMath math="\mathcal{L} = -\sum_{\substack{x \in \{x_1^g, x_2^g\} \\ x' \neq x}} \sum_i P_t(x)^{(i)} \log P_s(x')^{(i)}" />
        <p className="mt-1">
          Teacher uses lower temperature (<InlineMath math="\tau_t = 0.04" />) for sharper outputs;
          student uses higher (<InlineMath math="\tau_s = 0.1" />).
        </p>
      </DefinitionBlock>

      <CenteringViz />

      <TheoremBlock title="Multi-Crop Strategy" id="multi-crop">
        <p>DINO uses asymmetric crops to create a local-to-global correspondence:</p>
        <ul className="list-disc ml-5 mt-2 space-y-1">
          <li><strong>2 global views</strong> (224x224, covering >50% of image): processed by both student and teacher</li>
          <li><strong>N local views</strong> (96x96, covering &lt;50%): processed only by student</li>
        </ul>
        <p className="mt-2">
          The student must predict the teacher's global view output from local crops, encouraging
          learning of semantic features that generalize across spatial scales.
        </p>
      </TheoremBlock>

      <PythonCode
        title="DINO Loss with Centering"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F

class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # student_output: list of (B, D) for each crop
        # teacher_output: list of (B, D) for global crops only
        s_out = [s / self.student_temp for s in student_output]
        t_out = [(t - self.center) / self.teacher_temp for t in teacher_output]

        # Softmax
        s_probs = [F.log_softmax(s, dim=-1) for s in s_out]
        t_probs = [F.softmax(t, dim=-1).detach() for t in t_out]

        # Cross-entropy: each student crop vs each teacher global crop
        total_loss = 0
        n_loss_terms = 0
        for t in t_probs:
            for s in s_probs:
                loss = -torch.sum(t * s, dim=-1).mean()
                total_loss += loss
                n_loss_terms += 1

        # Update center with EMA
        with torch.no_grad():
            batch_center = torch.cat(teacher_output).mean(dim=0, keepdim=True)
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

        return total_loss / n_loss_terms

criterion = DINOLoss(out_dim=256)
student_out = [torch.randn(32, 256) for _ in range(8)]  # 2 global + 6 local
teacher_out = [torch.randn(32, 256) for _ in range(2)]   # 2 global only
loss = criterion(student_out, teacher_out)
print(f"DINO loss: {loss.item():.3f}")`}
      />

      <ExampleBlock title="Emergent Object Segmentation">
        <p>
          DINO ViT attention maps spontaneously learn to segment objects without any segmentation
          labels. The [CLS] token's self-attention in the last layer highlights foreground objects
          with sharp boundaries. This emergent property makes DINO features excellent for dense
          prediction tasks like semantic segmentation and object detection.
        </p>
      </ExampleBlock>

      <NoteBlock type="note" title="Why Self-Distillation Works">
        <p>
          The combination of centering (prevents collapse to uniform), sharpening (prevents
          collapse to one-hot), momentum teacher (provides stable targets), and multi-crop
          (creates difficulty asymmetry) creates a self-reinforcing learning signal. The teacher
          slowly improves, providing increasingly informative targets for the student.
        </p>
      </NoteBlock>
    </div>
  )
}
