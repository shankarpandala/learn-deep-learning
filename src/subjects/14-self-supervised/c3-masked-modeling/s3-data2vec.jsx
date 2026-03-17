import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ModalityViz() {
  const [modality, setModality] = useState('vision')
  const modalities = {
    vision: { input: 'Image patches', masking: '40-75% random patches', target: 'Teacher ViT features', examples: 'ImageNet, COCO' },
    speech: { input: 'Audio waveform frames', masking: 'Contiguous spans of frames', target: 'Teacher wav2vec features', examples: 'LibriSpeech' },
    text: { input: 'Subword tokens', masking: '15% random tokens (BERT-style)', target: 'Teacher Transformer features', examples: 'Books, Wikipedia' },
  }
  const m = modalities[modality]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-2 text-base font-bold text-gray-800 dark:text-gray-200">data2vec: Unified Across Modalities</h3>
      <div className="flex gap-2 mb-3">
        {Object.keys(modalities).map(key => (
          <button key={key} onClick={() => setModality(key)}
            className={`px-3 py-1 rounded-full text-xs font-medium capitalize transition-colors ${modality === key ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 hover:bg-gray-200 dark:bg-gray-800 dark:text-gray-400'}`}>
            {key}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
        <div><span className="text-gray-500">Input:</span> <span className="font-medium">{m.input}</span></div>
        <div><span className="text-gray-500">Masking:</span> <span className="font-medium">{m.masking}</span></div>
        <div><span className="text-gray-500">Target:</span> <span className="font-medium text-violet-600">{m.target}</span></div>
        <div><span className="text-gray-500">Data:</span> <span className="font-medium">{m.examples}</span></div>
      </div>
      <p className="text-xs text-violet-600 text-center mt-2 font-medium">
        Same algorithm, same objective, same architecture backbone across all modalities
      </p>
    </div>
  )
}

export default function Data2Vec() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        data2vec provides a unified self-supervised learning framework across vision, speech, and text.
        Instead of predicting modality-specific targets (pixels, tokens, waveforms), it predicts
        contextualized representations from a teacher network.
      </p>

      <DefinitionBlock title="data2vec Objective">
        <p>
          A student model with masked input predicts the representations from an EMA teacher
          that sees the full unmasked input:
        </p>
        <BlockMath math="\mathcal{L} = \frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}} \left\| f_\theta^{\text{student}}(\tilde{\mathbf{x}})_i - \bar{f}_\xi^{\text{teacher}}(\mathbf{x})_i \right\|^2" />
        <p className="mt-2">
          The teacher target <InlineMath math="\bar{f}_\xi" /> is the average of the top <InlineMath math="K" /> transformer
          layers, followed by instance normalization. The teacher is updated via EMA:
          <InlineMath math="\xi \leftarrow \tau \xi + (1-\tau)\theta" />.
        </p>
      </DefinitionBlock>

      <ModalityViz />

      <TheoremBlock title="Why Predict Representations?" id="pred-repr">
        <p>
          Teacher representations capture contextual information from the full input:
        </p>
        <BlockMath math="\mathbf{y}_i = \text{Normalize}\left(\frac{1}{K}\sum_{l=L-K+1}^{L} \mathbf{h}_i^{(l)}\right)" />
        <p className="mt-2">
          Unlike pixel/token targets, these representations are:
          (1) inherently high-level and semantic,
          (2) context-dependent (same patch has different targets in different images), and
          (3) modality-agnostic in their loss formulation.
        </p>
      </TheoremBlock>

      <ExampleBlock title="data2vec 2.0: Efficiency Improvements">
        <p>
          data2vec 2.0 introduces several efficiency improvements: (1) the teacher processes
          each sample only once and caches targets, (2) multi-mask training applies multiple
          different masks to the same teacher encoding, and (3) convolutional decoders replace
          transformer decoders. This yields 2-16x speedups over data2vec 1.0.
        </p>
      </ExampleBlock>

      <PythonCode
        title="data2vec Core: Teacher Targets and Student Loss"
        code={`import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class Data2Vec(nn.Module):
    def __init__(self, encoder, num_layers=12, top_k=8, tau=0.999):
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        self.top_k = top_k
        self.tau = tau

        for p in self.teacher.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update_teacher(self):
        for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
            pt.data = self.tau * pt.data + (1 - self.tau) * ps.data

    @torch.no_grad()
    def get_teacher_targets(self, x):
        # Get representations from top-K layers of teacher
        # (Simplified: in practice, hook into intermediate layers)
        teacher_out = self.teacher(x)  # assume returns list of layer outputs
        # Average top-K layers
        top_k_layers = teacher_out[-self.top_k:]
        target = torch.stack(top_k_layers).mean(dim=0)
        # Instance normalization
        target = F.layer_norm(target, target.shape[-1:])
        return target

    def forward(self, x, mask):
        self.update_teacher()
        targets = self.get_teacher_targets(x)  # (B, N, D)

        # Student sees masked input
        student_out = self.student(x, mask=mask)  # (B, N, D)

        # Loss only on masked positions
        loss = F.smooth_l1_loss(
            student_out[mask],
            targets[mask].detach(),
        )
        return loss

print("data2vec: predict teacher representations, not raw inputs")
print("Unified framework: same code for vision, speech, and text")
print("Key: EMA teacher + top-K layer averaging + instance norm")`}
      />

      <NoteBlock type="note" title="I-JEPA: Predicting in Representation Space">
        <p>
          I-JEPA (Image Joint Embedding Predictive Architecture) by LeCun et al. extends the idea of
          predicting representations: a predictor network maps context patch embeddings to predict
          target patch embeddings, without pixel-level reconstruction. This avoids the bias toward
          low-level features and produces representations that excel at semantic tasks.
        </p>
      </NoteBlock>
    </div>
  )
}
