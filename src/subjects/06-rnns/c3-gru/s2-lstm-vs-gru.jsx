import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'
import TheoremBlock from '../../../components/content/TheoremBlock.jsx'

function ComparisonTable() {
  const [metric, setMetric] = useState('all')
  const rows = [
    { feature: 'Gates', lstm: '3 (forget, input, output)', gru: '2 (update, reset)', category: 'arch' },
    { feature: 'State vectors', lstm: 'h_t and c_t', gru: 'h_t only', category: 'arch' },
    { feature: 'Parameters (h=256, d=64)', lstm: '~328K', gru: '~246K', category: 'cost' },
    { feature: 'Speed (relative)', lstm: '1.0x', gru: '~1.3x faster', category: 'cost' },
    { feature: 'Long-range deps', lstm: 'Excellent', gru: 'Good', category: 'perf' },
    { feature: 'Small datasets', lstm: 'May overfit', gru: 'Better generalization', category: 'perf' },
    { feature: 'Speech recognition', lstm: 'Preferred', gru: 'Comparable', category: 'perf' },
    { feature: 'Machine translation', lstm: 'Preferred', gru: 'Comparable', category: 'perf' },
  ]
  const filtered = metric === 'all' ? rows : rows.filter(r => r.category === metric)

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">LSTM vs GRU Comparison</h3>
      <div className="flex gap-2 mb-3 flex-wrap">
        {[['all', 'All'], ['arch', 'Architecture'], ['cost', 'Cost'], ['perf', 'Performance']].map(([k, l]) => (
          <button key={k} onClick={() => setMetric(k)}
            className={`px-3 py-1 rounded-lg text-sm ${metric === k ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {l}
          </button>
        ))}
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-gray-200 dark:border-gray-700">
            <th className="text-left py-2 text-gray-600 dark:text-gray-400">Feature</th>
            <th className="text-left py-2 text-violet-600 dark:text-violet-400">LSTM</th>
            <th className="text-left py-2 text-violet-600 dark:text-violet-400">GRU</th>
          </tr>
        </thead>
        <tbody>
          {filtered.map((r, i) => (
            <tr key={i} className="border-b border-gray-100 dark:border-gray-800">
              <td className="py-2 text-gray-700 dark:text-gray-300 font-medium">{r.feature}</td>
              <td className="py-2 text-gray-600 dark:text-gray-400">{r.lstm}</td>
              <td className="py-2 text-gray-600 dark:text-gray-400">{r.gru}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default function LSTMvsGRU() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        The choice between LSTM and GRU depends on the task, dataset size, and computational
        constraints. Empirical evidence shows neither consistently dominates the other, but
        understanding their trade-offs helps make informed decisions.
      </p>

      <ComparisonTable />

      <TheoremBlock title="Empirical Findings (Chung et al., 2014)" id="lstm-gru-empirical">
        <p>
          Across music modeling, speech signal modeling, and NLP tasks, the GRU and LSTM
          achieve comparable performance. The GRU tends to converge faster due to fewer
          parameters. However, on tasks requiring very long-range dependencies (sequences
          of 100+ steps), the LSTM's separate cell state provides a modest advantage.
        </p>
      </TheoremBlock>

      <ExampleBlock title="Decision Framework">
        <p>Use <strong>LSTM</strong> when:</p>
        <ul className="list-disc ml-6 mt-1 space-y-1">
          <li>Sequences are very long (hundreds of steps)</li>
          <li>You have sufficient compute and data</li>
          <li>The task requires fine-grained memory control</li>
        </ul>
        <p className="mt-2">Use <strong>GRU</strong> when:</p>
        <ul className="list-disc ml-6 mt-1 space-y-1">
          <li>Speed and efficiency matter</li>
          <li>The dataset is small (fewer parameters = less overfitting)</li>
          <li>Sequences are moderate length</li>
        </ul>
      </ExampleBlock>

      <PythonCode
        title="Benchmarking LSTM vs GRU"
        code={`import torch
import torch.nn as nn
import time

def benchmark(model, x, n_runs=100):
    # Warmup
    for _ in range(10):
        model(x)
    torch.cuda.synchronize() if x.is_cuda else None
    start = time.time()
    for _ in range(n_runs):
        model(x)
    torch.cuda.synchronize() if x.is_cuda else None
    return (time.time() - start) / n_runs * 1000  # ms

input_size, hidden_size, seq_len = 64, 256, 100
lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
gru = nn.GRU(input_size, hidden_size, num_layers=2, batch_first=True)

x = torch.randn(32, seq_len, input_size)

lstm_time = benchmark(lstm, x)
gru_time = benchmark(gru, x)

print(f"LSTM: {lstm_time:.2f} ms/batch")
print(f"GRU:  {gru_time:.2f} ms/batch")
print(f"GRU speedup: {lstm_time/gru_time:.2f}x")

# Typical result: GRU is 1.2-1.4x faster than LSTM`}
      />

      <NoteBlock type="note" title="Modern Perspective">
        <p>
          With the rise of Transformers, the LSTM vs GRU debate has become less central. However,
          both remain highly relevant for on-device inference, streaming applications, and tasks
          where the <InlineMath math="O(n^2)" /> attention cost of Transformers is prohibitive. In
          practice, <strong>try both and pick based on validation performance</strong>.
        </p>
      </NoteBlock>
    </div>
  )
}
