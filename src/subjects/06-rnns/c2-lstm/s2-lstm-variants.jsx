import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function VariantCompare() {
  const [variant, setVariant] = useState('standard')
  const descriptions = {
    standard: { title: 'Standard LSTM', gates: 3, params: '4h(h+d)', note: 'Separate forget, input, output gates with independent candidate.' },
    peephole: { title: 'Peephole LSTM', gates: 3, params: '4h(h+d)+3h^2', note: 'Gates peek at the cell state directly for more informed gating.' },
    coupled: { title: 'Coupled Gate LSTM', gates: 2, params: '3h(h+d)', note: 'Input gate = 1 - forget gate. Fewer parameters, enforces trade-off.' },
    gru: { title: 'GRU', gates: 2, params: '3h(h+d)', note: 'No separate cell state. Merges cell and hidden state into one.' },
  }
  const d = descriptions[variant]

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">LSTM Variant Comparison</h3>
      <div className="flex items-center gap-2 mb-4 flex-wrap">
        {Object.keys(descriptions).map(k => (
          <button key={k} onClick={() => setVariant(k)}
            className={`px-3 py-1 rounded-lg text-sm ${variant === k ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'}`}>
            {descriptions[k].title}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-3 gap-4 text-center text-sm">
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-3 py-2">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">Gates</div>
          <div className="text-lg font-mono">{d.gates}</div>
        </div>
        <div className="rounded-lg bg-violet-50 dark:bg-violet-900/30 px-3 py-2">
          <div className="text-violet-700 dark:text-violet-300 font-semibold">Parameters</div>
          <div className="text-lg font-mono">{d.params}</div>
        </div>
        <div className="col-span-3 text-left text-gray-600 dark:text-gray-400 mt-1">{d.note}</div>
      </div>
    </div>
  )
}

export default function LSTMVariants() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Several LSTM variants modify the gating mechanism to improve performance or reduce
        computational cost. Understanding these variants helps select the right architecture
        for a given task.
      </p>

      <DefinitionBlock title="Peephole Connections">
        <p>
          Peephole LSTMs allow the gates to access the cell state <InlineMath math="c_{t-1}" /> directly:
        </p>
        <BlockMath math="f_t = \sigma(W_f [h_{t-1}, x_t] + W_{pf} \odot c_{t-1} + b_f)" />
        <BlockMath math="i_t = \sigma(W_i [h_{t-1}, x_t] + W_{pi} \odot c_{t-1} + b_i)" />
        <BlockMath math="o_t = \sigma(W_o [h_{t-1}, x_t] + W_{po} \odot c_t + b_o)" />
        <p className="mt-2">
          The diagonal weight matrices <InlineMath math="W_{pf}, W_{pi}, W_{po}" /> add
          <InlineMath math="3h" /> extra parameters, letting the gates make more informed decisions.
        </p>
      </DefinitionBlock>

      <DefinitionBlock title="Coupled Forget-Input Gate">
        <p>Instead of independent forget and input gates, use a single gate:</p>
        <BlockMath math="c_t = f_t \odot c_{t-1} + (1 - f_t) \odot \tilde{c}_t" />
        <p className="mt-2">
          This couples forgetting and updating: the cell can only write new information to
          the extent it forgets old information, reducing parameters by one gate matrix.
        </p>
      </DefinitionBlock>

      <VariantCompare />

      <ExampleBlock title="GRU vs LSTM at a Glance">
        <p>The GRU merges forget and input gates into an <strong>update gate</strong> and eliminates the separate cell state:</p>
        <BlockMath math="z_t = \sigma(W_z [h_{t-1}, x_t])" />
        <BlockMath math="h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t" />
        <p>
          GRU has ~75% of the LSTM parameters and often matches LSTM performance on shorter sequences.
        </p>
      </ExampleBlock>

      <PythonCode
        title="GRU vs LSTM Parameter Count"
        code={`import torch.nn as nn

input_size, hidden_size = 64, 256

lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
gru = nn.GRU(input_size, hidden_size, batch_first=True)

lstm_params = sum(p.numel() for p in lstm.parameters())
gru_params = sum(p.numel() for p in gru.parameters())

print(f"LSTM params: {lstm_params:,}")    # 4 * 256 * (256+64) + 4*256
print(f"GRU  params: {gru_params:,}")     # 3 * 256 * (256+64) + 3*256
print(f"GRU / LSTM:  {gru_params/lstm_params:.2%}")  # ~75%

# Greff et al. (2017) LSTM variant ablation:
# - Forget gate and output activation are most critical
# - Peephole connections provide marginal benefit
# - Coupled gates perform comparably to standard LSTM`}
      />

      <NoteBlock type="note" title="Which Variant to Choose?">
        <p>
          Large-scale studies (Greff et al., 2017; Jozefowicz et al., 2015) found that <strong>no
          variant consistently outperforms the standard LSTM</strong>. The forget gate bias
          initialization (set to 1.0) matters more than architectural changes. Start with a standard
          LSTM or GRU and only try variants if you have a specific bottleneck.
        </p>
      </NoteBlock>
    </div>
  )
}
