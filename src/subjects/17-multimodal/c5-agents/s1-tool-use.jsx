import { useState } from 'react'
import { BlockMath, InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'
import DefinitionBlock from '../../../components/content/DefinitionBlock.jsx'
import ExampleBlock from '../../../components/content/ExampleBlock.jsx'
import NoteBlock from '../../../components/content/NoteBlock.jsx'
import PythonCode from '../../../components/content/PythonCode.jsx'

function ToolCallFlow() {
  const [step, setStep] = useState(0)
  const steps = [
    { label: 'User Query', content: '"What is the weather in Paris today?"', actor: 'User' },
    { label: 'LLM Decides to Call Tool', content: 'function_call: get_weather(location="Paris")', actor: 'LLM' },
    { label: 'Tool Execution', content: '{"temp": 18, "condition": "partly cloudy"}', actor: 'System' },
    { label: 'LLM Final Response', content: '"It\'s 18C and partly cloudy in Paris today."', actor: 'LLM' },
  ]
  const s = steps[step]
  const colors = { User: 'bg-gray-100 dark:bg-gray-800', LLM: 'bg-violet-50 dark:bg-violet-900/20', System: 'bg-green-50 dark:bg-green-900/20' }

  return (
    <div className="my-6 rounded-xl border border-gray-200 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/50">
      <h3 className="mb-1 text-base font-bold text-gray-800 dark:text-gray-200">Tool Use Flow</h3>
      <div className="flex gap-1 mb-3">
        {steps.map((st, i) => (
          <button key={i} onClick={() => setStep(i)}
            className={`flex-1 px-2 py-1 rounded text-xs transition ${step === i ? 'bg-violet-500 text-white' : 'bg-gray-100 text-gray-600 dark:bg-gray-800 dark:text-gray-400'}`}>
            Step {i + 1}
          </button>
        ))}
      </div>
      <div className={`p-3 rounded-lg ${colors[s.actor]} text-sm`}>
        <p className="font-medium text-gray-700 dark:text-gray-300">{s.label} <span className="text-xs text-gray-500">({s.actor})</span></p>
        <code className="text-xs block mt-1 text-gray-600 dark:text-gray-400">{s.content}</code>
      </div>
      <div className="flex mt-2 gap-1">
        {steps.map((_, i) => <div key={i} className={`h-1 flex-1 rounded ${i <= step ? 'bg-violet-500' : 'bg-gray-200 dark:bg-gray-700'}`} />)}
      </div>
    </div>
  )
}

export default function ToolUseFunctionCalling() {
  return (
    <div className="space-y-6">
      <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
        Tool use extends LLMs beyond text generation by allowing them to invoke external functions,
        APIs, and databases. The model learns to generate structured function calls and incorporate
        results into its responses.
      </p>

      <DefinitionBlock title="Function Calling Formulation">
        <p>Given a user query <InlineMath math="q" /> and available tools <InlineMath math="\mathcal{T} = \{t_1, \ldots, t_K\}" />, the model generates:</p>
        <BlockMath math="a = \text{LLM}(q, \mathcal{T}) = \begin{cases} \text{text response} & \text{if no tool needed} \\ (t_k, \text{args}) & \text{if tool } t_k \text{ should be called} \end{cases}" />
        <p className="mt-2">The tool schema (name, description, parameters with types) is provided in the system prompt or as structured metadata. The model must decide <em>whether</em> to call a tool, <em>which</em> tool, and <em>what arguments</em> to pass.</p>
      </DefinitionBlock>

      <ToolCallFlow />

      <ExampleBlock title="Training for Tool Use">
        <p>Models learn tool use through:</p>
        <ul className="list-disc list-inside mt-2 space-y-1">
          <li><strong>Fine-tuning:</strong> Supervised training on (query, tool_call, result, response) tuples</li>
          <li><strong>Self-play:</strong> Model generates tool calls, executes them, and is trained on successful trajectories</li>
          <li><strong>RLHF:</strong> Human preference on responses that use tools correctly vs incorrectly</li>
        </ul>
        <p className="mt-2">GPT-4 and Claude support parallel tool calls — invoking multiple tools simultaneously when appropriate.</p>
      </ExampleBlock>

      <PythonCode
        title="Simple Tool-Use Loop"
        code={`from dataclasses import dataclass
from typing import Any

@dataclass
class Tool:
    name: str
    description: str
    parameters: dict
    function: Any  # callable

def tool_use_loop(query, tools, llm_generate, max_steps=5):
    """Execute a tool-use conversation loop.

    Args:
        query: user question
        tools: list of Tool objects
        llm_generate: function(messages, tools) -> response
    """
    messages = [{"role": "user", "content": query}]
    tool_schemas = [{"name": t.name, "description": t.description,
                     "parameters": t.parameters} for t in tools]

    for step in range(max_steps):
        response = llm_generate(messages, tool_schemas)

        if response.get("tool_call"):
            # Execute the tool
            tool_name = response["tool_call"]["name"]
            args = response["tool_call"]["arguments"]
            tool = next(t for t in tools if t.name == tool_name)
            result = tool.function(**args)

            messages.append({"role": "assistant", "tool_call": response["tool_call"]})
            messages.append({"role": "tool", "content": str(result)})
        else:
            return response["content"]  # Final text response

    return "Max tool-use steps reached"

# Example usage (pseudocode)
print("Tool-use loop: query -> [tool_call -> result]* -> final_response")
print("Key: LLM decides WHEN and WHICH tools to call")`}
      />

      <NoteBlock type="note" title="Structured Output and JSON Mode">
        <p>
          Tool use relies on the model producing valid structured output (JSON). Constrained
          decoding techniques force the model to only generate tokens that form valid JSON
          matching the tool schema, eliminating parsing errors. This is critical for reliable
          agentic systems.
        </p>
      </NoteBlock>
    </div>
  )
}
