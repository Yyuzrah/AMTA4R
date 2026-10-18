"""Models (inference only): open instruction-tuned models with a permissive licence. fetch_models.py records the
hub licence tag and the licence of the model-card front matter in meta/licenses.json, together with the revision,
layer count L and width. `big` models (>= 24B) need one 80 GB GPU (bf16).
`chat_kwargs` are passed to the chat template (Qwen3: thinking disabled, answer-only regime)."""
MODELS = {
    # tag: (hub id, big, chat_kwargs)
    'qwen2.5-7b': ('Qwen/Qwen2.5-7B-Instruct', False, {}),
    'qwen2.5-14b': ('Qwen/Qwen2.5-14B-Instruct', False, {}),
    'qwen2.5-32b': ('Qwen/Qwen2.5-32B-Instruct', True, {}),
    'olmo2-7b': ('allenai/OLMo-2-1124-7B-Instruct', False, {}),
    'olmo2-13b': ('allenai/OLMo-2-1124-13B-Instruct', False, {}),
    'olmo2-32b': ('allenai/OLMo-2-0325-32B-Instruct', True, {}),
    'qwen3-8b': ('Qwen/Qwen3-8B', False, {'enable_thinking': False}),
    'qwen3-14b': ('Qwen/Qwen3-14B', False, {'enable_thinking': False}),
    'qwen3-32b': ('Qwen/Qwen3-32B', True, {'enable_thinking': False}),
    'phi4-14b': ('microsoft/phi-4', False, {}),
    'mistral-small-24b': ('mistralai/Mistral-Small-24B-Instruct-2501', True, {}),
}
FAMILY = {'qwen2.5-7b': 'qwen2.5', 'qwen2.5-14b': 'qwen2.5', 'qwen2.5-32b': 'qwen2.5', 'olmo2-7b': 'olmo2',
          'olmo2-13b': 'olmo2', 'olmo2-32b': 'olmo2', 'qwen3-8b': 'qwen3', 'qwen3-14b': 'qwen3', 'qwen3-32b': 'qwen3',
          'phi4-14b': 'phi4', 'mistral-small-24b': 'mistral'}
ORDER = ['qwen2.5-7b', 'qwen2.5-14b', 'olmo2-7b', 'olmo2-13b', 'qwen3-8b', 'qwen3-14b', 'phi4-14b',
         'qwen2.5-32b', 'olmo2-32b', 'qwen3-32b', 'mistral-small-24b']
SMALL = [m for m in ORDER if not MODELS[m][1]]     # models up to 14B (1 GPU, >= 40 GB)
BIG = [m for m in ORDER if MODELS[m][1]]           # models of 24B-32B (1 GPU, 80 GB)
# Excluded on licence grounds (not permissive): Qwen2.5-3B/72B-Instruct (Qwen Research / Qwen License),
# Llama 3.x (Llama Community License), Gemma (Gemma Terms of Use), Ministral-8B (Mistral Research License).
EXCLUDED = {'Qwen/Qwen2.5-3B-Instruct': 'license: other (Qwen Research License)',
            'Qwen/Qwen2.5-72B-Instruct': 'license: other (Qwen License)',
            'meta-llama/*': 'Llama Community License (use restrictions)',
            'google/gemma-*': 'Gemma Terms of Use', 'mistralai/Ministral-8B-Instruct-2410': 'Mistral Research License'}
PERMISSIVE = {'license:apache-2.0', 'license:mit'}
