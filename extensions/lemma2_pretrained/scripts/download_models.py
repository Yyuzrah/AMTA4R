# Downloads the eight public checkpoints (weights, configs, tokenizers, model cards) into the Hugging Face cache.
from huggingface_hub import snapshot_download, hf_hub_download
MODELS = ['gpt2', 'gpt2-medium', 'EleutherAI/gpt-neo-125m', 'EleutherAI/gpt-neo-1.3B',
          'EleutherAI/pythia-160m', 'EleutherAI/pythia-410m', 'EleutherAI/pythia-1.4b',
          'HuggingFaceTB/SmolLM2-360M']
for m in MODELS:
    p = snapshot_download(m, allow_patterns=['*.json', '*.txt', 'model.safetensors', 'tokenizer*', 'README.md', '*.model'])
    print(m, p, flush=True)
