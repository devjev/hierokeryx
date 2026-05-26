# Run on GPU

CPU is fine for documents up to a few hundred KB. For batch jobs over
hundreds of documents, GPU on both GLiNER and the sentence-transformer
embedder gives a 10–50x speedup depending on hardware.

## Install the GPU extra

```bash
uv add 'hierokeryx[gpu]'
# or
pip install 'hierokeryx[gpu]'
```

This pulls in `torch>=2.5` with CUDA support. On macOS, `torch` includes
MPS (Metal) support out of the box — no extra needed.

## Tell GLiNER and the embedder to use the GPU

The CLI doesn't expose a device flag at v0.1 — drop into library mode:

```python
from hierokeryx import pipeline
from hierokeryx.extract.gliner_runner import GLiNERExtractor
from hierokeryx.resolve.embed import SentenceTransformerEmbedder
from hierokeryx.llm.anthropic_client import AnthropicClient

extractor = GLiNERExtractor(device="cuda")             # or "mps" on Apple
embedder  = SentenceTransformerEmbedder(device="cuda")

run = pipeline.run(
    documents=docs,
    schema=schema,
    workdir="workdir/",
    extractor=extractor,
    embedder=embedder,
    llm_client=AnthropicClient(),
)
```

Both classes pass `device` straight through to the underlying
HuggingFace / sentence-transformers model. Any string those libraries
accept (`"cuda"`, `"cuda:0"`, `"mps"`, `"cpu"`) works.

## Verifying you're actually on GPU

```python
import torch
print(torch.cuda.is_available(), torch.cuda.device_count())
```

If your model is loading but you see CPU utilisation spike instead of
GPU, you almost certainly forgot to pass `device=` — the default is
`"cpu"` in both classes.

## NixOS

The flake in `flake.nix` provides the C++ runtime libs PyPI's `torch`
wheel needs (`libstdc++`, `libgcc_s`, `libgomp`). For CUDA, you have two
options:

1. **Add `cudaPackages.cudatoolkit` to `mlRuntimeLibs` in `flake.nix`**
   and put the CUDA libs on `LD_LIBRARY_PATH`. Painful but works.
2. **Use the upstream `nixpkgs.python313Packages.torchWithCuda`** instead
   of the PyPI wheel. This requires switching from the uv-managed venv to
   a Nix-managed Python environment, which is a bigger lift.

Most users find option 1 easier for development and option 2 better for
reproducible deployments. CUDA-on-NixOS is documented at
<https://nixos.wiki/wiki/CUDA>.

## Apple Silicon

`device="mps"` works for both GLiNER and the sentence-transformer
embedder. Expect roughly half the throughput of an NVIDIA RTX 4090 on
an M3 Max. The first call to the GPU takes a few extra seconds (kernel
compilation) and you may see warnings about unsupported ops — they
gracefully fall back to CPU per-op without crashing.

## Batch size

The GLiNER runner uses `batch_size=8` internally for its
`predict_entities` call. If you have a beefy GPU and are processing long
documents, instantiate the extractor with a larger batch size:

```python
extractor = GLiNERExtractor(device="cuda", batch_size=32)
```

Larger batches help with throughput; they don't change results.

## What stays on CPU regardless

- **The LLM calls.** Claude (and any provider you swap in) runs
  remotely — GPU on your machine doesn't affect API latency.
- **The clustering math.** NumPy union-find on a few hundred entities
  takes milliseconds; pinning it to GPU adds overhead.

The GPU only helps the two model loads: GLiNER's NER pass and the
sentence-transformer embedder. Everything else is either I/O or remote.
