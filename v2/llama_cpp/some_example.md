# Handy llama.cpp Binaries – Short Guide

_All paths assume you’re in `llama.cpp/build/bin`._

---

## 1. `llama-cli` – Command-Line Playground

**What it is**  
General-purpose CLI client for llama.cpp. Lets you chat or run one-shot completions from the terminal.

**Example**

```bash
llama-cli \
  -m models/my-model/your_model.gguf \
  -p "Hello, who are you?" \
  -n 64
```

- `-m` – model path  
- `-p` – prompt  
- `-n` – number of tokens to generate  

**Use when**  
You want to quickly test whether the model runs and try prompts without dealing with HTTP or a UI.

---

## 2. `llama-server` – HTTP API Backend (for TS/Next.js)

**What it is**  
An HTTP server exposing (mostly) OpenAI-compatible endpoints for chat, completion, etc. This is the one you’ll call from your TypeScript frontend.

**Example – start server**

```bash
llama-server \
  -m models/my-model/your_model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 4096 \
  -ngl 0
```

**Example – call from HTTP client**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your_model",
    "messages": [
      { "role": "user", "content": "Say one short sentence about llamas." }
    ],
    "max_tokens": 64
  }'
```

**Use when**  
You want your TS/Next.js app to talk to a local model like it’s an OpenAI API.

---

## 3. `llama-simple` / `llama-simple-chat` – Minimal C Examples

**What they are**

- `llama-simple`: minimal example showing how to load a model and run a prompt.  
- `llama-simple-chat`: minimal example showing a simple chat loop.

**Use when**  
You want to understand the bare minimum llama.cpp API usage (e.g. for embedding llama.cpp directly in your own C/C++ app).  
_Not required for a TS + HTTP stack, but good reference._

---

## 4. `llama-bench` – Performance Benchmark

**What it is**  
Benchmarks your model to measure tokens/sec under different settings.

**Example**

```bash
llama-bench -m models/my-model/your_model.gguf
```

**Use when**

- Comparing quantizations (Q4 vs Q5 vs Q8).
- Checking how much GPU offload or thread count helps.
- Ensuring your setup is fast enough for interactive chat.

---

## 5. `llama-perplexity` – Text Perplexity Evaluation

**What it is**  
Computes perplexity on a given text file to evaluate model quality.

**Example**

```bash
llama-perplexity \
  -m models/my-model/your_model.gguf \
  -f data.txt
```

**Use when**  
You want a numeric measure of how well the model fits some text, or to compare different finetunes/quantizations in a more principled way.

---

## 6. `llama-quantize` – Model Quantization

**What it is**  
Converts a full-precision GGUF into a quantized GGUF (smaller, faster, with some quality trade-off).

**Example**

```bash
llama-quantize \
  input-f16.gguf \
  output-q4_0.gguf \
  Q4_0
```

**Use when**

- You have a large F16/BF16 GGUF and want a smaller version that fits into RAM/VRAM.
- You want to experiment with different quantization levels and then benchmark them with `llama-bench`.

---

## 7. GGUF Utilities

### `llama-gguf`, `llama-gguf-hash`, `llama-gguf-split`

**What they are**

- `llama-gguf` – inspect/modify GGUF metadata and tensors.  
- `llama-gguf-hash` – compute a stable hash of a model file (for caching/debugging).  
- `llama-gguf-split` – split a big GGUF into parts.

**Use when**  
You’re debugging model files, distributing large models, or doing low-level GGUF work.

---

## 8. `llama-embedding` – Text Embeddings

**What it is**  
Generates embedding vectors from text using an embedding GGUF model.

**Example**

```bash
llama-embedding \
  -m models/embedding-model/embedding.gguf \
  -p "Hello world!"
```

**Use when**

- Building RAG (index documents by embeddings).
- Doing semantic search or clustering.
- Combining with a vector DB and your chat model.

---

## 9. `llama-retrieval` – RAG Demo

**What it is**  
End-to-end example of retrieval-augmented generation: indexes docs, retrieves relevant chunks, and feeds them to the LLM.

**Use when**  
You want to see how a simple RAG pipeline can be built around llama.cpp before writing your own TS/Node implementation.

---

## 10. `llama-tts` – Text-to-Speech

**What it is**  
TTS (text-to-speech) demo for GGUF TTS models.

**Example**

```bash
llama-tts \
  -m models/tts-model/tts.gguf \
  -p "Hello, this is a test."
```

**Use when**  
You want to add audio output to your demo (voice assistant, talking bot, etc.).

---

## 11. `llama-finetune` – In-Place Fine-Tuning

**What it is**  
Basic fine-tuning tool to train/finetune models directly with llama.cpp.

**Use when**

- You want to adapt a model to your own data without leaving the C++ / ggml ecosystem.
- (Alternative: fine-tune in PyTorch and convert to GGUF.)

---

## 12. `llama-run` – Advanced Launcher

**What it is**  
An advanced “launcher” for running models, sometimes used with external model registries or more complex setups.

**Use when**  
You’re doing more advanced orchestrations.  
For a simple local chat demo, `llama-server` and `llama-cli` are usually enough.

---

## What You Can Ignore for Now

- All `test-*` binaries (e.g. `test-chat`, `test-quantize-fns`, `test-grammar-parser`, etc.)  
  → These are internal tests/examples for llama.cpp developers.

---

## For a TS Chat Demo, Focus On

1. `llama-server` – to run the HTTP backend your TS frontend will call.  
2. `llama-cli` – for quick sanity checks and prompt experiments.  
3. `llama-bench` + `llama-quantize` – to tune performance.  
4. Later: `llama-embedding` / `llama-retrieval` for RAG, and `llama-tts` if you want voice output.
