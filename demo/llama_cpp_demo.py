import argparse
import os
from pathlib import Path

from llama_cpp import Llama


def load_llm(model_path: str) -> Llama:
    """
    Load a GGUF model using llama-cpp-python.

    By default this uses all available CPU cores. Adjust n_threads if needed.
    """
    model_path = str(Path(model_path).expanduser())
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=os.cpu_count() or 4,
    )
    return llm


def run_single_completion(llm: Llama) -> None:
    """Run a single demo completion."""
    prompt = (
        "You are a helpful, concise assistant.\n\n"
        "Question: What is the capital of France?\n"
        "Answer:"
    )

    print("\n=== Single completion demo ===")
    result = llm(
        prompt,
        max_tokens=64,
        temperature=0.2,
        stop=["\n\n", "Question:"],
    )

    text = result["choices"][0]["text"]
    print("Model answer:", text.strip())


def interactive_loop(llm: Llama) -> None:
    """
    Very simple REPL-style chat loop.

    This does NOT keep full conversation history; each turn is independent,
    but it's enough to verify the model is working.
    """
    system_prompt = (
        "You are a helpful assistant. Answer clearly and concisely.\n"
        "Do NOT write out your reasoning process or thoughts. Only output the final answer.\n"
        "finish your answer with '<END_OF_ANSWER>'"
    )
    print("\n=== Interactive demo ===")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        #prompt = f"{system_prompt}\n\nUser: {user_input}\nAssistant:"
        prompt = f"{system_prompt}\n\n{user_input}\n"
        result = llm(
            prompt,
            max_tokens=2000,
            temperature=0.5,
            stop=["<END_OF_ANSWER>", "\n\n"],
        )

        text = result["choices"][0]["text"].strip()
        print(f"Assistant: {text}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal demo of llama.cpp via llama-cpp-python"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.environ.get(
            "LLAMA_MODEL_PATH",
            "/home/alicekenway/Dev/project/argue/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        ),
        help=(
            "Path to the GGUF model file. "
            "Defaults to $LLAMA_MODEL_PATH or the Qwen3-8B-Q4_K_M.gguf in this repo."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = load_llm(args.model_path)

    # Simple one-shot completion
    run_single_completion(llm)

    # Optional interactive loop
    interactive_loop(llm)


if __name__ == "__main__":
    """
    Example usage:

        # If llama-cpp-python is not installed yet:
        #   pip install llama-cpp-python
        #
        # Then run:
        python demo/llama_cpp_demo.py \
          --model-path /home/alicekenway/Dev/project/argue/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf

        # Or set an environment variable:
        #   export LLAMA_MODEL_PATH=/path/to/model.gguf
        #   python demo/llama_cpp_demo.py
    """
    main()


