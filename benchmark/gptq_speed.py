from __future__ import annotations
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any
import torch
from transformers import AutoTokenizer


@dataclass(frozen=True)
class TrialSummary:
    framework: str
    device: str
    phase: str
    token_count: int
    samples_ms: list[float]

    @property
    def mean_ms(self) -> float:
        return sum(self.samples_ms) / len(self.samples_ms)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)

    @property
    def max_ms(self) -> float:
        return max(self.samples_ms)

    @property
    def toks_per_s(self) -> float:
        return self.token_count / (self.mean_ms / 1000.0)


@dataclass
class BenchmarkResult:
    """Container for benchmark results"""
    prompt_per_second: List[float]  # tokens per second for each trial
    predicted_per_second: List[float]  # tokens per second for each trial
    prefill_summaries: List[TrialSummary]
    decode_summaries: List[TrialSummary]


def _sync_cuda(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _bench(fn: Callable[[], None], *, device: str, warmup: int, trials: int) -> list[float]:
    samples_ms: list[float] = []
    for _ in range(warmup):
        fn()
        _sync_cuda(device)

    for _ in range(trials):
        _sync_cuda(device)
        t0 = time.perf_counter()
        fn()
        _sync_cuda(device)
        samples_ms.append((time.perf_counter() - t0) * 1000.0)
    return samples_ms


def _build_prompt(sentence_str: str, tokenizer: AutoTokenizer, target_tokens: int) -> tuple[str, int]:
    """Build a prompt by repeating a sentence until it reaches target_tokens"""
    prompt = sentence_str
    token_count = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    while token_count < target_tokens:
        prompt += " " + sentence_str
        token_count = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
    return prompt, token_count


def _gptq_prefill(model, tokenizer, prompt: str, device: str) -> tuple[Callable[[], None], int]:
    """Create prefill benchmark function"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    token_count = inputs["input_ids"].shape[1]

    def run_once() -> None:
        with torch.inference_mode():
            _ = model(**inputs)

    return run_once, token_count


def _gptq_decode(model, tokenizer, prompt: str, decode_tokens: int, device: str) -> tuple[Callable[[], None], int]:
    """Create decode benchmark function"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    def run_once() -> None:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=decode_tokens,
                do_sample=False,
                use_cache=True
            )

    return run_once, decode_tokens


def gptqmodel_speed(
    model,
    prompt: str,
    prompt_tokens: int = 128,
    decode_tokens: int = 32,
    warmup: int = 1,
    trials: int = 1,
    device: str = "cpu",
    verbose: bool = True
) -> BenchmarkResult:
    """
    Benchmark GPTQModel speed.
    
    Args:
        model: Loaded GPTQModel instance
        prompt: Input prompt text
        prompt_tokens: Target number of prompt tokens
        decode_tokens: Number of tokens to decode
        warmup: Number of warmup iterations
        trials: Number of measured trials
        device: Device to benchmark ("cpu", "cuda", or "both")
        verbose: Print detailed output
    
    Returns:
        BenchmarkResult containing timing data
    """
    try:
        tokenizer = model.tokenizer
        built_prompt, actual_prompt_tokens = _build_prompt(prompt, tokenizer, prompt_tokens)
        
        # Determine devices to benchmark
        if device == "both":
            devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        else:
            devices = [device]
        
        all_prompt_per_second = []
        all_predicted_per_second = []
        all_prefill_summaries = []
        all_decode_summaries = []

        for dev in devices:
            if dev == "cuda" and not torch.cuda.is_available():
                if verbose:
                    print(f"Skipping CUDA - not available")
                continue
                
            try:
                # Ensure model is on correct device
                if dev == "cuda" and hasattr(model, 'to'):
                    model.model = model.model.to(dev)
                
                # Prefill benchmark
                prefill_fn, prefill_tokens = _gptq_prefill(model.model, tokenizer, built_prompt, dev)
                prefill_samples = _bench(
                    prefill_fn, 
                    device=dev, 
                    warmup=warmup, 
                    trials=trials
                )
                
                # Create summary
                prefill_summary = TrialSummary(
                    framework="gptqmodel",
                    device=dev,
                    phase="prefill",
                    token_count=prefill_tokens,
                    samples_ms=prefill_samples
                )
                all_prefill_summaries.append(prefill_summary)
                
                # Calculate tokens per second for each trial
                prompt_tps = [prefill_tokens / (ms / 1000.0) for ms in prefill_samples]
                all_prompt_per_second.extend(prompt_tps)

                decode_fn, decode_tokens_actual = _gptq_decode(model.model, tokenizer, built_prompt, decode_tokens, dev)
                decode_samples = _bench(
                    decode_fn,
                    device=dev,
                    warmup=warmup,
                    trials=trials
                )
                
                # Create summary
                decode_summary = TrialSummary(
                    framework="gptqmodel",
                    device=dev,
                    phase="decode",
                    token_count=decode_tokens_actual,
                    samples_ms=decode_samples
                )
                all_decode_summaries.append(decode_summary)
                
                # Calculate tokens per second for each trial
                decode_tps = [decode_tokens_actual / (ms / 1000.0) for ms in decode_samples]
                all_predicted_per_second.extend(decode_tps)

            except Exception as e:
                if verbose:
                    print(f"Error benchmarking on {dev}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

        return BenchmarkResult(
            prompt_per_second=all_prompt_per_second,
            predicted_per_second=all_predicted_per_second,
            prefill_summaries=all_prefill_summaries,
            decode_summaries=all_decode_summaries
        )

    except Exception as e:
        if verbose:
            print(f"Error in benchmark: {e}")
            import traceback
            traceback.print_exc()
        return BenchmarkResult(
            prompt_per_second=[],
            predicted_per_second=[],
            prefill_summaries=[],
            decode_summaries=[]
        )