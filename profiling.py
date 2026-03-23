#!/usr/bin/env python3
"""Ollama profiling: measures prefill time, decode speed, and KV cache reuse.

Reports prompt_eval_duration and eval_duration from Ollama API responses
for accurate per-component breakdown.

Usage:
  MODEL=gpt-oss:120b python3 profiling.py
  MODEL=qwen3:30b python3 profiling.py
"""

import json
import os
import urllib.request
import time
import random
import sys

# Use tokenizer to generate consistent prompts
try:
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    print("Install: pip3 install --user transformers")
    sys.exit(1)

MODEL = os.environ.get("MODEL", "gpt-oss:120b")
URL = os.environ.get("URL", "http://localhost:11434")

random.seed(42)
prefix_ids = [random.randint(0, t.vocab_size - 1) for _ in range(16384)]
prefix_text = t.decode(prefix_ids)

random.seed(99)
suffix_4k_ids = [random.randint(0, t.vocab_size - 1) for _ in range(4096)]
suffix_4k_text = t.decode(suffix_4k_ids)


def send(label, text, num_predict=32):
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": text}],
        "stream": False,
        "options": {"num_predict": num_predict}
    }).encode()
    req = urllib.request.Request(f"{URL}/api/chat",
                                 data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=600)
    t1 = time.perf_counter()
    d = json.loads(resp.read())
    pe = d.get("prompt_eval_duration", 0)
    ev = d.get("eval_duration", 0)
    pc = d.get("prompt_eval_count", 0)
    ec = d.get("eval_count", 0)
    overhead = (t1 - t0) * 1000 - pe / 1e6 - ev / 1e6
    prefill_tps = pc / (pe / 1e9) if pe else 0
    decode_tps = ec / (ev / 1e9) if ev else 0
    print(f"{label}:")
    print(f"  Total:    {(t1-t0)*1000:>8.0f}ms")
    print(f"  Prefill:  {pe/1e6:>8.0f}ms  ({pc} tokens, {prefill_tps:.0f} tok/s)")
    print(f"  Decode:   {ev/1e6:>8.0f}ms  ({ec} tokens, {decode_tps:.1f} tok/s)")
    print(f"  Overhead: {overhead:>8.0f}ms")
    print()
    return {
        "label": label,
        "total_ms": round((t1 - t0) * 1000),
        "prefill_ms": round(pe / 1e6),
        "prefill_tokens": pc,
        "prefill_tps": round(prefill_tps, 1),
        "decode_ms": round(ev / 1e6),
        "decode_tokens": ec,
        "decode_tps": round(decode_tps, 1),
        "overhead_ms": round(overhead),
    }


# Pre-load model
print("Loading model...")
send("Warmup (load model)", "hi", 1)

# Flush KV
import subprocess
subprocess.run(["/usr/local/bin/ollama", "stop", MODEL], capture_output=True)
time.sleep(2)
send("Reload (fresh KV)", "hi", 1)

print("=" * 60)
print("  Mac M4 Max PROFILING: gpt-oss-120b via Ollama")
print("=" * 60)
print()

results = []
results.append(send("1. Cold 16K prefill", prefix_text))
results.append(send("2. Exact repeat (full cache hit)", prefix_text))
results.append(send("3. 16K prefix + short suffix",
                     prefix_text + " Tell me about distributed systems."))
results.append(send("4. 16K prefix + 4K new tokens",
                     prefix_text + suffix_4k_text))

random.seed(123)
diff_ids = [random.randint(0, t.vocab_size - 1) for _ in range(16384)]
diff_text = t.decode(diff_ids)
results.append(send("5. Different 16K (cache miss)", diff_text))

# Save results
with open("/tmp/mac_profiling_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved to /tmp/mac_profiling_results.json")
