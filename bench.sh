#!/bin/bash
#
# OpenClaw Agent Benchmark
#
# Simulates a 3-turn agent session with KV cache reuse:
#   Turn 1: 16K system prompt (cold)
#   Turn 2: 16K cached prefix + 128 new tokens (follow-up chat)
#   Turn 3: 16K cached prefix + 4K new tokens (tool call output)
#
# Usage:
#   # Against vLLM:
#   MODEL=openai/gpt-oss-120b URL=http://localhost:8000 BACKEND=openai-chat bash bench.sh
#
#   # Against Ollama:
#   MODEL=gpt-oss:120b URL=http://localhost:11434 BACKEND=openai-chat bash bench.sh
#
set -euo pipefail

MODEL="${MODEL:-openai/gpt-oss-120b}"
TOKENIZER="${TOKENIZER:-openai/gpt-oss-120b}"
URL="${URL:-http://localhost:8000}"
BACKEND="${BACKEND:-openai-chat}"
RESULT_DIR="${RESULT_DIR:-./results}"
RESULT_FILE="${RESULT_FILE:-agent_turns.json}"

# 3-turn agent session:
#   prefix,new_tokens,output_tokens
#   Turn 1: 0,16384,128       = cold 16K system prompt
#   Turn 2: 16384,128,128     = cached prefix + short follow-up
#   Turn 3: 16384,4096,512    = cached prefix + tool output (e.g. file read)
TURN_CONFIGS="${TURN_CONFIGS:-0,16384,128|16384,128,128|16384,4096,512}"

mkdir -p "$RESULT_DIR"

echo "============================================"
echo "  OpenClaw Agent Turn Benchmark"
echo "  Model:   $MODEL"
echo "  Backend: $BACKEND @ $URL"
echo "  Turns:   $TURN_CONFIGS"
echo "============================================"

python3 benchmark_serving.py \
    --backend "$BACKEND" \
    --base-url "$URL" \
    --endpoint "/v1/chat/completions" \
    --model "$MODEL" \
    --tokenizer "$TOKENIZER" \
    --dataset-name agent-turns \
    --turn-configs "$TURN_CONFIGS" \
    --skip-initial-test \
    --ignore-eos \
    --percentile-metrics ttft,tpot,itl,e2el \
    --max-concurrency 1 \
    --save-result \
    --result-dir "$RESULT_DIR" \
    --result-filename "$RESULT_FILE"

echo ""
echo "Results saved to: $RESULT_DIR/$RESULT_FILE"
