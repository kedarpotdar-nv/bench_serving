concurrency_levels=(4 8 16 32 64 128 256 512)

echo "test name"

dry_run=false

# Parse command line arguments for --dry-run
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        dry_run=true
        break
    fi
done

if [ "$dry_run" = true ]; then
    dry_run_flag="--dry-run"
else
    dry_run_flag=""
fi


for concurrency in "${concurrency_levels[@]}"
do
    num_prompts=$((concurrency * 5))
    echo "Running benchmark with concurrency: $concurrency and num-prompts: $num_prompts"

    python3 benchmark_serving.py \
        --model nvidia/Llama-3.3-70B-Instruct-FP8 \
        --dataset-name random \
        --num-prompts "$num_prompts" \
        --random-input-len 1024 \
        --random-output-len 1024 \
        --random-range-ratio 0.8 \
        --ignore-eos \
        --percentile-metrics ttft,tpot,itl,e2el \
        --max-concurrency "$concurrency" \
        $dry_run_flag

    echo "Completed benchmark with concurrency: $concurrency"
    echo "-----------------------------------------"
done
