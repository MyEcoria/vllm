#!/bin/bash
# usage: serve_run.sh <log> <model> [extra vllm serve args...]
S=/tmp/claude-1000/-home-volta-Desktop-Perso-vllm/18551204-76a8-4fa3-9198-c3f30d4df18c/scratchpad
LOG=$1; MODEL=$2; shift 2
cd ~/.cache/bonsai-ref
timeout 1500 /home/volta/Desktop/Perso/vllm/.venv/bin/vllm serve "$MODEL" --tokenizer Qwen/Qwen3.8-27B --served-model-name bonsai --port 8011 --max-model-len ${MAXLEN:-4096} --gpu-memory-utilization ${GMU:-0.92} --limit-mm-per-prompt '{"image":0,"video":0}' --max-num-seqs ${MAXSEQS:-4} "$@" > $LOG 2>&1 &
until curl -s localhost:8011/v1/models >/dev/null 2>&1; do
  if ! pgrep -f "bin/vllm serve" >/dev/null; then echo "SERVER DIED"; grep -E "Error|error" $LOG | tail -5; exit 1; fi
  sleep 3
done
/home/volta/Desktop/Perso/vllm/.venv/bin/python $S/tools/serve_bench.py
/home/volta/Desktop/Perso/vllm/.venv/bin/python $S/tools/serve_text.py 2>/dev/null
[ -n "$DUMP" ] && /home/volta/Desktop/Perso/vllm/.venv/bin/python $S/tools/serve_dump.py $DUMP
[ -n "$CHAT" ] && /home/volta/Desktop/Perso/vllm/.venv/bin/python $S/tools/serve_chat.py
[ -n "$LONG" ] && /home/volta/Desktop/Perso/vllm/.venv/bin/python $S/tools/longctx.py
sleep 11
grep -E "SpecDecoding|acceptance|Mean acceptance" $LOG | tail -2 | cut -c1-300
pkill -f "vllm serve" ; sleep 5
