#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 m_judgebench_eval.py --model_path ../model/M-Judger-RL-Qwen8B/ --benchmark_path ../data/m_judgebench_data.jsonl
python3 m_judgebench_eval.py --model_path ../model/M-Judger-RL-Qwen4B/ --benchmark_path ../data/m_judgebench_data.jsonl
