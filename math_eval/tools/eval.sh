#!/bin/bash  
#SBATCH --job-name=JOBNAME 
#SBATCH -A aisc
#SBATCH --partition=aisc           
#SBATCH --gres=gpu:2   
#SBATCH -n 64
#SBATCH --mem=64G
#SBATCH --output=/home/wxy320/ondemand/program/llm_skills/math_eval/scripts/DeepSeek-R1-Distill-Qwen-7B.txt      
#SBATCH --error=/home/wxy320/ondemand/program/llm_skills/math_eval/scripts/DeepSeek-R1-Distill-Qwen-7B.txt   
#SBATCH --time=0-20:00:00


python /home/wxy320/ondemand/program/llm_skills/math_eval/tools/eval.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --evals MATH500,AIME,GPQADiamond,GSM8K,OlympiadBenchMath \
    --tp 2 --output_file ./results/eval/DeepSeek-R1-Distill-Qwen-7B.txt \
    --result_dir ./results/generated
