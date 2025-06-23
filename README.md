# Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer

## Introduction
Chain-of-thought (CoT) reasoning has enabled transformer-based language models to excel at complex mathematics and multi-step planning. However, in standard decoder-only architectures, these reasoning steps are externalized in natural language, improving interpretability at the cost of efficiency. To capture reasoning that are not easily represented in words, many works have explored recurrent architectures that aim to internalize reasoning in latent space, potentially supporting latent CoT. In this paper, we investigate whether such reasoning structures emerge in Huginn-3.5B, a depth-recurrent Transformer that reuses layers at inference time without increasing parameter count. We examine the model’s internal behavior on arithmetic tasks using a suite of probing techniques including the Logit Lens and Coda Lens. Our findings reveal limited evidence of interpretable latent CoT by tracking rank trajectories of final and intermediate result tokens. Furthermore, we uncover significant probing inconsistencies across recurrent blocks, where the interpretability of hidden states depends heavily on both the layer index and the decoding method. Finally, we empirically show that increasing recurrence depth yields only marginal gains and falls well short of models that explicitly externalize reasoning steps.

## Experiments

### We dicover significant discontinuities in hidden space interpretability in depth recurrent transformer.

final predicted token ranks, top-5 tokens, 

experiment 
coda_lens: local_exp.py

logit_lens: logit_lens_exp.py

Generate graph for rank trajectory: unrolled_rank_comparison.py

Generate graph for numeric prefixes: frequency_analysis_combined.py


### We trace the rank trajectory of the intermediate and final result tokens in one-digit composite arithmetic task, but find little evidence for latent CoT.

filter dataset --> 67
latent_cot_analysis.py

local_exp_inter.py
Generate correct/intermediate ranks for coda

logit_lens_exp_inter.py
Generate correct/intermediate ranks for logit


### We benchmark Huginn's performance on GSM8k dataset under the condition of suppressing explicit CoT. The conclusion is we still NEED explicit CoT reasoning to achieve optimal performance!

To be consistent with original Huginn paper, we use lm_eval to conduct evaluation on GSM8k, and use the checkpoint [tomg-group-umd/huginn_swa_100_10_avg_0.9_merge](https://huggingface.co/tomg-group-umd/huginn_swa_75_7_ema_0.9_merge/tree/main).

```shell
accelerate launch --num_processes 1 -m lm_eval   --model hf   --model_args pretrained=tomg-group-umd/huginn_swa_100_10_avg_0.9_merge,trust_remote_code=True,dtype=bfloat16,mean_recurrence=128   
--tasks gsm8k_stan   --include_path ./lm_eval/tasks/gsm8k   --batch_size 1   --num_fewshot 8  --output_path outputs/gsm8k_re128   --fewshot_as_multiturn   --apply_chat_template=True   --system_instruction="You are a concise and helpful assistant. Always return only the final answer straightway." --log_samples
```