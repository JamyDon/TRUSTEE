set -xeuo pipefail

HOME='path/to/your/home'
model_scale='8B'    # '8B-Base', '4B', '8B'

pip install bfcl-eval soundfile

project_name='TRUSTEE'
EXP_NAME="TRUSTEE-Qwen3-${model_scale}"
STEP=200

export BFCL_PROJECT_ROOT=$HOME/outputs/bfcl/${project_name}/${EXP_NAME}-step_${STEP}
rm -rf $BFCL_PROJECT_ROOT

FSDP_DIR=$HOME/checkpoints/${project_name}/${EXP_NAME}/global_step_${STEP}/actor
HF_DIR=$HOME/checkpoints/${project_name}/${EXP_NAME}/global_step_${STEP}/actor/hf

if [ -f "$HF_DIR/config.json" ] && [ -f "$HF_DIR/tokenizer_config.json" ]; then
  echo "Target directory already exists, skipping merge"
else
  python -m verl.model_merger merge --backend fsdp --local_dir $FSDP_DIR --target_dir $HF_DIR
fi

bfcl generate \
  --model Qwen/Qwen3-8B-FC \
  --test-category all_scoring \
  --backend vllm \
  --num-gpus 8 \
  --gpu-memory-utilization 0.9 \
  --local-model-path $HF_DIR

bfcl evaluate \
  --model Qwen/Qwen3-8B-FC \
  --test-category all_scoring
