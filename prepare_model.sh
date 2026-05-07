HOME=

MODEL_ID=Qwen/Qwen3-4B
MODEL_DIR=$HOME/model/$MODEL_ID
huggingface-cli download --resume-download $MODEL_ID --force-download --local-dir $MODEL_DIR --local-dir-use-symlinks False

MODEL_ID=Qwen/Qwen3-8B
MODEL_DIR=$HOME/model/$MODEL_ID
huggingface-cli download --resume-download $MODEL_ID --force-download --local-dir $MODEL_DIR --local-dir-use-symlinks False

MODEL_ID=Qwen/Qwen3-8B-Base
MODEL_DIR=$HOME/model/$MODEL_ID
huggingface-cli download --resume-download $MODEL_ID --force-download --local-dir $MODEL_DIR --local-dir-use-symlinks False