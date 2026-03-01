export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export SG_OUTPUT_DIR=/home/acipolletti/models
mkdir -p "$SG_OUTPUT_DIR"

sglang serve \
  --model-path /home/acipolletti/models/MOVA-720p \
  --host 0.0.0.0 \
  --port 30002 \
  --adjust-frames false \
  --num-gpus 1 \
  --tp 1 \
  --no-dit-cpu-offload \
  --no-dit-layerwise-offload \
  --no-text-encoder-cpu-offload \
  --no-vae-cpu-offload \
  --enable-torch-compile \
  --save-output \
  --output-dir "$SG_OUTPUT_DIR"

