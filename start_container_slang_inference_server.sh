# Launch container with GPU support and port mapping
docker run --gpus all -it --rm \
  -p 30000:30000 \
  -v /home/acipolletti/models:/tmp \
  lmsysorg/sglang:spark \
  bash

