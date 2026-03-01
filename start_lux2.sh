/home/acipolletti/llama.cpp/build/bin/llama-server \
    -m /home/acipolletti/models/flux2/flux2-dev-BF16.gguf \
    -ngl 99 \
    -c 8192 \
    --mlock \
    -fa on \
    --host 0.0.0.0 \
    --port 8080
