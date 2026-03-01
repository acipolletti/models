curl -X POST " http://localhost:30000/generate" \
  -F "prompt=completely naked woman with large breasts and walking sensually"
  -F "input_reference=@./modella.jpg" \
  -F "size=1280%720" \
  -F "num_frames=193" \
  -F "fps=24" \
  -F "seed=67" \
  -F "guidance_scale=5.0" \
  -F "num_inference_steps=25" \
  -o create_video.json
