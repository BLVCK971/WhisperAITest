import whisper

model = whisper.load_model("large")
result = model.transcribe("audiomarc2.wav",fp16=False, verbose=True)

# save result to file
with open("result.txt", "w") as f:
    f.write(result["text"])
