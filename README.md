# GPT-SoVITS API server for Sylvia TTS

Extract GPT-SoVITS repo to the root of this project to make it work!

Also need valid python installation on runtime folder in order to function.

db.zip -> /
faster-whisper-large-v3 -> /sylvia-tts-api-server/tools/asr/models/faster-whisper-large-v3
G2PWModel.zip -> /sylvia-tts-api-server/GPT_SoVITS/text/G2PWModel
models.zip -> /sylvia-tts-api-server/models
pretrained_models.zip -> /sylvia-tts-api-server/GPT_SoVITS/pretrained_models
recordings.zip -> /sylvia-tts-file-server/recordings
sylvia-tts-api-server.zip -> /sylvia-tts-api-server

.\runtime\python.exe -m pip install -r requirements.txt will install requirements.

.\runtime\python.exe webui.py will bring up web UI for training.

.\runtime\python.exe api_infer.py will boot up inference API server. (Bring up webui.py first to set up paths.)

- Use slicer and ASR to prepare dataset.
