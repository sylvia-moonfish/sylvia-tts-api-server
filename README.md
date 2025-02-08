# GPT-SoVITS API server for Sylvia TTS

Extract GPT-SoVITS repo to the root of this project to make it work!

Also need valid python installation on runtime folder in order to function.

.\runtime\python.exe -m pip install -r requirements.txt will install requirements.

.\runtime\python.exe webui.py will bring up web UI for training.

.\runtime\python.exe api_infer.py will boot up inference API server. (Bring up webui.py first to set up paths.)

- Use slicer and ASR to prepare dataset.
