# Synn Core

Synn is an asynchronous AI assistant that blends the provided Cyn Elliot cognitive core with speech, GUI, memory logging, and FastAPI control endpoints. Run locally to interact with Synn through voice or text.

## Features
- Async orchestrator bootstrapping speech I/O, Tkinter GUI, FastAPI, and reflection logging.
- Integration of the original Cyn Elliot cognitive blueprint for dialogue grounding.
- Speech recognition (SpeechRecognition + Vosk/Google) with graceful fallbacks.
- Text-to-speech via pyttsx3 when available.
- Persistent dialogue and emotion logging in `logs/`.
- REST API for `/speak`, `/emotion`, `/remember`, and `/analyze_face` commands.

## Requirements
Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

This launches the orchestrator, speech services, FastAPI server, and GUI window. Use the text box to converse in real time.

## Packaging
Build a standalone binary with PyInstaller:

```bash
pyinstaller --onefile --noconsole main.py
```

## Logs
Conversation transcripts and emotion state history are written to the `logs/` directory. Reflection summaries are appended on shutdown.
