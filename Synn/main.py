"""Entry point for the modern Synn assistant."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import queue
import signal
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tkinter as tk
    from tkinter import scrolledtext
except Exception:  # pragma: no cover - Tk may be unavailable in CI
    tk = None
    scrolledtext = None

try:
    import speech_recognition as sr
except Exception:  # pragma: no cover
    sr = None

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None

try:
    from fastapi import FastAPI, Body
    from fastapi.middleware.cors import CORSMiddleware
except Exception:  # pragma: no cover
    FastAPI = None  # type: ignore
    Body = None  # type: ignore
    CORSMiddleware = None  # type: ignore

try:
    import uvicorn
except Exception:  # pragma: no cover
    uvicorn = None  # type: ignore

from Real_Cyn_Elliot import RealCynElliotCore


class EmotionState:
    """Track valence/arousal and derive a categorical label."""

    def __init__(self) -> None:
        self.valence: float = 0.0
        self.arousal: float = 0.0

    def apply(self, delta: Dict[str, float]) -> None:
        self.valence = self._clamp(self.valence + delta.get("valence", 0.0))
        self.arousal = self._clamp(self.arousal + delta.get("arousal", 0.0))

    @staticmethod
    def _clamp(value: float) -> float:
        return max(-1.0, min(1.0, value))

    @property
    def label(self) -> str:
        if self.valence > 0.2:
            return "happy"
        if self.valence < -0.2:
            return "sad"
        if self.arousal > 0.4:
            return "excited"
        if self.arousal < -0.4:
            return "calm"
        return "neutral"

    def snapshot(self) -> Dict[str, float]:
        return {"valence": self.valence, "arousal": self.arousal, "label": self.label}


class DialogueMemory:
    """Persist dialogue/emotion data locally."""

    def __init__(self, dialogue_path: Path, emotion_path: Path, max_history: int = 10) -> None:
        self.dialogue_path = dialogue_path
        self.emotion_path = emotion_path
        self.dialogue_history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.dialogue_path.parent.mkdir(parents=True, exist_ok=True)
        self.emotion_path.parent.mkdir(parents=True, exist_ok=True)

    def add_dialogue(self, speaker: str, text: str, emotion: str) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "speaker": speaker,
            "text": text,
            "emotion": emotion,
        }
        self.dialogue_history.append(entry)
        self.dialogue_history = self.dialogue_history[-self.max_history :]
        with self.dialogue_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def add_emotion(self, state: Dict[str, float]) -> None:
        payload = {"timestamp": datetime.utcnow().isoformat(), **state}
        with self.emotion_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def remember(self, message: str, source: str = "dialogue") -> Dict[str, Any]:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "memory": message,
            "source": source,
        }
        return entry


class SynnGUI:
    """Tkinter front-end for live conversations."""

    def __init__(self, on_message: callable) -> None:
        self._on_message = on_message
        self._thread: Optional[threading.Thread] = None
        self._root: Optional[tk.Tk] = None
        self._output: Any = None
        self._input: Any = None
        self._send_button: Any = None
        self._ready = threading.Event()
        self._shutdown = threading.Event()

    def start(self) -> None:
        if tk is None:
            logging.warning("Tkinter not available; GUI will not start.")
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        try:
            self._root = tk.Tk()
            self._root.title("Synn Console")
            self._root.protocol("WM_DELETE_WINDOW", self.close)

            self._output = scrolledtext.ScrolledText(self._root, wrap=tk.WORD, width=80, height=24)
            self._output.pack(padx=10, pady=10)

            self._input = tk.Entry(self._root, width=80)
            self._input.pack(padx=10, pady=(0, 10))
            self._input.bind("<Return>", self._handle_send)

            self._send_button = tk.Button(self._root, text="Send", command=self._handle_send)
            self._send_button.pack(pady=(0, 10))
            self._ready.set()
            self._root.mainloop()
        except Exception as exc:  # pragma: no cover
            logging.exception("GUI failure: %s", exc)
            self._ready.set()

    def _handle_send(self, event: Optional[Any] = None) -> None:
        if self._input is None:
            return
        text = self._input.get().strip()
        if not text:
            return
        self._input.delete(0, tk.END)
        self._on_message(text)

    def display(self, speaker: str, message: str) -> None:
        if self._output is None:
            return
        def _append() -> None:
            self._output.configure(state=tk.NORMAL)
            self._output.insert(tk.END, f"{speaker}: {message}\n")
            self._output.see(tk.END)
            self._output.configure(state=tk.DISABLED)
        self._output.after(0, _append)

    def close(self) -> None:
        self._shutdown.set()
        if self._root is not None:
            self._root.after(0, self._root.destroy)
        if self._thread is not None:
            self._thread.join(timeout=2)


class SpeechService:
    """Speech recognition and synthesis helpers."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.recognizer = sr.Recognizer() if sr else None
        mic: Optional[Any] = None
        if sr:
            try:
                mic = sr.Microphone()
            except Exception as exc:
                logging.warning("Microphone unavailable: %s", exc)
        self.microphone = mic
        self.engine = pyttsx3.init() if pyttsx3 else None
        self.voice_rate = config.get("voice_rate", 175)
        self.volume = config.get("volume", 0.8)
        self.voice_id = config.get("voice")
        if self.engine:
            self.engine.setProperty("rate", self.voice_rate)
            self.engine.setProperty("volume", self.volume)
            if self.voice_id is not None:
                try:
                    self.engine.setProperty("voice", self.voice_id)
                except Exception:
                    logging.warning("Requested voice id unavailable: %s", self.voice_id)

    async def listen(self, wake_word: str) -> Optional[str]:
        if not self.recognizer or not self.microphone:
            await asyncio.sleep(1)
            return None
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.3)
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=10)
            text = await asyncio.to_thread(self.recognizer.recognize_google, audio)
            if text.lower().startswith(wake_word.lower()):
                text = text[len(wake_word) :].strip()
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            logging.info("Speech unintelligible")
            return None
        except Exception as exc:
            logging.exception("Speech recognition error: %s", exc)
            return None

    async def speak(self, text: str) -> None:
        if not self.engine:
            return
        try:
            await asyncio.to_thread(self._speak_blocking, text)
        except Exception as exc:
            logging.exception("TTS error: %s", exc)

    def _speak_blocking(self, text: str) -> None:
        self.engine.say(text)
        self.engine.runAndWait()

    def shutdown(self) -> None:
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass


class SynnCore:
    """Async orchestrator for all Synn subsystems."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self.config = self._load_config()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.stop_event = asyncio.Event()
        self.core = RealCynElliotCore()
        logging.debug("Core traits cached: %s", self.core._cached_traits)

        logging_paths = self.config.get("logging", {})
        self.dialogue_memory = DialogueMemory(
            dialogue_path=config_path.parent / logging_paths.get("dialogue_log", "logs/dialogue_log.jsonl"),
            emotion_path=config_path.parent / logging_paths.get("emotion_log", "logs/emotion_log.jsonl"),
            max_history=self.config.get("dialogue", {}).get("max_history", 10),
        )
        self.emotion = EmotionState()
        self.speech = SpeechService(self.config.get("speech", {}))
        self.gui = SynnGUI(self._handle_gui_message)
        self.gui_queue: "queue.Queue[str]" = queue.Queue()
        self.history: List[Dict[str, str]] = []
        self._api_server: Optional[uvicorn.Server] = None
        self.api_app = self._create_api()
        self._tasks: List[asyncio.Task[Any]] = []
        self._reflection_path = config_path.parent / logging_paths.get("reflection_log", "logs/reflection.log")
        self.wake_word = self.config.get("wake_word", "synn")

    def _load_config(self) -> Dict[str, Any]:
        with self.config_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def _create_api(self) -> Optional[FastAPI]:
        if FastAPI is None or CORSMiddleware is None:
            logging.warning("FastAPI not installed; API endpoints disabled.")
            return None
        app = FastAPI(title="Synn API", version="1.0.0")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.post("/speak")
        async def speak_endpoint(message: str = Body(..., embed=True)) -> Dict[str, Any]:
            response = await self.process_text(message, source="api")
            return {"response": response, "emotion": self.emotion.snapshot()}

        @app.get("/emotion")
        async def emotion_endpoint() -> Dict[str, Any]:
            return self.emotion.snapshot()

        @app.post("/remember")
        async def remember_endpoint(message: str = Body(..., embed=True)) -> Dict[str, Any]:
            entry = self.core.remember(message)
            self.dialogue_memory.add_dialogue("memory", message, self.emotion.label)
            return entry

        @app.get("/analyze_face")
        async def analyze_face_endpoint() -> Dict[str, Any]:
            return {"status": "pending", "detail": "Vision module not connected"}

        return app

    def _handle_gui_message(self, text: str) -> None:
        self.gui_queue.put(text)

    async def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.gui.start()
        task_specs = [
            asyncio.create_task(self._gui_loop(), name="gui-loop"),
            asyncio.create_task(self._speech_loop(), name="speech-loop"),
        ]
        if self.api_app is not None and uvicorn is not None:
            task_specs.append(asyncio.create_task(self._run_api(), name="api-server"))
        else:
            logging.warning("API server not started; missing FastAPI or uvicorn.")
        self._tasks.extend(task_specs)
        logging.info("SynnCore started. Awaiting input.")

    async def stop(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        logging.info("Stopping SynnCore...")
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self.speech.shutdown()
        self.gui.close()
        await asyncio.to_thread(self._write_reflection)
        logging.info("SynnCore stopped cleanly.")

    async def _run_api(self) -> None:
        if uvicorn is None or self.api_app is None:
            return
        config = self.config.get("api", {})
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8002)
        server = uvicorn.Server(uvicorn.Config(self.api_app, host=host, port=port, log_level="info", loop="asyncio"))
        self._api_server = server
        try:
            await server.serve()
        except asyncio.CancelledError:
            server.should_exit = True
            raise

    async def _gui_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                text = await asyncio.to_thread(self.gui_queue.get, True, 0.1)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue
            await self.process_text(text, source="gui")

    async def _speech_loop(self) -> None:
        while not self.stop_event.is_set():
            text = await self.speech.listen(self.wake_word)
            if not text:
                continue
            await self.process_text(text, source="speech")

    async def process_text(self, text: str, source: str = "text") -> str:
        sentiment = self._simple_sentiment(text)
        self.dialogue_memory.add_dialogue("user", text, sentiment)
        response, delta = self.core.generate_response(text, sentiment)
        self.emotion.apply(delta)
        self.dialogue_memory.add_emotion(self.emotion.snapshot())
        self.history.append({"user": text, "response": response, "emotion": self.emotion.label})
        self.dialogue_memory.add_dialogue("synn", response, self.emotion.label)
        if self.gui:
            self.gui.display("You", text)
            self.gui.display("Synn", response)
        await self.speech.speak(response)
        logging.info("User: %s", text)
        logging.info("Synn: %s", response)
        return response

    @staticmethod
    def _simple_sentiment(text: str) -> str:
        lowered = text.lower()
        positive = {"love", "great", "good", "thanks", "awesome"}
        negative = {"hate", "bad", "angry", "upset", "sad"}
        if any(word in lowered for word in positive):
            return "positive"
        if any(word in lowered for word in negative):
            return "negative"
        return "neutral"

    def _write_reflection(self) -> None:
        reflection = self.core.reflect(self.history)
        line = f"{datetime.utcnow().isoformat()} {reflection}\n"
        self._reflection_path.parent.mkdir(parents=True, exist_ok=True)
        with self._reflection_path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def configure_logging(base_path: Path) -> None:
    log_path = base_path / "logs" / f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


async def async_main(args: argparse.Namespace) -> None:
    base_path = Path(__file__).resolve().parent
    configure_logging(base_path)
    config_path = base_path / "config.json"
    core = SynnCore(config_path)
    await core.start()

    loop = asyncio.get_running_loop()
    stop_future = loop.create_future()

    def _signal_handler(*_: Any) -> None:
        if not stop_future.done():
            stop_future.set_result(True)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _signal_handler)
        except NotImplementedError:  # pragma: no cover
            signal.signal(sig, lambda *_: asyncio.create_task(core.stop()))

    if args.headless_test:
        await asyncio.sleep(2)
        await core.stop()
        return

    await stop_future
    await core.stop()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Synn assistant")
    parser.add_argument("--headless-test", action="store_true", help="Run briefly without launching blocking UI")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
