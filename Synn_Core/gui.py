"""Simple Tkinter-based console for Synn Core."""
from __future__ import annotations

import logging
import queue
import threading
from typing import Callable, Optional

try:  # pragma: no cover - optional dependency on GUI availability
    import tkinter as tk
    from tkinter import scrolledtext
except Exception:  # pragma: no cover
    tk = None  # type: ignore
    scrolledtext = None  # type: ignore

LOGGER = logging.getLogger(__name__)


class SynnConsoleGUI:
    """Minimal interface for testing conversations."""

    def __init__(self, on_user_input: Callable[[str], None], output_queue: "queue.Queue[str]", title: str = "Synn Console") -> None:
        self.on_user_input = on_user_input
        self.output_queue = output_queue
        self.title = title
        self.root: Optional[tk.Tk] = None
        self.text_area: Optional[scrolledtext.ScrolledText] = None
        self.input_field: Optional[tk.Entry] = None
        self.active = False

    def run(self) -> None:
        """Start the Tkinter main loop in the current thread."""

        if not tk:
            LOGGER.warning("Tkinter not available; GUI disabled")
            return
        self.root = tk.Tk()
        self.root.title(self.title)
        self.text_area = scrolledtext.ScrolledText(self.root, state="disabled", wrap=tk.WORD)
        self.text_area.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.input_field = tk.Entry(self.root)
        self.input_field.pack(padx=10, pady=(0, 10), fill=tk.X)
        self.input_field.bind("<Return>", self._on_submit)
        self.active = True
        self.root.protocol("WM_DELETE_WINDOW", self.stop)
        self.root.after(100, self._poll_output)
        self.root.mainloop()

    def display_response(self, message: str) -> None:
        """Append assistant output to the console widget."""

        if not self.text_area:
            return
        self.text_area.configure(state="normal")
        self.text_area.insert(tk.END, f"Synn: {message}\n")
        self.text_area.configure(state="disabled")
        self.text_area.see(tk.END)

    def stop(self) -> None:
        """Safely terminate the GUI."""

        self.active = False
        if self.root:
            self.root.quit()
            self.root.destroy()
            self.root = None

    def _on_submit(self, event: Optional[tk.Event] = None) -> None:  # type: ignore[override]
        if not self.input_field:
            return
        text = self.input_field.get().strip()
        if text:
            threading.Thread(target=self.on_user_input, args=(text,), daemon=True).start()
            self.input_field.delete(0, tk.END)
            if self.text_area:
                self.text_area.configure(state="normal")
                self.text_area.insert(tk.END, f"You: {text}\n")
                self.text_area.configure(state="disabled")
                self.text_area.see(tk.END)

    def _poll_output(self) -> None:
        if not self.active or not self.root:
            return
        try:
            while True:
                message = self.output_queue.get_nowait()
                self.display_response(message)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._poll_output)


__all__ = ["SynnConsoleGUI"]
