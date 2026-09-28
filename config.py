"""Загрузка / сохранение config.json."""
from __future__ import annotations

import copy
import json
import logging
import os
import sys
from typing import Any

log = logging.getLogger(__name__)


def app_dir() -> str:
    """Каталог приложения (рядом с .exe при сборке PyInstaller --onefile)."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


CONFIG_PATH = os.path.join(app_dir(), "config.json")

DEFAULTS: dict[str, Any] = {
    "api_token": "",
    "base_url": "https://vibecode.moe/v1",
    "model": "claude-sonnet-5",
    # auto: claude-* → /v1/messages (Anthropic), остальные → /v1/chat/completions
    "api_format": "auto",
    "max_tokens": 1024,
    "system_prompt": "Опиши, что на скриншоте. Отвечай кратко на русском.",
    "opacity": 0.85,
    "always_on_top": True,
    "watch_clipboard": True,
    "widget_pos": [200, 200],
    "widget_size": [420, 320],
    "hotkeys": {
        "toggle_widget": "ctrl+shift+h",
        "open_settings": "ctrl+shift+s",
        "toggle_watch": "ctrl+shift+p",
        "quit": "ctrl+shift+q",
        "hide_to_tray": "ctrl+shift+x",
        "opacity_up": "ctrl+alt+up",
        "opacity_down": "ctrl+alt+down",
    },
}


class Config:
    """Словарь настроек с записью в JSON рядом с приложением."""

    def __init__(self, path: str = CONFIG_PATH) -> None:
        self.path = path
        self.first_run = not os.path.exists(path)
        self.data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        if self.first_run:
            self.save()
            log.info("Создан шаблон конфига: %s", path)
        else:
            self.load()

    # --- IO ---------------------------------------------------------------
    def load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Не удалось прочитать %s (%s), беру значения по умолчанию", self.path, exc)
            return
        if not isinstance(loaded, dict):
            log.warning("config.json не является объектом, беру значения по умолчанию")
            return
        for key, value in loaded.items():
            if key == "hotkeys" and isinstance(value, dict):
                self.data["hotkeys"] = {**DEFAULTS["hotkeys"], **value}
            else:
                self.data[key] = value

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh, ensure_ascii=False, indent=2)
            os.replace(tmp, self.path)
        except OSError as exc:
            log.error("Не удалось сохранить конфиг: %s", exc)

    # --- доступ -----------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        if default is None:
            default = DEFAULTS.get(key)
        return self.data.get(key, default)

    def set(self, key: str, value: Any, *, save: bool = False) -> None:
        self.data[key] = value
        if save:
            self.save()

    def update(self, values: dict[str, Any], *, save: bool = True) -> None:
        self.data.update(values)
        if save:
            self.save()

    def hotkey(self, name: str) -> str:
        return self.get("hotkeys", {}).get(name, DEFAULTS["hotkeys"].get(name, ""))

    def set_hotkey(self, name: str, combo: str, *, save: bool = False) -> None:
        """Пустая строка = хоткей отключён."""
        hotkeys = dict(self.get("hotkeys", {}))
        hotkeys[name] = combo.strip().lower()
        self.data["hotkeys"] = hotkeys
        if save:
            self.save()
