"""Точка входа: приложение, трей, глобальные хоткеи, склейка модулей."""
from __future__ import annotations

import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QMessageBox

from clipboard_watcher import ClipboardWatcher
from config import Config, app_dir
from llm_client import LLMError, assistant_turn, client_from_config, user_turn
from settings_dialog import SettingsDialog
from tray import Tray
from widget import OPACITY_STEP, ChatWidget

log = logging.getLogger("llm_widget")


def setup_logging() -> None:
    path = os.path.join(app_dir(), "app.log")
    handlers: list[logging.Handler] = [
        RotatingFileHandler(path, maxBytes=512_000, backupCount=2, encoding="utf-8")
    ]
    if sys.stderr is not None:
        # консоль на Windows часто в cp1251 — не роняем логгер на юникоде
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
            except (ValueError, OSError):
                pass
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
    )


class Bridge(QObject):
    """Сигналы из фоновых потоков (хоткеи, HTTP) в поток Qt."""

    toggle_widget = pyqtSignal()
    open_settings = pyqtSignal()
    toggle_watch = pyqtSignal()
    hide_widget = pyqtSignal()
    quit_app = pyqtSignal()
    opacity_delta = pyqtSignal(float)
    answer_ready = pyqtSignal(int, str)
    answer_failed = pyqtSignal(int, str)


class App:
    def __init__(self) -> None:
        self.qapp = QApplication(sys.argv)
        self.qapp.setApplicationName("LLM Widget")
        self.qapp.setQuitOnLastWindowClosed(False)

        self.cfg = Config()
        self.request_id = 0
        # история переписки по текущему скриншоту (для уточняющих вопросов)
        self.history: list[dict] = []
        self.settings_dialog: SettingsDialog | None = None
        self.hotkeys_ok = False

        self.bridge = Bridge()
        self.widget = ChatWidget(self.cfg)
        self.tray = Tray()
        self.watcher = ClipboardWatcher(bool(self.cfg.get("watch_clipboard", True)))

        self._wire()
        self.tray.set_watching(self.watcher.enabled)
        self.tray.show()
        self.widget.set_hide_hotkey(self.cfg.hotkey("hide_to_tray"))
        self.widget.show()
        self._register_hotkeys()

        if self.cfg.first_run or not self.cfg.get("api_token"):
            QTimer.singleShot(300, self.open_settings)

    # --- связи ------------------------------------------------------------
    def _wire(self) -> None:
        self.widget.settings_requested.connect(self.open_settings)
        self.widget.hide_requested.connect(self.hide_widget)
        self.widget.question_asked.connect(self.on_question)

        self.tray.toggle_widget_requested.connect(self.toggle_widget)
        self.tray.settings_requested.connect(self.open_settings)
        self.tray.toggle_watch_requested.connect(self.toggle_watch)
        self.tray.quit_requested.connect(self.quit)

        self.watcher.image_captured.connect(self.on_image)

        self.bridge.toggle_widget.connect(self.toggle_widget)
        self.bridge.open_settings.connect(self.open_settings)
        self.bridge.toggle_watch.connect(self.toggle_watch)
        self.bridge.hide_widget.connect(self.hide_widget)
        self.bridge.quit_app.connect(self.quit)
        self.bridge.opacity_delta.connect(self.bump_opacity)
        self.bridge.answer_ready.connect(self.on_answer)
        self.bridge.answer_failed.connect(self.on_failure)

    # --- запрос к LLM -----------------------------------------------------
    def on_image(self, png: bytes) -> None:
        if not self.cfg.get("api_token"):
            self.widget.show_error(
                "Не задан API-токен. Нажмите ⚙ и вставьте ключ `vk-...`."
            )
            self.tray.set_status("Нужен токен")
            return

        prompt = str(self.cfg.get("system_prompt", ""))
        self.history = [user_turn(prompt, png)]
        self.widget.set_model_label(str(self.cfg.get("model", "")))
        self.widget.show_thinking(reset=True)
        self._dispatch("Отправляю…")

    def on_question(self, text: str) -> None:
        """Уточнение из строки ввода: продолжаем тот же диалог."""
        if not self.cfg.get("api_token"):
            self.widget.add_user_message(text)
            self.widget.show_error("Не задан API-токен. Нажмите ⚙ и вставьте ключ `vk-...`.")
            return
        self.history.append(user_turn(text))
        self.widget.add_user_message(text)
        self.widget.show_thinking()
        log.info("Уточнение: %s", text[:80])
        self._dispatch("Уточняю…")

    def _dispatch(self, status: str) -> None:
        """Отправка текущей истории в фоне."""
        self.request_id += 1
        req = self.request_id
        client = client_from_config(self.cfg)
        turns = list(self.history)
        self.tray.set_status(status)

        def run() -> None:
            try:
                answer = client.ask(turns)
                self.bridge.answer_ready.emit(req, answer)
            except LLMError as exc:
                self.bridge.answer_failed.emit(req, str(exc))
            except Exception as exc:
                log.exception("Непредвиденный сбой запроса")
                self.bridge.answer_failed.emit(req, f"Непредвиденная ошибка: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def on_answer(self, req: int, text: str) -> None:
        if req != self.request_id:
            return  # пришёл ответ на устаревший запрос
        self.history.append(assistant_turn(text))
        self.widget.show_answer(text)
        self.tray.set_status("Готово")

    def on_failure(self, req: int, message: str) -> None:
        if req != self.request_id:
            return
        self.widget.show_error(message)
        self.tray.set_status(f"Ошибка: {message[:60]}")

    # --- действия ---------------------------------------------------------
    def toggle_widget(self) -> None:
        visible = self.widget.toggle_visibility()
        log.info("Виджет %s", "показан" if visible else "скрыт")

    def hide_widget(self) -> None:
        self.widget.hide()

    def bump_opacity(self, delta: float) -> None:
        value = self.widget.bump_opacity(delta)
        self.tray.set_status(f"Прозрачность {int(value * 100)} %")
        if self.settings_dialog is not None and self.settings_dialog.isVisible():
            self.settings_dialog.sync_opacity(value)

    def toggle_watch(self) -> None:
        enabled = self.watcher.toggle()
        self.cfg.set("watch_clipboard", enabled, save=True)
        self.tray.set_watching(enabled)
        self.tray.set_status("Жду скриншот…" if enabled else "Пауза")

    def open_settings(self) -> None:
        if self.settings_dialog is not None and self.settings_dialog.isVisible():
            self.settings_dialog.raise_()
            self.settings_dialog.activateWindow()
            return
        dialog = SettingsDialog(self.cfg)
        self.settings_dialog = dialog
        dialog.opacity_preview.connect(lambda v: self.widget.set_opacity(v, save=False))
        dialog.applied.connect(self.apply_config)
        dialog.finished.connect(lambda _: setattr(self, "settings_dialog", None))
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def apply_config(self) -> None:
        self.widget.set_opacity(float(self.cfg.get("opacity", 0.85)), save=False)
        self.widget.set_always_on_top(bool(self.cfg.get("always_on_top", True)))
        self.widget.set_model_label(str(self.cfg.get("model", "")))
        watch = bool(self.cfg.get("watch_clipboard", True))
        self.watcher.set_enabled(watch)
        self.tray.set_watching(watch)
        self.tray.set_status("Жду скриншот…" if watch else "Пауза")

        # хоткей сворачивания мог измениться — перевешиваем и показываем его на ✕
        hide_combo = self.cfg.hotkey("hide_to_tray")
        self.widget.set_hide_hotkey(hide_combo)
        failed = self._register_hotkeys()
        if hide_combo and hide_combo in failed:
            self.tray.set_status(f"Хоткей {hide_combo} занят другой программой")
            log.warning("Хоткей сворачивания занят: %s", hide_combo)

    def quit(self) -> None:
        log.info("Выход")
        self.cfg.save()
        if self.hotkeys_ok:
            try:
                import keyboard

                keyboard.unhook_all()
            except Exception:  # noqa: BLE001
                pass
        self.tray.hide()
        self.qapp.quit()

    # --- хоткеи -----------------------------------------------------------
    def _register_hotkeys(self) -> list[str]:
        """Перевешивает все хоткеи из конфига. Возвращает список неудачных."""
        try:
            import keyboard
        except ImportError:
            log.warning("Пакет keyboard не установлен — глобальные хоткеи недоступны")
            self.tray.set_status("Жду скриншот… (нет хоткеев)")
            return []
        if self.hotkeys_ok:
            try:
                keyboard.unhook_all()
            except Exception:  # noqa: BLE001
                log.warning("Не удалось снять прежние хоткеи", exc_info=True)
        mapping = {
            "toggle_widget": self.bridge.toggle_widget.emit,
            "open_settings": self.bridge.open_settings.emit,
            "toggle_watch": self.bridge.toggle_watch.emit,
            "hide_to_tray": self.bridge.hide_widget.emit,
            "quit": self.bridge.quit_app.emit,
            "opacity_up": lambda: self.bridge.opacity_delta.emit(OPACITY_STEP),
            "opacity_down": lambda: self.bridge.opacity_delta.emit(-OPACITY_STEP),
        }
        registered = 0
        failed: list[str] = []
        for name, callback in mapping.items():
            combo = self.cfg.hotkey(name)
            if not combo:
                continue
            try:
                keyboard.add_hotkey(combo, callback, suppress=False)
                registered += 1
            except Exception as exc:  # noqa: BLE001
                log.warning("Не удалось повесить хоткей %s (%s): %s", name, combo, exc)
                failed.append(combo)
        self.hotkeys_ok = registered > 0
        log.info("Зарегистрировано хоткеев: %d", registered)
        if registered == 0:
            self.tray.set_status("Жду скриншот… (хоткеи не активны)")
        return failed

    def run(self) -> int:
        return self.qapp.exec()


def main() -> int:
    setup_logging()
    log.info("Старт LLM Widget, каталог: %s", app_dir())
    app = App()
    if not app.tray.isSystemTrayAvailable():
        QMessageBox.warning(
            None, "LLM Widget", "Системный трей недоступен: иконки в трее не будет."
        )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
