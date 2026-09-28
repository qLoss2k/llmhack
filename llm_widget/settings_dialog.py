"""Окно настроек: токен, base url, модель, system prompt, прозрачность."""
from __future__ import annotations

import logging
import threading

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QKeySequence
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)

from llm_client import FORMAT_AUTO, FORMATS, LLMClient, LLMError, detect_format

# на случай, когда список с сервера ещё не подтянут
KNOWN_MODELS = (
    "claude-sonnet-5",
    "claude-opus-5",
    "claude-haiku-4-5",
    "claude-fable-5-1",
    "gpt-6-sol",
    "gemini-3.1-pro-preview",
)

FORMAT_LABELS = {
    FORMAT_AUTO: "auto — по модели (claude → /messages)",
    "anthropic": "anthropic — POST /v1/messages",
    "openai": "openai — POST /v1/chat/completions",
}

# Qt-имена клавиш → имена, которые понимает пакет keyboard
_KEY_ALIASES = {
    "Up": "up", "Down": "down", "Left": "left", "Right": "right",
    "PgUp": "page up", "PgDown": "page down", "Ins": "insert", "Del": "delete",
    "Esc": "esc", "Return": "enter", "Enter": "enter", "Space": "space",
    "Backspace": "backspace", "Tab": "tab", "Home": "home", "End": "end",
}

log = logging.getLogger(__name__)


class HotkeyEdit(QLineEdit):
    """Поле, которое записывает нажатую комбинацию вместо ввода текста."""

    changed = pyqtSignal(str)

    def __init__(self, combo: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setPlaceholderText("нажмите комбинацию…")
        self.setToolTip(
            "Щёлкните и нажмите комбинацию, например Ctrl+Shift+X.\n"
            "Нужен хотя бы один модификатор (Ctrl / Alt / Shift / Win).\n"
            "Backspace или Delete — очистить (хоткей отключён)."
        )
        self.set_combo(combo)

    def set_combo(self, combo: str) -> None:
        self._combo = (combo or "").strip().lower()
        self.setText(self._pretty(self._combo))

    def combo(self) -> str:
        return self._combo

    @staticmethod
    def _pretty(combo: str) -> str:
        if not combo:
            return ""
        return "+".join(part.capitalize() for part in combo.split("+"))

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        key = event.key()
        mods = event.modifiers()

        if key in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete) and not mods:
            self.set_combo("")
            self.changed.emit("")
            return
        # сами модификаторы комбинацией не считаются — ждём основную клавишу
        if key in (Qt.Key.Key_Control, Qt.Key.Key_Alt, Qt.Key.Key_Shift,
                   Qt.Key.Key_Meta, Qt.Key.Key_unknown):
            return

        parts = []
        if mods & Qt.KeyboardModifier.ControlModifier:
            parts.append("ctrl")
        if mods & Qt.KeyboardModifier.AltModifier:
            parts.append("alt")
        if mods & Qt.KeyboardModifier.ShiftModifier:
            parts.append("shift")
        if mods & Qt.KeyboardModifier.MetaModifier:
            parts.append("windows")
        if not parts:
            # без модификатора хоткей перехватывал бы обычный ввод во всей системе
            self.setText("нужен Ctrl / Alt / Shift / Win")
            return

        name = QKeySequence(key).toString()
        name = _KEY_ALIASES.get(name, name.lower())
        if not name:
            return
        parts.append(name)

        self.set_combo("+".join(parts))
        self.changed.emit(self._combo)


class SettingsDialog(QDialog):
    """Изменения применяются сразу после «Сохранить» (прозрачность — на лету)."""

    applied = pyqtSignal()
    opacity_preview = pyqtSignal(float)
    _test_done = pyqtSignal(bool, str)
    _models_loaded = pyqtSignal(list)

    def __init__(self, cfg, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self._initial_opacity = float(cfg.get("opacity", 0.85))
        self.setWindowTitle("Настройки — LLM Widget")
        self.setMinimumWidth(460)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        self._build_ui()
        self._load_values()
        self._test_done.connect(self._on_test_done)
        self._models_loaded.connect(self._on_models_loaded)

    # --- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("vk-...")
        self.token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.eye_btn = QPushButton("👁")
        self.eye_btn.setFixedWidth(34)
        self.eye_btn.setCheckable(True)
        self.eye_btn.setToolTip("Показать / скрыть токен")
        self.eye_btn.toggled.connect(self._toggle_echo)
        token_row = QHBoxLayout()
        token_row.setContentsMargins(0, 0, 0, 0)
        token_row.addWidget(self.token_edit, 1)
        token_row.addWidget(self.eye_btn)
        form.addRow("API-токен:", token_row)

        self.base_url_edit = QLineEdit()
        self.base_url_edit.setPlaceholderText("https://vibecode.moe/v1")
        form.addRow("Base URL:", self.base_url_edit)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(KNOWN_MODELS)
        self.model_combo.lineEdit().setPlaceholderText("claude-sonnet-5")
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        form.addRow("Модель:", self.model_combo)

        self.format_combo = QComboBox()
        for fmt in FORMATS:
            self.format_combo.addItem(FORMAT_LABELS.get(fmt, fmt), fmt)
        self.format_combo.setToolTip(
            "claude-* у vibecode.moe отвечают только на /v1/messages (формат Anthropic).\n"
            "auto подбирает эндпоинт по имени модели."
        )
        form.addRow("Формат API:", self.format_combo)

        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(64, 32000)
        self.max_tokens_spin.setSingleStep(256)
        self.max_tokens_spin.setToolTip("Максимальная длина ответа модели.")
        form.addRow("max_tokens:", self.max_tokens_spin)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setMinimumHeight(90)
        form.addRow("System prompt:", self.prompt_edit)

        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setSingleStep(5)
        self.opacity_value = QLabel("85 %")
        self.opacity_value.setFixedWidth(46)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row = QHBoxLayout()
        opacity_row.setContentsMargins(0, 0, 0, 0)
        opacity_row.addWidget(self.opacity_slider, 1)
        opacity_row.addWidget(self.opacity_value)
        form.addRow("Прозрачность:", opacity_row)

        self.hide_hotkey_edit = HotkeyEdit()
        self.hotkey_clear_btn = QPushButton("Очистить")
        self.hotkey_clear_btn.setToolTip("Отключить этот хоткей")
        self.hotkey_clear_btn.clicked.connect(lambda: self.hide_hotkey_edit.set_combo(""))
        hotkey_row = QHBoxLayout()
        hotkey_row.setContentsMargins(0, 0, 0, 0)
        hotkey_row.addWidget(self.hide_hotkey_edit, 1)
        hotkey_row.addWidget(self.hotkey_clear_btn)
        form.addRow("Свернуть в трей:", hotkey_row)

        self.on_top_check = QCheckBox("Всегда поверх окон")
        self.watch_check = QCheckBox("Следить за буфером обмена")
        form.addRow("", self.on_top_check)
        form.addRow("", self.watch_check)
        root.addLayout(form)

        hint = QLabel(
            "Остальные хоткеи (показать/скрыть, настройки, пауза, выход) — "
            "в разделе <code>hotkeys</code> файла <code>config.json</code>."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888888;")
        root.addWidget(hint)

        self.test_label = QLabel("")
        self.test_label.setWordWrap(True)
        root.addWidget(self.test_label)

        buttons = QDialogButtonBox()
        self.test_btn = buttons.addButton(
            "Проверить соединение", QDialogButtonBox.ButtonRole.ActionRole
        )
        buttons.addButton(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        self.test_btn.clicked.connect(self._test_connection)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _toggle_echo(self, shown: bool) -> None:
        mode = QLineEdit.EchoMode.Normal if shown else QLineEdit.EchoMode.Password
        self.token_edit.setEchoMode(mode)

    def _on_opacity_changed(self, value: int) -> None:
        self.opacity_value.setText(f"{value} %")
        self.opacity_preview.emit(value / 100.0)

    def _on_model_changed(self, model: str) -> None:
        """В режиме auto подсказываем, какой эндпоинт поедет."""
        if self.format_combo.currentData() != FORMAT_AUTO:
            return
        endpoint = "/messages" if detect_format(model) == "anthropic" else "/chat/completions"
        self.format_combo.setItemText(0, f"auto → {endpoint}")

    # --- значения ---------------------------------------------------------
    def _load_values(self) -> None:
        cfg = self.cfg
        self.token_edit.setText(str(cfg.get("api_token", "")))
        self.base_url_edit.setText(str(cfg.get("base_url", "")))
        self.model_combo.setCurrentText(str(cfg.get("model", "")))
        fmt = str(cfg.get("api_format", FORMAT_AUTO))
        index = self.format_combo.findData(fmt)
        self.format_combo.setCurrentIndex(index if index >= 0 else 0)
        self._on_model_changed(self.model_combo.currentText())
        self.max_tokens_spin.setValue(int(cfg.get("max_tokens", 1024)))
        self.prompt_edit.setPlainText(str(cfg.get("system_prompt", "")))
        self.opacity_slider.setValue(int(round(float(cfg.get("opacity", 0.85)) * 100)))
        self.opacity_value.setText(f"{self.opacity_slider.value()} %")
        self.on_top_check.setChecked(bool(cfg.get("always_on_top", True)))
        self.watch_check.setChecked(bool(cfg.get("watch_clipboard", True)))
        self.hide_hotkey_edit.set_combo(cfg.hotkey("hide_to_tray"))

    def sync_opacity(self, value: float) -> None:
        """Подтянуть прозрачность, изменённую хоткеями, пока окно открыто."""
        percent = int(round(value * 100))
        if percent != self.opacity_slider.value():
            self.opacity_slider.blockSignals(True)
            self.opacity_slider.setValue(percent)
            self.opacity_slider.blockSignals(False)
            self.opacity_value.setText(f"{percent} %")

    def _values(self) -> dict:
        hotkeys = dict(self.cfg.get("hotkeys", {}))
        hotkeys["hide_to_tray"] = self.hide_hotkey_edit.combo()
        return {
            "hotkeys": hotkeys,
            "api_token": self.token_edit.text().strip(),
            "base_url": self.base_url_edit.text().strip() or "https://vibecode.moe/v1",
            "model": self.model_combo.currentText().strip() or "claude-sonnet-5",
            "api_format": self.format_combo.currentData() or FORMAT_AUTO,
            "max_tokens": int(self.max_tokens_spin.value()),
            "system_prompt": self.prompt_edit.toPlainText().strip(),
            "opacity": round(self.opacity_slider.value() / 100.0, 2),
            "always_on_top": self.on_top_check.isChecked(),
            "watch_clipboard": self.watch_check.isChecked(),
        }

    # --- кнопки -----------------------------------------------------------
    def accept(self) -> None:
        self.cfg.update(self._values())
        log.info("Настройки сохранены (модель=%s)", self.cfg.get("model"))
        self.applied.emit()
        super().accept()

    def reject(self) -> None:
        self.opacity_preview.emit(self._initial_opacity)
        super().reject()

    def _test_connection(self) -> None:
        values = self._values()
        if not values["api_token"]:
            self._show_test(False, "Введите API-токен.")
            return
        self.test_btn.setEnabled(False)
        self._show_test(True, "Проверяю…", neutral=True)
        client = LLMClient(
            base_url=values["base_url"],
            api_token=values["api_token"],
            model=values["model"],
            timeout=30,
            max_tokens=64,
            api_format=values["api_format"],
        )

        def run() -> None:
            try:
                answer = client.test_connection()
            except LLMError as exc:
                self._test_done.emit(False, str(exc))
                return
            except Exception as exc:  # на всякий случай — не падаем
                log.exception("Сбой проверки соединения")
                self._test_done.emit(False, f"Непредвиденная ошибка: {exc}")
                return
            # заодно подтянем актуальный список моделей в выпадашку
            try:
                self._models_loaded.emit(client.list_models())
            except (LLMError, Exception):  # noqa: BLE001 - список не критичен
                log.info("Список моделей получить не удалось", exc_info=True)
            self._test_done.emit(True, f"Соединение есть. Ответ модели: {answer[:80]}")

        threading.Thread(target=run, daemon=True).start()

    def _on_models_loaded(self, models: list) -> None:
        if not models:
            return
        current = self.model_combo.currentText()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models)
        self.model_combo.setCurrentText(current)
        self.model_combo.blockSignals(False)
        log.info("Список моделей обновлён: %d", len(models))

    def _on_test_done(self, ok: bool, message: str) -> None:
        self.test_btn.setEnabled(True)
        self._show_test(ok, message)

    def _show_test(self, ok: bool, message: str, *, neutral: bool = False) -> None:
        if neutral:
            color = "#888888"
            prefix = ""
        elif ok:
            color = "#2e7d32"
            prefix = "✓ "
        else:
            color = "#c62828"
            prefix = "✕ "
        self.test_label.setStyleSheet(f"color: {color};")
        self.test_label.setText(prefix + message)
