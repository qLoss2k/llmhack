"""Прозрачный виджет без рамки: перетаскивание, resize, markdown-ответ."""
from __future__ import annotations

import logging

from PyQt6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizeGrip,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

SPINNER_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
MIN_OPACITY = 0.10
MAX_OPACITY = 1.00
OPACITY_STEP = 0.05

CARD_QSS = """
#card {
    background-color: rgba(22, 24, 30, 218);
    border: 1px solid rgba(255, 255, 255, 46);
    border-radius: 12px;
}
#title {
    color: rgba(255, 255, 255, 190);
    font-size: 11px;
    font-weight: 600;
}
#status {
    color: rgba(150, 220, 255, 220);
    font-size: 11px;
}
QPushButton#tool {
    background: transparent;
    border: none;
    color: rgba(255, 255, 255, 170);
    font-size: 14px;
    padding: 0px;
    min-width: 22px;
    max-width: 22px;
    min-height: 22px;
    max-height: 22px;
    border-radius: 11px;
}
QPushButton#tool:hover { background: rgba(255, 255, 255, 38); color: white; }
QPushButton#tool:pressed { background: rgba(255, 255, 255, 64); }
QTextBrowser#view {
    background: transparent;
    border: none;
    color: rgba(240, 242, 248, 240);
    font-size: 13px;
    selection-background-color: rgba(90, 150, 255, 120);
}
QScrollBar:vertical {
    background: transparent; width: 8px; margin: 0px;
}
QScrollBar::handle:vertical {
    background: rgba(255, 255, 255, 55); border-radius: 4px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: rgba(255, 255, 255, 90); }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QLineEdit#ask {
    background-color: rgba(255, 255, 255, 22);
    border: 1px solid rgba(255, 255, 255, 40);
    border-radius: 9px;
    padding: 5px 8px;
    color: rgba(245, 247, 252, 245);
    font-size: 12px;
    selection-background-color: rgba(90, 150, 255, 140);
}
QLineEdit#ask:focus { border: 1px solid rgba(120, 180, 255, 170); }
QLineEdit#ask:disabled { color: rgba(255, 255, 255, 90); }
QPushButton#send {
    background-color: rgba(90, 150, 255, 120);
    border: none;
    border-radius: 9px;
    color: white;
    font-size: 13px;
    min-width: 30px;
    max-width: 30px;
    min-height: 26px;
    max-height: 26px;
}
QPushButton#send:hover { background-color: rgba(110, 170, 255, 170); }
QPushButton#send:disabled { background-color: rgba(255, 255, 255, 30); color: rgba(255,255,255,80); }
"""

DOC_QSS = """
body { color: #f0f2f8; }
h1, h2, h3, h4 { color: #ffffff; margin: 6px 0 4px 0; }
p { margin: 4px 0; }
ul, ol { margin: 4px 0 4px 16px; }
li { margin: 2px 0; }
a { color: #7fb4ff; }
code { background-color: #2b2f3a; color: #ffd9a0; font-family: Consolas, monospace; }
pre { background-color: #12141a; color: #e6e9f0; padding: 6px;
      font-family: Consolas, monospace; }
"""


class ChatWidget(QWidget):
    """Полупрозрачное окно поверх всех окон с ответом модели."""

    settings_requested = pyqtSignal()
    hide_requested = pyqtSignal()
    question_asked = pyqtSignal(str)

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self._answer_text = ""
        self._drag_offset: QPoint | None = None
        self._spinner_index = 0
        # история для отображения: список (роль, текст), роль = user|assistant|error
        self._transcript: list[tuple[str, str]] = []

        self.setWindowTitle("LLM Widget")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMinimumSize(300, 200)
        self.setStyleSheet(CARD_QSS)

        self._build_ui()
        self._restore_geometry()
        self.set_opacity(float(cfg.get("opacity", 0.85)), save=False)
        self.set_always_on_top(bool(cfg.get("always_on_top", True)), reshow=False)

        self._spinner = QTimer(self)
        self._spinner.setInterval(110)
        self._spinner.timeout.connect(self._tick_spinner)

        self._geometry_save = QTimer(self)
        self._geometry_save.setSingleShot(True)
        self._geometry_save.setInterval(600)
        self._geometry_save.timeout.connect(self._save_geometry)

        self.show_idle()

    # --- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame(self)
        card.setObjectName("card")
        outer.addWidget(card)

        root = QVBoxLayout(card)
        root.setContentsMargins(10, 8, 8, 6)
        root.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(4)
        self.title = QLabel(self.cfg.get("model", "claude-sonnet-5"))
        self.title.setObjectName("title")
        self.status = QLabel("")
        self.status.setObjectName("status")
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.status)

        self.btn_settings = self._tool_button("⚙", "Настройки")
        self.btn_copy = self._tool_button("📋", "Копировать ответ")
        self.btn_close = self._tool_button("✕", "Свернуть в трей")
        self.btn_settings.clicked.connect(self.settings_requested.emit)
        self.btn_copy.clicked.connect(self.copy_answer)
        self.btn_close.clicked.connect(self.hide_requested.emit)
        for btn in (self.btn_settings, self.btn_copy, self.btn_close):
            header.addWidget(btn)
        root.addLayout(header)

        self.view = QTextBrowser(card)
        self.view.setObjectName("view")
        self.view.setOpenExternalLinks(True)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.viewport().setAutoFillBackground(False)
        self.view.document().setDefaultStyleSheet(DOC_QSS)
        # перетаскивание важнее выделения текста: копирование — по кнопке 📋
        self.view.setTextInteractionFlags(Qt.TextInteractionFlag.LinksAccessibleByMouse)
        self.view.viewport().installEventFilter(self)
        root.addWidget(self.view, 1)

        footer = QHBoxLayout()
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(5)
        self.ask_edit = QLineEdit(card)
        self.ask_edit.setObjectName("ask")
        self.ask_edit.setPlaceholderText("Уточнить задание… (Enter — отправить)")
        self.ask_edit.setClearButtonEnabled(True)
        self.ask_edit.returnPressed.connect(self._submit_question)
        self.btn_send = QPushButton("➤", card)
        self.btn_send.setObjectName("send")
        self.btn_send.setToolTip("Отправить уточнение (Enter)")
        self.btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_send.clicked.connect(self._submit_question)
        footer.addWidget(self.ask_edit, 1)
        footer.addWidget(self.btn_send)
        grip = QSizeGrip(card)
        grip.setFixedSize(14, 14)
        footer.addWidget(grip, 0, Qt.AlignmentFlag.AlignBottom)
        root.addLayout(footer)

    def _tool_button(self, text: str, tip: str) -> QPushButton:
        btn = QPushButton(text, self)
        btn.setObjectName("tool")
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return btn

    # --- геометрия --------------------------------------------------------
    def _restore_geometry(self) -> None:
        size = self.cfg.get("widget_size", [420, 320])
        pos = self.cfg.get("widget_pos", [200, 200])
        try:
            w, h = int(size[0]), int(size[1])
            x, y = int(pos[0]), int(pos[1])
        except (TypeError, ValueError, IndexError):
            w, h, x, y = 420, 320, 200, 200
        self.resize(max(w, 260), max(h, 160))
        screen = QGuiApplication.screenAt(QPoint(x, y)) or QGuiApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            x = min(max(x, area.left()), max(area.right() - self.width(), area.left()))
            y = min(max(y, area.top()), max(area.bottom() - self.height(), area.top()))
        self.move(x, y)

    def _save_geometry(self) -> None:
        self.cfg.set("widget_pos", [self.x(), self.y()])
        self.cfg.set("widget_size", [self.width(), self.height()])
        self.cfg.save()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        if self.isVisible():
            self._geometry_save.start()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.isVisible():
            self._geometry_save.start()

    # --- перетаскивание ---------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_offset = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_offset = None
        event.accept()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802
        """Тянуть окно можно и за область текста."""
        if obj is self.view.viewport():
            if event.type() == QEvent.Type.MouseButtonPress:
                self.mousePressEvent(event)  # type: ignore[arg-type]
                return True
            if event.type() == QEvent.Type.MouseMove:
                self.mouseMoveEvent(event)  # type: ignore[arg-type]
                return True
            if event.type() == QEvent.Type.MouseButtonRelease:
                self.mouseReleaseEvent(event)  # type: ignore[arg-type]
                return True
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.hide_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    # --- прозрачность / поверх окон ---------------------------------------
    def set_opacity(self, value: float, *, save: bool = True) -> float:
        value = round(min(max(float(value), MIN_OPACITY), MAX_OPACITY), 2)
        self.setWindowOpacity(value)
        if save:
            self.cfg.set("opacity", value, save=True)
        return value

    def bump_opacity(self, delta: float) -> float:
        return self.set_opacity(self.windowOpacity() + delta)

    def set_always_on_top(self, on_top: bool, *, reshow: bool = True) -> None:
        flags = self.windowFlags()
        if on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        was_visible = self.isVisible()
        self.setWindowFlags(flags)
        # смена флагов скрывает окно на Windows — возвращаем обратно
        if reshow and was_visible:
            self.show()

    # --- содержимое -------------------------------------------------------
    def set_model_label(self, model: str) -> None:
        self.title.setText(model)

    def set_hide_hotkey(self, combo: str) -> None:
        """Показать актуальную комбинацию сворачивания в подсказке ✕."""
        pretty = "+".join(part.capitalize() for part in combo.split("+")) if combo else ""
        tip = "Свернуть в трей (Esc"
        tip += f", {pretty})" if pretty else ")"
        self.btn_close.setToolTip(tip)

    def show_idle(self) -> None:
        self._spinner.stop()
        self.status.setText("")
        self._transcript.clear()
        self.view.setMarkdown(
            "**Жду скриншот.**\n\n"
            "Сделайте снимок `Win+Shift+S` — ответ появится здесь, "
            "а уточнения можно дописать в строке снизу.\n\n"
            "- `Ctrl+Shift+H` — скрыть/показать\n"
            "- `Ctrl+Shift+S` — настройки\n"
            "- `Ctrl+Shift+P` — пауза слежения\n"
            "- `Ctrl+Alt+↑/↓` — прозрачность"
        )

    def reset_transcript(self) -> None:
        """Новый скриншот — начинаем диалог с чистого листа."""
        self._transcript.clear()
        self._answer_text = ""

    def add_user_message(self, text: str) -> None:
        self._transcript.append(("user", text))
        self._render()

    def show_thinking(self, *, reset: bool = False) -> None:
        if reset:
            self.reset_transcript()
        self._spinner_index = 0
        self.status.setText("Думаю…")
        self.set_busy(True)
        self._render(pending=True)
        self.pop_up()

    def show_answer(self, text: str) -> None:
        self._spinner.stop()
        self._answer_text = text
        self.status.setText("")
        self._transcript.append(("assistant", text))
        self.set_busy(False)
        self._render()
        self.pop_up()

    def show_error(self, text: str) -> None:
        self._spinner.stop()
        self._answer_text = text
        self.status.setText("Ошибка")
        self._transcript.append(("error", text))
        self.set_busy(False)
        self._render()
        self.pop_up()

    # --- рендер диалога ---------------------------------------------------
    def _render(self, *, pending: bool = False) -> None:
        parts: list[str] = []
        for role, text in self._transcript:
            if role == "user":
                parts.append(f"**Вы:** {text}")
            elif role == "error":
                parts.append(f"**⚠ Ошибка**\n\n{text}")
            else:
                parts.append(text)
        if pending:
            parts.append("_Думаю…_")
            self._spinner.start()
        self.view.setMarkdown("\n\n---\n\n".join(parts) if parts else "")
        bar = self.view.verticalScrollBar()
        # первый ответ читают сверху, продолжение диалога — снизу
        bar.setValue(bar.maximum() if len(self._transcript) > 1 or pending else 0)

    def set_busy(self, busy: bool) -> None:
        """Пока запрос в полёте, строку ввода блокируем."""
        self.ask_edit.setEnabled(not busy)
        self.btn_send.setEnabled(not busy)
        if not busy and self.ask_edit.isVisible():
            self.ask_edit.setFocus(Qt.FocusReason.OtherFocusReason)

    def _submit_question(self) -> None:
        text = self.ask_edit.text().strip()
        if not text:
            return
        self.ask_edit.clear()
        self.question_asked.emit(text)

    def _tick_spinner(self) -> None:
        frame = SPINNER_FRAMES[self._spinner_index % len(SPINNER_FRAMES)]
        self._spinner_index += 1
        self.status.setText(f"{frame} Думаю…")

    def copy_answer(self) -> None:
        if not self._answer_text:
            self.status.setText("Нечего копировать")
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._answer_text)
        self.status.setText("Скопировано")
        QTimer.singleShot(1500, lambda: self.status.setText(""))

    def pop_up(self) -> None:
        """Показать окно, не перехватывая фокус ввода."""
        if not self.isVisible():
            self.show()
        self.raise_()

    def toggle_visibility(self) -> bool:
        if self.isVisible():
            self.hide()
        else:
            self.pop_up()
        return self.isVisible()
