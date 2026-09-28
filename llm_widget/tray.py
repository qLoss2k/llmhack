"""Иконка в системном трее с меню и статусом."""
from __future__ import annotations

import logging

from PyQt6.QtCore import QRect, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QMenu, QSystemTrayIcon

log = logging.getLogger(__name__)


def make_icon() -> QIcon:
    """Иконка рисуется в коде, чтобы не тащить внешний .ico в сборку."""
    pix = QPixmap(64, 64)
    pix.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pix)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setBrush(QBrush(QColor(38, 42, 54)))
    painter.setPen(QPen(QColor(120, 180, 255), 3))
    painter.drawRoundedRect(QRect(4, 4, 56, 56), 14, 14)
    painter.setPen(QPen(QColor(235, 240, 250)))
    font = QFont("Segoe UI", 22, QFont.Weight.Bold)
    painter.setFont(font)
    painter.drawText(QRect(4, 4, 56, 56), Qt.AlignmentFlag.AlignCenter, "AI")
    painter.end()
    return QIcon(pix)


class Tray(QSystemTrayIcon):
    toggle_widget_requested = pyqtSignal()
    settings_requested = pyqtSignal()
    toggle_watch_requested = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(make_icon(), parent)
        self._status = "Жду скриншот…"
        self._watching = True

        menu = QMenu()
        self.action_toggle = QAction("Показать / скрыть", menu)
        self.action_settings = QAction("Настройки…", menu)
        self.action_watch = QAction("Пауза слежения", menu)
        self.action_watch.setCheckable(True)
        self.action_quit = QAction("Выход", menu)

        self.action_toggle.triggered.connect(self.toggle_widget_requested.emit)
        self.action_settings.triggered.connect(self.settings_requested.emit)
        self.action_watch.triggered.connect(self.toggle_watch_requested.emit)
        self.action_quit.triggered.connect(self.quit_requested.emit)

        menu.addAction(self.action_toggle)
        menu.addAction(self.action_settings)
        menu.addSeparator()
        menu.addAction(self.action_watch)
        menu.addSeparator()
        menu.addAction(self.action_quit)
        self._menu = menu
        self.setContextMenu(menu)

        self.activated.connect(self._on_activated)
        self._refresh_tooltip()

    def _on_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.toggle_widget_requested.emit()

    # --- статус -----------------------------------------------------------
    def set_status(self, text: str) -> None:
        self._status = text
        self._refresh_tooltip()

    def set_watching(self, watching: bool) -> None:
        self._watching = watching
        # чекбокс «Пауза слежения» отмечен, когда слежение выключено
        self.action_watch.setChecked(not watching)
        self._refresh_tooltip()

    def _refresh_tooltip(self) -> None:
        suffix = "" if self._watching else "  (слежение на паузе)"
        self.setToolTip(f"LLM Widget — {self._status}{suffix}")
