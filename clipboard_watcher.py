"""Слежение за буфером обмена: реагируем только на изображения."""
from __future__ import annotations

import hashlib
import logging

from PyQt6.QtCore import QBuffer, QByteArray, QIODevice, QObject, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication, QImage

log = logging.getLogger(__name__)

DEBOUNCE_MS = 500
POLL_MS = 2000


def qimage_to_png(image: QImage) -> bytes:
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buf, "PNG")
    data = bytes(buf.data())
    buf.close()
    return data


class ClipboardWatcher(QObject):
    """Эмитит image_captured(png_bytes) при появлении новой картинки в буфере."""

    image_captured = pyqtSignal(bytes)

    def __init__(self, enabled: bool = True, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._enabled = enabled
        self._last_hash: str | None = None
        self._last_signature: tuple[int, int, int] | None = None

        self._clipboard = QGuiApplication.clipboard()
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._read_clipboard)

        # Основной триггер — сигнал буфера; опрос нужен как страховка,
        # т.к. на Windows dataChanged иногда теряется.
        self._poll = QTimer(self)
        self._poll.setInterval(POLL_MS)
        self._poll.timeout.connect(self._poll_tick)

        if self._clipboard is not None:
            self._clipboard.dataChanged.connect(self._on_data_changed)
        if enabled:
            self._poll.start()

    # --- управление -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        if enabled == self._enabled:
            return
        self._enabled = enabled
        if enabled:
            # текущее содержимое считаем «уже виденным», чтобы не выстрелить сразу
            self._remember_current()
            self._poll.start()
        else:
            self._poll.stop()
            self._debounce.stop()
        log.info("Слежение за буфером: %s", "вкл" if enabled else "выкл")

    def toggle(self) -> bool:
        self.set_enabled(not self._enabled)
        return self._enabled

    # --- внутреннее -------------------------------------------------------
    def _current_image(self) -> QImage | None:
        if self._clipboard is None:
            return None
        md = self._clipboard.mimeData()
        if md is None or not md.hasImage():
            return None
        image = self._clipboard.image()
        if image.isNull() or image.width() == 0 or image.height() == 0:
            return None
        return image

    def _signature(self, image: QImage) -> tuple[int, int, int]:
        return (image.width(), image.height(), image.sizeInBytes())

    def _remember_current(self) -> None:
        image = self._current_image()
        if image is None:
            self._last_signature = None
            return
        self._last_signature = self._signature(image)
        self._last_hash = hashlib.sha1(qimage_to_png(image)).hexdigest()

    def _on_data_changed(self) -> None:
        if not self._enabled:
            return
        self._debounce.start()

    def _poll_tick(self) -> None:
        if not self._enabled or self._debounce.isActive():
            return
        image = self._current_image()
        if image is None:
            return
        # дешёвая проверка, чтобы не кодировать PNG каждые 2 секунды
        if self._signature(image) == self._last_signature:
            return
        self._debounce.start()

    def _read_clipboard(self) -> None:
        if not self._enabled:
            return
        image = self._current_image()
        if image is None:
            return
        png = qimage_to_png(image)
        if not png:
            log.warning("Не удалось закодировать изображение из буфера в PNG")
            return
        digest = hashlib.sha1(png).hexdigest()
        if digest == self._last_hash:
            return
        self._last_hash = digest
        self._last_signature = self._signature(image)
        log.info("Новый скриншот из буфера: %dx%d, %d КБ",
                 image.width(), image.height(), len(png) // 1024)
        self.image_captured.emit(QByteArray(png).data())
