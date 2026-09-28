"""Проверочный прогон без реального API: UI, буфер, разбор ответов."""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="backslashreplace")

from PyQt6.QtCore import Qt, QTimer  # noqa: E402
from PyQt6.QtGui import QColor, QImage, QKeyEvent  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

import llm_client  # noqa: E402
from clipboard_watcher import ClipboardWatcher, qimage_to_png  # noqa: E402
from config import Config  # noqa: E402
from llm_client import (  # noqa: E402
    LLMClient,
    LLMError,
    _status_message,
    assistant_turn,
    detect_format,
    user_turn,
)
from settings_dialog import SettingsDialog  # noqa: E402
from tray import Tray  # noqa: E402
from widget import ChatWidget  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, extra: str = "") -> None:
    print(("PASS " if cond else "FAIL ") + name + (f" :: {extra}" if extra else ""))
    if not cond:
        failures.append(name)


class FakeResponse:
    def __init__(self, status: int, payload):
        self.status_code = status
        self._payload = payload
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


ANTHROPIC_OK = {
    "content": [{"type": "text", "text": "Это кот."}],
    "stop_reason": "end_turn",
    "model": "claude-sonnet-5",
}
WRONG_ENDPOINT = {
    "error": {
        "message": "this model is only available via /v1/messages (Anthropic format)",
        "type": "invalid_request_error",
    }
}


def test_format_detection() -> None:
    check("claude → anthropic", detect_format("claude-sonnet-5") == "anthropic")
    check("claude-opus → anthropic", detect_format("claude-opus-5") == "anthropic")
    check("gpt → openai", detect_format("gpt-6-sol") == "openai")
    check("gemini → openai", detect_format("gemini-3.1-pro-preview") == "openai")


def test_anthropic_request() -> None:
    """claude-sonnet-5 должен ехать на /messages в формате Anthropic."""
    sent = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        sent["url"] = url
        sent["headers"] = headers
        sent["body"] = json
        return FakeResponse(200, ANTHROPIC_OK)

    original = llm_client.requests.post
    llm_client.requests.post = fake_post
    try:
        client = LLMClient("https://vibecode.moe/v1", "vk-test", "claude-sonnet-5")
        answer = client.ask_image(b"\x89PNG-fake", "Опиши скриншот")
        check("ask_image возвращает текст", answer == "Это кот.", answer)
        check("эндпоинт /messages", sent["url"] == "https://vibecode.moe/v1/messages", sent["url"])
        check("Bearer-заголовок", sent["headers"]["Authorization"] == "Bearer vk-test")
        check("модель в теле", sent["body"]["model"] == "claude-sonnet-5")
        check("max_tokens обязателен", sent["body"]["max_tokens"] == 1024, str(sent["body"].get("max_tokens")))
        content = sent["body"]["messages"][0]["content"]
        check("текстовый блок", content[0]["text"] == "Опиши скриншот")
        source = content[1]["source"]
        check("блок image/base64", content[1]["type"] == "image" and source["type"] == "base64")
        check("media_type png", source["media_type"] == "image/png")
        check("картинка закодирована", base64.b64decode(source["data"]) == b"\x89PNG-fake")

        # обрезанный по лимиту ответ помечается
        llm_client.requests.post = lambda *a, **k: FakeResponse(
            200, {"content": [{"type": "text", "text": "длинный"}], "stop_reason": "max_tokens"}
        )
        check("пометка про max_tokens", "обрезан" in client.ask_text("hi"))
    finally:
        llm_client.requests.post = original


def test_openai_request_and_fallback() -> None:
    original = llm_client.requests.post
    try:
        sent = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            sent["url"] = url
            sent["body"] = json
            return FakeResponse(200, {"choices": [{"message": {"content": "gpt сказал"}}]})

        llm_client.requests.post = fake_post
        gpt = LLMClient("https://vibecode.moe/v1", "vk-test", "gpt-6-sol")
        check("gpt-модель отвечает", gpt.ask_image(b"png", "опиши") == "gpt сказал")
        check("эндпоинт /chat/completions",
              sent["url"] == "https://vibecode.moe/v1/chat/completions", sent["url"])
        check("data-url картинки",
              sent["body"]["messages"][0]["content"][1]["image_url"]["url"].startswith(
                  "data:image/png;base64,"))

        # auto: получили «не тот эндпоинт» → повтор в другом формате
        calls = []

        def flipping_post(url, headers=None, json=None, timeout=None):
            calls.append(url)
            if url.endswith("/chat/completions"):
                return FakeResponse(400, WRONG_ENDPOINT)
            return FakeResponse(200, ANTHROPIC_OK)

        llm_client.requests.post = flipping_post
        # модель без «claude» в имени, но живущая на /messages
        odd = LLMClient("https://vibecode.moe/v1", "vk-test", "some-proxy-model")
        check("auto делает fallback на /messages", odd.ask_text("hi") == "Это кот.", str(calls))
        check("fallback = ровно 2 запроса", len(calls) == 2, str(calls))

        # ручной формат fallback не делает
        llm_client.requests.post = flipping_post
        forced = LLMClient("https://vibecode.moe/v1", "vk-test", "x", api_format="openai")
        try:
            forced.ask_text("hi")
            check("ручной формат не фоллбечит", False)
        except LLMError as exc:
            check("ручной формат не фоллбечит", "400" in str(exc), str(exc))

        # content списком блоков у openai-формата
        llm_client.requests.post = lambda *a, **k: FakeResponse(
            200, {"choices": [{"message": {"content": [{"type": "text", "text": "блочный"}]}}]}
        )
        check("content списком блоков", gpt.ask_text("hi") == "блочный")
    finally:
        llm_client.requests.post = original


def test_followup_turns() -> None:
    """Уточнение должно уезжать вместе со скриншотом и прошлым ответом."""
    original = llm_client.requests.post
    sent = {}
    try:
        def fake_post(url, headers=None, json=None, timeout=None):
            sent["body"] = json
            return FakeResponse(200, ANTHROPIC_OK)

        llm_client.requests.post = fake_post
        client = LLMClient("https://vibecode.moe/v1", "vk-test", "claude-sonnet-5")
        turns = [
            user_turn("Опиши скриншот", b"\x89PNG-fake"),
            assistant_turn("Это кот."),
            user_turn("А какой породы?"),
        ]
        check("ask(turns) отвечает", client.ask(turns) == "Это кот.")
        messages = sent["body"]["messages"]
        check("три реплики в теле", len(messages) == 3, str(len(messages)))
        check("роли по порядку",
              [m["role"] for m in messages] == ["user", "assistant", "user"],
              str([m["role"] for m in messages]))
        check("скриншот только в первой реплике",
              any(b.get("type") == "image" for b in messages[0]["content"])
              and all(b.get("type") != "image" for b in messages[2]["content"]))
        check("текст уточнения доехал",
              messages[2]["content"][0]["text"] == "А какой породы?",
              messages[2]["content"][0]["text"])
        check("прошлый ответ модели в истории",
              messages[1]["content"][0]["text"] == "Это кот.")

        # тот же диалог в openai-формате
        llm_client.requests.post = lambda *a, **k: FakeResponse(
            200, {"choices": [{"message": {"content": "ок"}}]})
        gpt = LLMClient("https://vibecode.moe/v1", "vk-test", "gpt-6-sol")
        sent.clear()

        def fake_post2(url, headers=None, json=None, timeout=None):
            sent["body"] = json
            return FakeResponse(200, {"choices": [{"message": {"content": "ок"}}]})

        llm_client.requests.post = fake_post2
        check("openai-формат тоже держит диалог", gpt.ask(turns) == "ок")
        messages = sent["body"]["messages"]
        check("openai: 3 реплики", len(messages) == 3, str(len(messages)))
        check("openai: ответ модели строкой", messages[1]["content"] == "Это кот.",
              str(messages[1]["content"])[:40])
        check("openai: уточнение строкой", messages[2]["content"] == "А какой породы?")

        try:
            client.ask([])
            check("пустой диалог ловится", False)
        except LLMError as exc:
            check("пустой диалог ловится", "Пустой диалог" in str(exc), str(exc))
    finally:
        llm_client.requests.post = original


def test_llm_errors() -> None:
    original = llm_client.requests.post
    try:
        client = LLMClient("https://vibecode.moe/v1", "vk-test", "claude-sonnet-5",
                           api_format="anthropic")
        for status, needle in ((401, "Неверный токен"), (429, "Лимит"),
                               (404, "не найдены"), (500, "провайдера")):
            llm_client.requests.post = lambda *a, s=status, **k: FakeResponse(
                s, {"error": {"message": "boom"}}
            )
            try:
                client.ask_text("hi")
                check(f"ошибка {status} поднимает LLMError", False)
            except LLMError as exc:
                check(f"ошибка {status}: «{needle}»", needle in str(exc), str(exc))

        def raise_timeout(*a, **k):
            raise llm_client.requests.Timeout()

        llm_client.requests.post = raise_timeout
        try:
            client.ask_text("hi")
            check("таймаут поднимает LLMError", False)
        except LLMError as exc:
            check("таймаут → «Нет соединения»", "Нет соединения" in str(exc), str(exc))

        empty = LLMClient("https://vibecode.moe/v1", "", "claude-sonnet-5")
        try:
            empty.ask_text("hi")
            check("пустой токен ловится до запроса", False)
        except LLMError as exc:
            check("пустой токен ловится до запроса", "токен" in str(exc), str(exc))
    finally:
        llm_client.requests.post = original

    check("маппинг 402", "Лимит" in _status_message(402, "{}"))


def test_config(tmpdir: str) -> Config:
    path = os.path.join(tmpdir, "config.json")
    cfg = Config(path)
    check("первый запуск определён", cfg.first_run)
    check("config.json создан", os.path.exists(path))
    check("дефолтная модель", cfg.get("model") == "claude-sonnet-5", cfg.get("model"))
    check("дефолтная прозрачность", cfg.get("opacity") == 0.85)
    check("формат API по умолчанию auto", cfg.get("api_format") == "auto", str(cfg.get("api_format")))
    check("max_tokens по умолчанию", cfg.get("max_tokens") == 1024, str(cfg.get("max_tokens")))
    check("хоткей прозрачности", cfg.hotkey("opacity_up") == "ctrl+alt+up")
    check("хоткей сворачивания по умолчанию", cfg.hotkey("hide_to_tray") == "ctrl+shift+x",
          cfg.hotkey("hide_to_tray"))

    cfg.set_hotkey("hide_to_tray", "  Ctrl+Alt+M  ", save=True)
    check("set_hotkey нормализует регистр и пробелы",
          Config(path).hotkey("hide_to_tray") == "ctrl+alt+m",
          Config(path).hotkey("hide_to_tray"))
    cfg.set_hotkey("hide_to_tray", "", save=True)
    check("хоткей можно отключить пустой строкой",
          Config(path).hotkey("hide_to_tray") == "")
    check("прочие хоткеи не пострадали",
          Config(path).hotkey("quit") == "ctrl+shift+q")
    cfg.set_hotkey("hide_to_tray", "ctrl+shift+x", save=True)

    # частичный блок hotkeys в файле дополняется значениями по умолчанию
    partial = os.path.join(tmpdir, "partial.json")
    with open(partial, "w", encoding="utf-8") as fh:
        json.dump({"hotkeys": {"quit": "ctrl+alt+q"}}, fh)
    merged = Config(partial)
    check("старый конфиг без hide_to_tray получает дефолт",
          merged.hotkey("hide_to_tray") == "ctrl+shift+x" and merged.hotkey("quit") == "ctrl+alt+q")

    cfg.update({"api_token": "vk-abc", "opacity": 0.5})
    reread = Config(path)
    check("настройки переживают перезапуск", reread.get("api_token") == "vk-abc" and reread.get("opacity") == 0.5)
    check("повторный запуск не первый", not reread.first_run)
    return reread


def test_ui(cfg: Config) -> None:
    widget = ChatWidget(cfg)
    widget.show()
    QApplication.processEvents()
    check("виджет виден", widget.isVisible())
    check("окно без рамки", bool(widget.windowFlags() & 0x800))
    check("прозрачность из конфига", abs(widget.windowOpacity() - 0.5) < 0.01, str(widget.windowOpacity()))

    widget.bump_opacity(0.05)
    check("хоткей +5 %", abs(widget.windowOpacity() - 0.55) < 0.01, str(widget.windowOpacity()))
    for _ in range(30):
        widget.bump_opacity(-0.05)
    check("нижний предел 10 %", abs(widget.windowOpacity() - 0.10) < 0.01, str(widget.windowOpacity()))
    widget.set_opacity(0.85)
    check("прозрачность сохранилась в конфиг", cfg.get("opacity") == 0.85, str(cfg.get("opacity")))

    widget.show_thinking()
    QApplication.processEvents()
    check("спиннер запущен", "Думаю" in widget.status.text(), widget.status.text())

    widget.show_answer("**Жирный** и `код`\n\n- пункт 1\n- пункт 2")
    html = widget.view.toHtml()
    check("markdown → html (жирный)", "font-weight" in html)
    check("markdown → html (список)", "<li" in html)
    widget.copy_answer()
    clip = QApplication.clipboard().text()
    check("кнопка 📋 копирует ответ", "Жирный" in clip, clip[:30])

    widget.show_error("Неверный токен (401).")
    check("ошибка в виджете", "401" in widget.view.toPlainText(), widget.view.toPlainText()[:60])

    # --- строка уточнений ---
    asked: list[str] = []
    widget.question_asked.connect(asked.append)
    widget.show_thinking(reset=True)
    check("новый скриншот чистит диалог", widget.view.toPlainText().strip() == "Думаю…",
          widget.view.toPlainText()[:40])
    check("во время запроса ввод заблокирован",
          not widget.ask_edit.isEnabled() and not widget.btn_send.isEnabled())

    widget.show_answer("На скриншоте кот.")
    check("после ответа ввод разблокирован",
          widget.ask_edit.isEnabled() and widget.btn_send.isEnabled())

    widget.ask_edit.setText("   ")
    widget._submit_question()
    check("пустое уточнение не отправляется", not asked)

    widget.ask_edit.setText("А какой породы?")
    widget._submit_question()
    check("Enter отправляет уточнение", asked == ["А какой породы?"], str(asked))
    check("поле ввода очищено", widget.ask_edit.text() == "")

    widget.add_user_message("А какой породы?")
    widget.show_thinking()
    text = widget.view.toPlainText()
    check("вопрос виден в диалоге", "Вы: А какой породы?" in text, text[:80])
    check("прошлый ответ остался в диалоге", "кот" in text, text[:80])
    widget.show_answer("Рыжий, похож на мейн-куна.")
    check("оба ответа в диалоге",
          "кот" in widget.view.toPlainText() and "мейн-куна" in widget.view.toPlainText())
    check("реплики разделены линией", "<hr" in widget.view.toHtml())

    widget.move(340, 260)
    widget.resize(500, 380)
    widget._save_geometry()
    check("позиция сохранена", cfg.get("widget_pos") == [340, 260], str(cfg.get("widget_pos")))
    check("размер сохранён", cfg.get("widget_size") == [500, 380], str(cfg.get("widget_size")))

    widget.set_always_on_top(False)
    check("поверх окон выключается", not (widget.windowFlags() & 0x40000))
    widget.set_always_on_top(True)
    check("поверх окон включается", bool(widget.windowFlags() & 0x40000))

    visible = widget.toggle_visibility()
    check("toggle скрывает", not visible)
    check("toggle показывает", widget.toggle_visibility())

    hidden: list[bool] = []
    widget.hide_requested.connect(lambda: hidden.append(True))
    widget.keyPressEvent(QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Escape,
                                   Qt.KeyboardModifier.NoModifier))
    check("Esc просит свернуть в трей", hidden == [True])
    widget.btn_close.click()
    check("✕ просит свернуть в трей", hidden == [True, True])

    widget.set_hide_hotkey("ctrl+alt+m")
    check("подсказка ✕ показывает хоткей",
          "Ctrl+Alt+M" in widget.btn_close.toolTip() and "Esc" in widget.btn_close.toolTip(),
          widget.btn_close.toolTip())
    widget.set_hide_hotkey("")
    check("без хоткея подсказка только про Esc",
          widget.btn_close.toolTip() == "Свернуть в трей (Esc)", widget.btn_close.toolTip())
    widget.close()


def test_settings_dialog(cfg: Config) -> None:
    previews: list[float] = []
    dialog = SettingsDialog(cfg)
    dialog.opacity_preview.connect(previews.append)
    check("токен скрыт звёздочками", dialog.token_edit.echoMode().value == 2)
    dialog.eye_btn.setChecked(True)
    check("глаз показывает токен", dialog.token_edit.echoMode().value == 0)
    check("токен подставлен", dialog.token_edit.text() == "vk-abc", dialog.token_edit.text())
    check("base url подставлен", dialog.base_url_edit.text() == "https://vibecode.moe/v1")
    check("слайдер = 85", dialog.opacity_slider.value() == 85, str(dialog.opacity_slider.value()))

    dialog.opacity_slider.setValue(40)
    check("слайдер шлёт preview на лету", previews and abs(previews[-1] - 0.40) < 0.01, str(previews))
    check("метка процентов", dialog.opacity_value.text() == "40 %", dialog.opacity_value.text())

    check("модель подставлена в выпадашку",
          dialog.model_combo.currentText() == "claude-sonnet-5", dialog.model_combo.currentText())
    check("формат auto выбран", dialog.format_combo.currentData() == "auto")
    check("auto показывает /messages для claude",
          "/messages" in dialog.format_combo.itemText(0), dialog.format_combo.itemText(0))
    dialog.model_combo.setCurrentText("gpt-6-sol")
    check("auto показывает /chat/completions для gpt",
          "/chat/completions" in dialog.format_combo.itemText(0), dialog.format_combo.itemText(0))

    # --- хоткей сворачивания в трей ---
    check("текущий хоткей показан по-человечески",
          dialog.hide_hotkey_edit.text() == "Ctrl+Shift+X", dialog.hide_hotkey_edit.text())
    check("поле только для чтения", dialog.hide_hotkey_edit.isReadOnly())

    def press(key, mods=Qt.KeyboardModifier.NoModifier):
        dialog.hide_hotkey_edit.keyPressEvent(
            QKeyEvent(QKeyEvent.Type.KeyPress, key, mods))

    press(Qt.Key.Key_M, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)
    check("комбинация записывается с клавиатуры",
          dialog.hide_hotkey_edit.combo() == "ctrl+alt+m", dialog.hide_hotkey_edit.combo())
    check("в поле красивый вид", dialog.hide_hotkey_edit.text() == "Ctrl+Alt+M",
          dialog.hide_hotkey_edit.text())

    press(Qt.Key.Key_Z)
    check("без модификатора комбинация не принимается",
          dialog.hide_hotkey_edit.combo() == "ctrl+alt+m" and "нужен" in dialog.hide_hotkey_edit.text(),
          dialog.hide_hotkey_edit.text())

    press(Qt.Key.Key_Control, Qt.KeyboardModifier.ControlModifier)
    check("один модификатор комбинацией не считается",
          dialog.hide_hotkey_edit.combo() == "ctrl+alt+m", dialog.hide_hotkey_edit.combo())

    press(Qt.Key.Key_Down, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
    check("стрелки пишутся как понимает keyboard",
          dialog.hide_hotkey_edit.combo() == "ctrl+shift+down", dialog.hide_hotkey_edit.combo())

    press(Qt.Key.Key_Backspace)
    check("Backspace отключает хоткей", dialog.hide_hotkey_edit.combo() == "",
          dialog.hide_hotkey_edit.combo())
    dialog.hotkey_clear_btn.click()
    check("кнопка «Очистить» тоже обнуляет", dialog.hide_hotkey_edit.combo() == "")

    press(Qt.Key.Key_X, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
          | Qt.KeyboardModifier.MetaModifier)
    check("Win-модификатор поддержан",
          dialog.hide_hotkey_edit.combo() == "ctrl+shift+windows+x", dialog.hide_hotkey_edit.combo())
    press(Qt.Key.Key_T, Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.AltModifier)

    dialog.model_combo.setCurrentText("claude-sonnet-5")
    dialog.max_tokens_spin.setValue(2048)
    dialog.prompt_edit.setPlainText("Переведи текст на картинке.")
    dialog.token_edit.setText("vk-new")
    applied: list[bool] = []
    dialog.applied.connect(lambda: applied.append(True))
    dialog.accept()
    check("Сохранить пишет в конфиг", cfg.get("api_token") == "vk-new" and cfg.get("opacity") == 0.40)
    check("system_prompt сохранён", cfg.get("system_prompt") == "Переведи текст на картинке.")
    check("max_tokens сохранён", cfg.get("max_tokens") == 2048, str(cfg.get("max_tokens")))
    check("api_format сохранён", cfg.get("api_format") == "auto", str(cfg.get("api_format")))
    check("сигнал applied", applied == [True])
    check("новый хоткей сохранён в конфиг", cfg.hotkey("hide_to_tray") == "ctrl+alt+t",
          cfg.hotkey("hide_to_tray"))
    check("остальные хоткеи не затёрты", cfg.hotkey("toggle_widget") == "ctrl+shift+h"
          and cfg.hotkey("quit") == "ctrl+shift+q")

    dialog2 = SettingsDialog(cfg)
    check("сохранённый хоткей подставлен при открытии",
          dialog2.hide_hotkey_edit.combo() == "ctrl+alt+t", dialog2.hide_hotkey_edit.combo())
    dialog2._on_models_loaded(["claude-sonnet-5", "gpt-6-sol", "grok-4-7"])
    check("список моделей подтягивается", dialog2.model_combo.count() == 3,
          str(dialog2.model_combo.count()))
    check("выбор не сбился при обновлении списка",
          dialog2.model_combo.currentText() == "claude-sonnet-5", dialog2.model_combo.currentText())
    dialog2.sync_opacity(0.75)
    check("sync_opacity подтягивает хоткей", dialog2.opacity_slider.value() == 75)
    dialog2._show_test(False, "Неверный токен")
    check("тест соединения показывает ошибку", "✕" in dialog2.test_label.text())
    dialog2.close()


def test_clipboard(app: QApplication) -> None:
    got: list[bytes] = []
    watcher = ClipboardWatcher(True)
    watcher.image_captured.connect(got.append)

    QApplication.clipboard().setText("просто текст, игнорируем")
    QApplication.processEvents()
    _wait(app, 900)
    check("текст в буфере игнорируется", got == [], str(len(got)))

    image = QImage(64, 48, QImage.Format.Format_RGB32)
    image.fill(QColor(10, 120, 200))
    QApplication.clipboard().setImage(image)
    _wait(app, 1200)
    check("картинка из буфера перехвачена", len(got) == 1, str(len(got)))
    if got:
        check("это PNG", got[0][:4] == b"\x89PNG", repr(got[0][:4]))

    # тот же самый кадр не должен уехать повторно (debounce + хеш)
    QApplication.clipboard().setImage(image)
    _wait(app, 1200)
    check("дубликат не отправляется", len(got) == 1, str(len(got)))

    watcher.set_enabled(False)
    image2 = QImage(32, 32, QImage.Format.Format_RGB32)
    image2.fill(QColor(200, 30, 30))
    QApplication.clipboard().setImage(image2)
    _wait(app, 1200)
    check("на паузе ничего не шлём", len(got) == 1, str(len(got)))
    check("toggle возвращает слежение", watcher.toggle() is True)
    check("PNG-кодирование работает", qimage_to_png(image2)[:4] == b"\x89PNG")


def test_tray() -> None:
    tray = Tray()
    check("иконка трея не пустая", not tray.icon().isNull())
    tray.set_status("Отправляю…")
    check("тултип со статусом", "Отправляю" in tray.toolTip(), tray.toolTip())
    tray.set_watching(False)
    check("тултип про паузу", "паузе" in tray.toolTip(), tray.toolTip())
    check("пункт меню «Пауза» отмечен", tray.action_watch.isChecked())
    check("меню собрано", len(tray.contextMenu().actions()) == 6)
    tray.hide()


def _wait(app: QApplication, ms: int) -> None:
    done = {"v": False}
    QTimer.singleShot(ms, lambda: done.__setitem__("v", True))
    while not done["v"]:
        app.processEvents()


def main() -> int:
    app = QApplication(sys.argv)
    test_format_detection()
    test_anthropic_request()
    test_openai_request_and_fallback()
    test_followup_turns()
    test_llm_errors()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = test_config(tmpdir)
        test_ui(cfg)
        test_settings_dialog(cfg)
        test_clipboard(app)
        test_tray()
    print()
    if failures:
        print(f"ПРОВАЛЕНО {len(failures)}: " + ", ".join(failures))
        return 1
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
