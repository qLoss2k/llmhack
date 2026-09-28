"""Клиент vibecode.moe: два формата API в одном интерфейсе.

У провайдера claude-* модели доступны ТОЛЬКО через /v1/messages (формат Anthropic),
а gpt-*/gemini-*/grok-* — через /v1/chat/completions (формат OpenAI). Поэтому формат
выбирается по модели (api_format="auto"), либо задаётся вручную в настройках.

Наружу никогда не летят низкоуровневые исключения: только LLMError с текстом
для показа пользователю.
"""
from __future__ import annotations

import base64
import json
import logging
from typing import Any

import requests

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 90
DEFAULT_MAX_TOKENS = 1024

FORMAT_AUTO = "auto"
FORMAT_ANTHROPIC = "anthropic"
FORMAT_OPENAI = "openai"
FORMATS = (FORMAT_AUTO, FORMAT_ANTHROPIC, FORMAT_OPENAI)

# подсказка провайдера, по которой в режиме auto делаем повтор в другом формате
_WRONG_ENDPOINT_HINTS = (
    "only available via /v1/messages",
    "anthropic format",
    "only available via /v1/chat/completions",
)


class LLMError(Exception):
    """Ошибка с понятным пользователю текстом."""


class WrongEndpointError(LLMError):
    """Модель живёт на другом эндпоинте — можно повторить в другом формате."""


def detect_format(model: str) -> str:
    """claude-* → Anthropic /messages, остальное → OpenAI /chat/completions."""
    return FORMAT_ANTHROPIC if "claude" in (model or "").lower() else FORMAT_OPENAI


def _extract_detail(body: str) -> str:
    try:
        data = json.loads(body)
    except (ValueError, TypeError):
        return (body or "")[:300]
    if isinstance(data, dict):
        err = data.get("error")
        if isinstance(err, dict):
            return str(err.get("message") or err)[:300]
        if isinstance(err, str):
            return err[:300]
        for key in ("message", "detail"):
            if key in data:
                return str(data[key])[:300]
    return (body or "")[:300]


def _status_message(status: int, body: str) -> str:
    detail = _extract_detail(body)
    if status in (401, 403):
        return "Неверный токен (401). Проверьте API-ключ в настройках."
    if status in (402, 429):
        return f"Лимит / нет средств ({status}). {detail}".strip()
    if status == 404:
        return f"Модель или эндпоинт не найдены (404). Проверьте Base URL и модель. {detail}".strip()
    if status == 400:
        return f"Запрос отклонён (400). {detail}".strip()
    if 500 <= status < 600:
        return f"Ошибка на стороне провайдера ({status}). {detail}".strip()
    return f"HTTP {status}. {detail}".strip()


def image_to_data_url(png_bytes: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(png_bytes).decode("ascii")


# --- реплики диалога ------------------------------------------------------
# Нейтральный формат: {"role": "user"|"assistant", "text": str, "png": bytes|None}.
# В тело запроса переводится под нужный протокол уже в _anthropic_body/_openai_body.

def user_turn(text: str, png: bytes | None = None) -> dict[str, Any]:
    return {"role": "user", "text": text or "", "png": png}


def assistant_turn(text: str) -> dict[str, Any]:
    return {"role": "assistant", "text": text or "", "png": None}


def _blocks_to_text(content: Any) -> str:
    """content бывает строкой или списком блоков — приводим к тексту."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") in (None, "text", "output_text"):
                    parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return "" if content is None else str(content)


def _parse_openai(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError("Пустой ответ от API (нет choices).")
    message = choices[0].get("message") or {}
    text = _blocks_to_text(message.get("content")).strip()
    if not text:
        raise LLMError("Модель вернула пустой ответ.")
    return text


def _parse_anthropic(payload: dict[str, Any]) -> str:
    text = _blocks_to_text(payload.get("content")).strip()
    if not text:
        raise LLMError("Модель вернула пустой ответ.")
    if payload.get("stop_reason") == "max_tokens":
        text += "\n\n_…ответ обрезан по лимиту max_tokens (увеличьте в настройках)._"
    return text


class LLMClient:
    """Один запрос — один вызов. Формат выбирается по модели или вручную."""

    def __init__(
        self,
        base_url: str,
        api_token: str,
        model: str,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_format: str = FORMAT_AUTO,
        use_sdk: bool = False,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_token = (api_token or "").strip()
        self.model = (model or "").strip()
        self.timeout = timeout
        self.max_tokens = max(int(max_tokens or DEFAULT_MAX_TOKENS), 1)
        self.api_format = api_format if api_format in FORMATS else FORMAT_AUTO
        self.use_sdk = use_sdk

    # --- публичное API ----------------------------------------------------
    def ask(self, turns: list[dict[str, Any]]) -> str:
        """Основной вход: вся история диалога целиком (см. user_turn/assistant_turn)."""
        if not turns:
            raise LLMError("Пустой диалог: нечего отправлять.")
        return self._run(turns)

    def ask_image(self, png_bytes: bytes, prompt: str) -> str:
        return self._run([user_turn(prompt or "", png_bytes)])

    def ask_text(self, prompt: str) -> str:
        return self._run([user_turn(prompt)])

    def test_connection(self) -> str:
        return self.ask_text("Ответь одним словом: ok")

    # --- выбор формата ----------------------------------------------------
    def _run(self, turns: list[dict[str, Any]]) -> str:
        self._validate()
        if self.api_format == FORMAT_AUTO:
            primary = detect_format(self.model)
            fallback = FORMAT_OPENAI if primary == FORMAT_ANTHROPIC else FORMAT_ANTHROPIC
            try:
                return self._send(primary, turns)
            except WrongEndpointError as exc:
                log.info("Формат %s не подошёл (%s), пробую %s", primary, exc, fallback)
                return self._send(fallback, turns)
        return self._send(self.api_format, turns)

    def _validate(self) -> None:
        if not self.api_token:
            raise LLMError("Не задан API-токен. Откройте настройки (⚙) и вставьте ключ vk-...")
        if not self.base_url:
            raise LLMError("Не задан Base URL.")
        if not self.model:
            raise LLMError("Не задана модель.")

    def _send(self, fmt: str, turns: list[dict[str, Any]]) -> str:
        if fmt == FORMAT_ANTHROPIC:
            url = f"{self.base_url}/messages"
            body = self._anthropic_body(turns)
            parser = _parse_anthropic
        else:
            url = f"{self.base_url}/chat/completions"
            body = self._openai_body(turns)
            parser = _parse_openai
        if self.use_sdk and fmt == FORMAT_OPENAI:
            return parser(self._post_sdk(body))
        return parser(self._post(url, body))

    # --- тела запросов ----------------------------------------------------
    def _anthropic_body(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        messages = []
        for turn in turns:
            content: list[dict[str, Any]] = [{"type": "text", "text": turn.get("text", "")}]
            png = turn.get("png")
            if png:
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(png).decode("ascii"),
                    },
                })
            messages.append({"role": turn.get("role", "user"), "content": content})
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

    def _openai_body(self, turns: list[dict[str, Any]]) -> dict[str, Any]:
        messages = []
        for turn in turns:
            png = turn.get("png")
            text = turn.get("text", "")
            if png:
                content: Any = [
                    {"type": "text", "text": text},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(png)}},
                ]
            else:
                content = text
            messages.append({"role": turn.get("role", "user"), "content": content})
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }

    # --- транспорт --------------------------------------------------------
    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        log.info("POST %s model=%s", url, body.get("model"))
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=self.timeout)
        except requests.Timeout as exc:
            raise LLMError("Нет соединения: превышено время ожидания ответа.") from exc
        except requests.ConnectionError as exc:
            raise LLMError("Нет соединения с API. Проверьте интернет и Base URL.") from exc
        except requests.RequestException as exc:
            raise LLMError(f"Сбой запроса: {exc}") from exc

        if resp.status_code >= 400:
            log.warning("API %s: %s", resp.status_code, resp.text[:500])
            detail = _extract_detail(resp.text).lower()
            if resp.status_code in (400, 404) and any(h in detail for h in _WRONG_ENDPOINT_HINTS):
                raise WrongEndpointError(_status_message(resp.status_code, resp.text))
            raise LLMError(_status_message(resp.status_code, resp.text))
        try:
            payload = resp.json()
        except ValueError as exc:
            raise LLMError("API вернул не-JSON ответ.") from exc
        if not isinstance(payload, dict):
            raise LLMError("API вернул неожиданную структуру ответа.")
        return payload

    def _post_sdk(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - зависит от окружения
            raise LLMError("Не установлен пакет openai (pip install openai).") from exc
        try:
            client = OpenAI(
                api_key=self.api_token,
                base_url=self.base_url,
                timeout=float(self.timeout),
                max_retries=0,
            )
            completion = client.chat.completions.create(**body)
        except Exception as exc:  # SDK оборачивает всё в свои типы
            status = getattr(exc, "status_code", None)
            if isinstance(status, int):
                raise LLMError(_status_message(status, str(exc))) from exc
            raise LLMError(f"Сбой запроса: {exc}") from exc
        return completion.model_dump() if hasattr(completion, "model_dump") else dict(completion)

    # --- справочное -------------------------------------------------------
    def list_models(self) -> list[str]:
        """Список id моделей для выпадашки в настройках."""
        self._validate()
        url = f"{self.base_url}/models"
        try:
            resp = requests.get(
                url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                timeout=min(self.timeout, 30),
            )
        except requests.RequestException as exc:
            raise LLMError(f"Не удалось получить список моделей: {exc}") from exc
        if resp.status_code >= 400:
            raise LLMError(_status_message(resp.status_code, resp.text))
        try:
            data = resp.json().get("data", [])
        except ValueError as exc:
            raise LLMError("Список моделей пришёл не в JSON.") from exc
        return sorted(str(m.get("id", "")) for m in data if isinstance(m, dict) and m.get("id"))


def client_from_config(cfg: Any) -> LLMClient:
    return LLMClient(
        base_url=cfg.get("base_url"),
        api_token=cfg.get("api_token"),
        model=cfg.get("model"),
        max_tokens=int(cfg.get("max_tokens", DEFAULT_MAX_TOKENS)),
        api_format=str(cfg.get("api_format", FORMAT_AUTO)),
    )
