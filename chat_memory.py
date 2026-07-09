"""
Lokaler Chat-Speicher.

Die Chats werden lokal als JSON gespeichert und koennen wieder in den Chat
geladen werden.
"""

from __future__ import annotations

import base64
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st


_MEMORY_DIR = Path(__file__).resolve().parent / ".chat_memory"
_MEMORY_FILE = _MEMORY_DIR / "chat_store.json"


def _load_store() -> list[dict[str, Any]]:
    if not _MEMORY_FILE.exists():
        return []
    try:
        with _MEMORY_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_store(items: list[dict[str, Any]]) -> None:
    _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with _MEMORY_FILE.open("w", encoding="utf-8") as file:
        json.dump(items, file, ensure_ascii=False, indent=2)


def _message_text(message: dict[str, Any]) -> str:
    role = str(message.get("role", ""))
    content = str(message.get("content", ""))
    diagram_note = " [Diagramm]" if message.get("figure_bytes") or message.get("figure_b64") else ""
    return f"{role}: {content}{diagram_note}".strip()


def _conversation_preview(messages: list[dict[str, Any]], max_chars: int = 900) -> str:
    text = "\n".join(_message_text(message) for message in messages if message.get("content"))
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[gekuerzt]"


def _encode_figure_bytes(value: Any) -> str | None:
    if isinstance(value, bytes):
        raw_bytes = value
    elif isinstance(value, bytearray):
        raw_bytes = bytes(value)
    else:
        return None
    return base64.b64encode(raw_bytes).decode("ascii")


def _decode_figure_bytes(value: Any) -> bytes | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception:
        return None


def _clean_message_for_storage(message: dict[str, Any]) -> dict[str, Any] | None:
    role = str(message.get("role", "")).strip()
    content = str(message.get("content", "")).strip()
    if not role or not content:
        return None

    clean_message: dict[str, Any] = {
        "role": role,
        "content": content,
    }

    figure_b64 = message.get("figure_b64")
    if not isinstance(figure_b64, str):
        figure_b64 = _encode_figure_bytes(message.get("figure_bytes"))
    if figure_b64:
        clean_message["figure_b64"] = figure_b64

    return clean_message


def _build_chat_record(
    messages: list[dict[str, Any]],
    source_name: str,
    title: str | None = None,
    chat_id: str | None = None,
    existing_chat: dict[str, Any] | None = None,
    autosaved: bool = False,
) -> dict[str, Any]:
    clean_messages = []
    for message in messages:
        clean_message = _clean_message_for_storage(message)
        if clean_message:
            clean_messages.append(clean_message)

    if not clean_messages:
        raise ValueError("Es gibt noch keine Chatnachrichten zum Speichern.")

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    item_id = chat_id or uuid.uuid4().hex
    auto_title = clean_messages[0]["content"][:70].strip() or "Gespeicherter Chat"
    existing_title = str((existing_chat or {}).get("title", "")).strip()
    title_text = title.strip() if title and title.strip() else existing_title or auto_title

    return {
        "id": item_id,
        "title": title_text,
        "source_name": source_name,
        "created_at": str((existing_chat or {}).get("created_at") or now),
        "updated_at": now,
        "autosaved": autosaved or bool((existing_chat or {}).get("autosaved")),
        "messages": clean_messages,
    }


def upsert_chat(
    messages: list[dict[str, Any]],
    source_name: str,
    title: str | None = None,
    chat_id: str | None = None,
    autosaved: bool = False,
) -> dict[str, Any]:
    items = _load_store()
    existing_chat = None
    if chat_id:
        for index, item in enumerate(items):
            if item.get("id") == chat_id:
                existing_chat = items.pop(index)
                break

    chat = _build_chat_record(
        messages=messages,
        source_name=source_name,
        title=title,
        chat_id=chat_id,
        existing_chat=existing_chat,
        autosaved=autosaved,
    )
    items.insert(0, chat)
    _save_store(items)
    return chat


def save_chat(messages: list[dict[str, Any]], source_name: str, title: str | None = None, chat_id: str | None = None) -> str:
    chat = upsert_chat(messages=messages, source_name=source_name, title=title, chat_id=chat_id)
    return str(chat["id"])


def _clean_loaded_messages(messages: Any) -> list[dict[str, Any]]:
    clean_messages = []
    if not isinstance(messages, list):
        return clean_messages

    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            clean_message: dict[str, Any] = {"role": role, "content": content}
            figure_bytes = _decode_figure_bytes(message.get("figure_b64"))
            if figure_bytes:
                clean_message["figure_bytes"] = figure_bytes
            clean_messages.append(clean_message)
    return clean_messages


def list_chats() -> list[dict[str, Any]]:
    return _load_store()


def load_chat(chat_id: str) -> dict[str, Any] | None:
    for item in _load_store():
        if item.get("id") == chat_id:
            return item
    return None


def _remember_opened_chat(chat: dict[str, Any]) -> None:
    st.session_state["opened_saved_chat_id"] = str(chat.get("id", ""))
    st.session_state["opened_saved_chat_title"] = str(chat.get("title", "Gespeicherter Chat"))
    st.session_state["opened_saved_chat_source_name"] = str(chat.get("source_name", ""))


def autosave_current_chat(messages: list[dict[str, Any]], source_name: str) -> dict[str, Any] | None:
    if not messages:
        return None

    try:
        chat = upsert_chat(
            messages=messages,
            source_name=source_name,
            chat_id=st.session_state.get("opened_saved_chat_id"),
            autosaved=True,
        )
        _remember_opened_chat(chat)
        st.session_state["chat_autosave_last_saved_at"] = str(chat.get("updated_at", ""))
        st.session_state.pop("chat_autosave_error", None)
        return chat
    except Exception as error:
        st.session_state["chat_autosave_error"] = str(error)
        return None


def delete_chat(chat_id: str) -> None:
    items = [item for item in _load_store() if item.get("id") != chat_id]
    _save_store(items)


def open_saved_chat(chat_id: str) -> None:
    chat = load_chat(chat_id)
    if not chat:
        raise ValueError("Der gespeicherte Chat wurde nicht gefunden.")

    messages = _clean_loaded_messages(chat.get("messages", []))
    if not messages:
        raise ValueError("Der gespeicherte Chat enthält keine lesbaren Nachrichten.")

    st.session_state["chat_messages"] = messages
    _remember_opened_chat(chat)


def render_chat_memory_panel() -> None:
    st.subheader("Chat-Speicher")

    opened_title = st.session_state.get("opened_saved_chat_title")
    if opened_title:
        opened_source = st.session_state.get("opened_saved_chat_source_name", "")
        active_source = st.session_state.get("chat_source_name", "Keine Datenquelle")
        st.info(f"Geöffnet: {opened_title}")
        if opened_source and opened_source != active_source:
            st.caption(
                f"Gespeichert mit: {opened_source}. Aktive Datenquelle fuer neue Fragen: {active_source}."
            )

    autosave_error = st.session_state.get("chat_autosave_error")
    if autosave_error:
        st.warning(f"Autosave fehlgeschlagen: {autosave_error}")

    saved_chats = list_chats()
    st.caption(f"Gespeicherte Chats: {len(saved_chats)}")

    if not saved_chats:
        st.info("Noch keine Chats gespeichert.")
        return

    for chat in saved_chats:
        title_text = str(chat.get("title", "Gespeicherter Chat"))
        with st.expander(title_text, expanded=False):
            st.caption(f"{chat.get('created_at', '')} · {chat.get('source_name', '')}")
            st.text(_conversation_preview(chat.get("messages", [])))
            open_col, delete_col = st.columns(2)
            with open_col:
                if st.button("Öffnen", key=f"open_chat_{chat.get('id')}", use_container_width=True):
                    try:
                        open_saved_chat(str(chat.get("id")))
                        st.rerun()
                    except Exception as error:
                        st.error(f"Chat konnte nicht geöffnet werden: {error}")
            with delete_col:
                if st.button("Loeschen", key=f"delete_chat_{chat.get('id')}", use_container_width=True):
                    delete_chat(str(chat.get("id")))
                    if st.session_state.get("opened_saved_chat_id") == str(chat.get("id")):
                        st.session_state.pop("opened_saved_chat_id", None)
                        st.session_state.pop("opened_saved_chat_title", None)
                        st.session_state.pop("opened_saved_chat_source_name", None)
                        st.session_state.pop("chat_autosave_last_saved_at", None)
                        st.session_state.pop("chat_autosave_error", None)
                    st.rerun()
