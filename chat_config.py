"""
Konfiguration und Secret-Handling fuer den KI-Chat.
"""

import os
from pathlib import Path
from typing import Optional

import streamlit as st


DEFAULT_ACADEMICCLOUD_API_BASE = "https://chat-ai.academiccloud.de/v1"
DEFAULT_ACADEMICCLOUD_MODEL = "meta-llama-3.1-8b-instruct"

FALLBACK_ACADEMICCLOUD_MODELS = [
    "meta-llama-3.1-8b-instruct",
    "qwen3-30b-a3b-instruct-2507",
    "qwen3-coder-30b-a3b-instruct",
    "openai-gpt-oss-120b",
    "deepseek-r1-distill-llama-70b",
    "mistral-large-3-675b-instruct-2512",
    "apertus-70b-instruct-2509",
]


def _candidate_secret_paths() -> list[Path]:
    """Mögliche Orte für secrets.toml.

    Streamlit selbst liest die Projekt-Secrets aus dem Working Directory.
    Diese Zusatzsuche macht die App robuster, falls `streamlit run` aus
    einem anderen Ordner gestartet wurde.
    """
    here = Path(__file__).resolve().parent
    cwd = Path.cwd().resolve()
    return [
        cwd / ".streamlit" / "secrets.toml",
        here / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]


def _read_toml_secret(name: str) -> Optional[str]:
    """Liest einen Root-Level-Key manuell aus secrets.toml als Fallback."""
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # Python <= 3.10
        except ModuleNotFoundError:
            return None

    for path in _candidate_secret_paths():
        if not path.exists():
            continue
        try:
            with path.open("rb") as file:
                data = tomllib.load(file)
            value = data.get(name)
            if value:
                return str(value)
        except Exception:
            continue

    return None


def _read_secret_or_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Liest Streamlit-Secrets, Environment und als Fallback secrets.toml direkt."""
    try:
        value = st.secrets.get(name)  # type: ignore[attr-defined]
        if value:
            return str(value)
    except Exception:
        pass

    value = os.getenv(name)
    if value:
        return value

    value = _read_toml_secret(name)
    if value:
        return value

    return default


def _mask_secret(value: Optional[str]) -> str:
    """Zeigt nur, ob ein Secret vorhanden ist, ohne den Key offenzulegen."""
    if not value:
        return "nicht gefunden"
    if len(value) <= 8:
        return "gefunden, aber sehr kurz"
    return f"gefunden: {value[:4]}…{value[-4:]}"


def _render_secret_diagnostics() -> None:
    """Hilft beim Debuggen, wenn Streamlit den API-Key nicht findet."""
    with st.expander("Debug: Secret-Suche anzeigen", expanded=True):
        st.write(f"**Working Directory:** `{Path.cwd().resolve()}`")
        st.write(f"**Ordner von langchain_agent.py:** `{Path(__file__).resolve().parent}`")
        st.write("**Geprüfte secrets.toml-Pfade:**")
        for path in _candidate_secret_paths():
            st.write(f"- `{path}` — {'gefunden' if path.exists() else 'nicht gefunden'}")

        try:
            st_secret = st.secrets.get("ACADEMICCLOUD_API_KEY")  # type: ignore[attr-defined]
        except Exception as error:
            st_secret = None
            st.caption(f"st.secrets konnte nicht gelesen werden: {error}")

        st.write(f"**st.secrets ACADEMICCLOUD_API_KEY:** {_mask_secret(str(st_secret) if st_secret else None)}")
        st.write(f"**Environment ACADEMICCLOUD_API_KEY:** {_mask_secret(os.getenv('ACADEMICCLOUD_API_KEY'))}")
        st.write(f"**Manueller TOML-Fallback:** {_mask_secret(_read_toml_secret('ACADEMICCLOUD_API_KEY'))}")



@st.cache_data(show_spinner=False, ttl=600)
def _fetch_available_models(api_key: str, api_base: str) -> list[str]:
    """Lädt verfügbare Modell-IDs direkt vom AcademicCloud-/SAIA-API-Endpunkt.

    Laut SAIA-Doku ist `/models` verfügbar; dort wird ein POST-Request gezeigt.
    Manche OpenAI-kompatiblen Gateways akzeptieren GET. Deshalb probieren wir zuerst
    POST und danach GET.
    """
    try:
        import requests
    except ImportError:
        return []

    url = api_base.rstrip("/") + "/models"
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    responses = []
    for method in ("post", "get"):
        try:
            if method == "post":
                response = requests.post(url, headers=headers, timeout=15)
            else:
                response = requests.get(url, headers=headers, timeout=15)
            responses.append(response)
            if response.ok:
                payload = response.json()
                raw_models = payload.get("data", payload if isinstance(payload, list) else [])
                model_ids = []
                for item in raw_models:
                    if isinstance(item, dict):
                        model_id = item.get("id") or item.get("model") or item.get("name")
                    else:
                        model_id = str(item)
                    if model_id:
                        model_ids.append(str(model_id))
                return sorted(set(model_ids))
        except Exception:
            continue

    return []


