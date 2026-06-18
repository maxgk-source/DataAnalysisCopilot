"""
Streamlit-UI fuer den KI-Chat im Data Analysis Copilot.
"""

import io
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from chat_backend import _build_direct_dataset_overview, _run_agent
from chat_config import (
    DEFAULT_ACADEMICCLOUD_API_BASE,
    DEFAULT_ACADEMICCLOUD_MODEL,
    FALLBACK_ACADEMICCLOUD_MODELS,
    _fetch_available_models,
    _read_secret_or_env,
    _render_secret_diagnostics,
)


def _read_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Liest CSV, JSON oder Excel-Dateien als pandas DataFrame."""
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue()

    if file_name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes))

    if file_name.endswith(".json"):
        return pd.read_json(io.BytesIO(file_bytes))

    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(file_bytes))

    raise ValueError("Nur CSV-, JSON- und Excel-Dateien werden unterstützt.")


def render_langchain_agent(
    uploaded_df: Optional[pd.DataFrame] = None,
    db_df: Optional[pd.DataFrame] = None,
    db_source_name: str = "Datenbanktabelle",
    database_config: Optional[dict] = None,
    list_database_tables_func: Optional[Callable[[dict], list[str]]] = None,
    load_table_from_database_func: Optional[Callable[[dict, str, Optional[int]], pd.DataFrame]] = None,
) -> None:
    """
    Rendert den KI-Chat für Streamlit.

    Parameter:
    - uploaded_df/db_df: bereits in der App geladene DataFrames
    - database_config: bestehende PostgreSQL-Konfiguration aus app.py
    - list_database_tables_func/load_table_from_database_func: bestehende DB-Hilfsfunktionen
    """
    st.header("KI-Chat für Datenanalyse")
    st.caption(
        "Wähle eine Datenquelle aus und stelle danach deinen Analyseauftrag im Chat. "
        "Der Agent arbeitet auf einem pandas DataFrame."
    )

    api_key = _read_secret_or_env("ACADEMICCLOUD_API_KEY")
    api_base = _read_secret_or_env("ACADEMICCLOUD_API_BASE", DEFAULT_ACADEMICCLOUD_API_BASE)
    configured_model = _read_secret_or_env("ACADEMICCLOUD_MODEL", DEFAULT_ACADEMICCLOUD_MODEL)

    available_models: list[str] = []
    model_options = list(FALLBACK_ACADEMICCLOUD_MODELS)
    if configured_model and configured_model not in model_options:
        model_options.insert(0, configured_model)

    if api_key:
        available_models = _fetch_available_models(api_key=api_key, api_base=api_base or DEFAULT_ACADEMICCLOUD_API_BASE)
        if available_models:
            model_options = available_models
            if configured_model and configured_model not in model_options:
                st.warning(
                    f"Das konfigurierte Modell `{configured_model}` ist laut `/models` nicht verfügbar. "
                    "Ich verwende stattdessen ein verfügbares Modell aus der Liste."
                )

    default_index = 0
    if configured_model in model_options:
        default_index = model_options.index(configured_model)
    elif DEFAULT_ACADEMICCLOUD_MODEL in model_options:
        default_index = model_options.index(DEFAULT_ACADEMICCLOUD_MODEL)

    with st.expander("Modell- und Sicherheitseinstellungen", expanded=False):
        st.write(f"**API Base:** `{api_base}`")
        model = st.selectbox(
            "AcademicCloud-Modell",
            options=model_options,
            index=default_index,
            key="academiccloud_model_select",
            help=(
                "Falls 'Model Not Found' erscheint, ist die Modell-ID falsch oder für deinen Key nicht freigeschaltet. "
                "Die Liste wird nach Möglichkeit über /models geladen."
            ),
        )
        if available_models:
            st.caption(f"Verfügbare Modelle über `/models` geladen: {len(available_models)}")
        else:
            st.caption("Konnte `/models` nicht laden; es wird eine lokale Fallback-Liste angezeigt.")

        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="0.0 ist für Datenanalyse meistens am reproduzierbarsten.",
        )

    if not api_key:
        st.error(
            "ACADEMICCLOUD_API_KEY fehlt. Lege den Key als Umgebungsvariable "
            "oder in `.streamlit/secrets.toml` ab."
        )
        st.code(
            "ACADEMICCLOUD_API_BASE = \"https://chat-ai.academiccloud.de/v1\"\n"
            "ACADEMICCLOUD_MODEL = \"meta-llama-3.1-8b-instruct\"\n"
            "ACADEMICCLOUD_API_KEY = \"dein_key_hier\"",
            language="toml",
        )
        _render_secret_diagnostics()
        return

    st.subheader("1. Datenquelle wählen")

    source_options = ["PostgreSQL-Tabelle auswählen", "Datei hochladen"]

    if db_df is not None:
        source_options.insert(1, "Bereits geladene PostgreSQL-/CSV-Daten verwenden")

    if uploaded_df is not None:
        source_options.append("Bereits hochgeladene Datei verwenden")

    selected_source_mode = st.radio(
        "Quelle",
        options=source_options,
        horizontal=False,
        key="chat_source_mode",
    )

    chat_dataframe = st.session_state.get("chat_dataframe")
    chat_source_name = st.session_state.get("chat_source_name", "Keine Datenquelle gewählt")

    if selected_source_mode == "PostgreSQL-Tabelle auswählen":
        if not database_config or not list_database_tables_func or not load_table_from_database_func:
            st.error("Die Datenbankfunktionen wurden dem Chat-Tab nicht übergeben.")
        else:
            try:
                tables = list_database_tables_func(database_config)
            except Exception as error:
                st.error(f"PostgreSQL-Tabellen konnten nicht gelesen werden: {error}")
                tables = []

            if not tables:
                st.info("Keine Tabellen gefunden oder Datenbankverbindung nicht verfügbar.")
            else:
                selected_table = st.selectbox(
                    "Tabelle für den Agenten auswählen",
                    options=tables,
                    key="chat_selected_database_table",
                )
                max_rows = st.number_input(
                    "Maximale Zeilen für den Agenten laden",
                    min_value=100,
                    max_value=200000,
                    value=10000,
                    step=1000,
                    key="chat_max_rows",
                    help="Für LLM-gestützte Analyse ist ein Limit sinnvoll. Große Tabellen können langsam und teuer werden.",
                )

                if st.button("Tabelle in den KI-Chat laden", key="load_table_for_chat"):
                    try:
                        chat_dataframe = load_table_from_database_func(
                            database_config,
                            selected_table,
                            int(max_rows),
                        )
                        chat_source_name = f"PostgreSQL-Tabelle: {selected_table}"
                        st.session_state["chat_dataframe"] = chat_dataframe
                        st.session_state["chat_source_name"] = chat_source_name
                        st.success(f"'{selected_table}' wurde für den KI-Chat geladen.")
                    except Exception as error:
                        st.error(f"Die Tabelle konnte nicht geladen werden: {error}")

    elif selected_source_mode == "Bereits geladene PostgreSQL-/CSV-Daten verwenden":
        chat_dataframe = db_df
        chat_source_name = db_source_name
        st.session_state["chat_dataframe"] = chat_dataframe
        st.session_state["chat_source_name"] = chat_source_name
        st.info(f"Aktive Quelle: {chat_source_name}")

    elif selected_source_mode == "Bereits hochgeladene Datei verwenden":
        chat_dataframe = uploaded_df
        chat_source_name = st.session_state.get("uploaded_source_name", "Bereits hochgeladene Datei")
        st.session_state["chat_dataframe"] = chat_dataframe
        st.session_state["chat_source_name"] = chat_source_name
        st.info(f"Aktive Quelle: {chat_source_name}")

    elif selected_source_mode == "Datei hochladen":
        agent_upload = st.file_uploader(
            "Datei nur für den KI-Chat hochladen",
            type=["csv", "json", "xlsx", "xls"],
            key="agent_file_upload",
        )

        if agent_upload is not None:
            try:
                chat_dataframe = _read_uploaded_file(agent_upload)
                chat_source_name = f"Upload im KI-Chat: {agent_upload.name}"
                st.session_state["chat_dataframe"] = chat_dataframe
                st.session_state["chat_source_name"] = chat_source_name
                st.success(f"'{agent_upload.name}' wurde für den KI-Chat geladen.")
            except Exception as error:
                st.error(f"Datei konnte nicht gelesen werden: {error}")

    st.divider()
    st.subheader("2. Chat")

    if chat_dataframe is None:
        st.info("Wähle zuerst eine Datenquelle aus.")
        return

    st.write(f"**Aktive Datenquelle:** {chat_source_name}")

    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Zeilen", len(chat_dataframe))
    metric_col_2.metric("Spalten", len(chat_dataframe.columns))

    with st.expander("Datenvorschau anzeigen", expanded=True):
        st.dataframe(chat_dataframe.head(10), use_container_width=True)

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []

    if st.button("Chatverlauf löschen", key="clear_chat_messages"):
        st.session_state["chat_messages"] = []
        st.rerun()

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "Analyseauftrag eingeben, z. B. 'Welche Spalten haben die meisten fehlenden Werte?'"
    )

    if prompt:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent analysiert die Daten..."):
                try:
                    answer = _run_agent(
                        dataframe=chat_dataframe,
                        source_name=chat_source_name,
                        user_prompt=prompt,
                        api_key=api_key,
                        api_base=api_base,
                        model=model,
                        temperature=temperature,
                    )
                    st.markdown(answer)
                except Exception as error:
                    error_text = str(error)
                    if "Model Not Found" in error_text or "model" in error_text.lower() and "not found" in error_text.lower():
                        answer = (
                            "Fehler beim Ausführen des LangChain-Agenten: Model Not Found. "
                            "Die Modell-ID ist für AcademicCloud/SAIA nicht verfügbar. "
                            "Wähle oben in den Modell-Einstellungen ein anderes Modell, z. B. "
                            "`meta-llama-3.1-8b-instruct` oder `qwen3-30b-a3b-instruct-2507`."
                        )
                    elif "iteration limit" in error_text.lower() or "time limit" in error_text.lower():
                        answer = (
                            "Der Agent ist in eine zu lange Tool-Schleife gelaufen. "
                            "Für einfache Maximum-/Minimum-, Missing-Values- und Überblicksfragen nutzt diese Version nun zuerst eine direkte pandas-Auswertung. "
                            "Bitte lade die v6-Dateien neu oder stelle die Frage mit exaktem Spaltennamen, z. B. `Gib mir die Zeile mit dem höchsten year_of_study`."
                        )
                    elif "500" in error_text or "internal server" in error_text.lower() or "server error" in error_text.lower():
                        answer = (
                            "Der LLM-Endpunkt hat einen Serverfehler 500 zurückgegeben. "
                            "Ich gebe dir deshalb eine direkte pandas-Zusammenfassung ohne LLM/API-Call:\n\n"
                            + _build_direct_dataset_overview(chat_dataframe, chat_source_name)
                        )
                    else:
                        answer = f"Fehler beim Ausführen des LangChain-Agenten: {error}"
                    if answer.startswith("Der LLM-Endpunkt hat einen Serverfehler 500"):
                        st.markdown(answer)
                    else:
                        st.error(answer)

        st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
