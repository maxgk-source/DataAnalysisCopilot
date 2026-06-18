"""
Backend-Logik fuer den KI-Chat: LLM, Agent und direkte pandas-Auswertungen.
"""

import re
from typing import Optional

import pandas as pd
import streamlit as st


@st.cache_resource(show_spinner=False)
def _get_llm(api_key: str, api_base: str, model: str, temperature: float):
    """Erstellt ein OpenAI-kompatibles Chat-Modell für AcademicCloud."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as error:
        raise ImportError(
            "Das Paket 'langchain-openai' fehlt. Installiere es mit: "
            "pip install langchain-openai"
        ) from error

    # Aktuelle langchain-openai-Versionen nutzen api_key/base_url.
    # Einige ältere Versionen nutzen openai_api_key/openai_api_base.
    try:
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=api_base,
            temperature=temperature,
        )
    except TypeError:
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=api_base,
            temperature=temperature,
        )


def _create_dataframe_agent(dataframe: pd.DataFrame, api_key: str, api_base: str, model: str, temperature: float):
    """Baut den LangChain Pandas DataFrame Agent."""
    try:
        from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
    except ImportError:
        try:
            from langchain_experimental.agents import create_pandas_dataframe_agent
        except ImportError as error:
            raise ImportError(
                "Das Paket 'langchain-experimental' fehlt. Installiere es mit: "
                "pip install langchain-experimental"
            ) from error

    llm = _get_llm(
        api_key=api_key,
        api_base=api_base,
        model=model,
        temperature=temperature,
    )

    return create_pandas_dataframe_agent(
        llm=llm,
        df=dataframe,
        verbose=False,
        allow_dangerous_code=True,
        max_iterations=30,
        max_execution_time=180,
        early_stopping_method="generate",
        agent_executor_kwargs={"handle_parsing_errors": True},
    )


def _column_name_in_prompt(prompt: str, dataframe: pd.DataFrame) -> Optional[str]:
    """Findet einen Spaltennamen im Prompt, tolerant gegenüber Backticks/Quotes."""
    normalized_prompt = prompt.lower().replace("`", "").replace('"', "").replace("'", "")

    # Exakte Spaltennamen zuerst. Bei year_of_study funktioniert das direkt.
    for column_name in dataframe.columns:
        if str(column_name).lower() in normalized_prompt:
            return str(column_name)

    # Danach einfache Normalisierung: Leerzeichen, Bindestrich und Unterstrich gleich behandeln.
    compact_prompt = re.sub(r"[\s_\-]+", "", normalized_prompt)
    for column_name in dataframe.columns:
        compact_column = re.sub(r"[\s_\-]+", "", str(column_name).lower())
        if compact_column and compact_column in compact_prompt:
            return str(column_name)

    return None


def _format_dataframe_result(dataframe: pd.DataFrame, max_rows: int = 20) -> str:
    """Formatiert ein Ergebnis zuverlässig als Markdown-Tabelle."""
    if dataframe.empty:
        return "Es wurden keine passenden Zeilen gefunden."

    result = dataframe.head(max_rows).copy()
    return result.to_markdown(index=True)


def _looks_like_dataset_overview_request(user_prompt: str) -> bool:
    """Erkennt allgemeine Fragen wie: 'Was kannst du mir anhand dieser CSV sagen?'."""
    prompt_lower = user_prompt.lower()

    overview_patterns = [
        "was kannst du",
        "was siehst du",
        "was fällt auf",
        "was kann man",
        "erzähl mir",
        "sag mir etwas",
        "zusammenfassung",
        "überblick",
        "ueberblick",
        "summary",
        "overview",
        "eda",
        "explorative",
        "erste analyse",
        "kurze analyse",
        "anhand dieser csv",
        "anhand der csv",
        "über diese csv",
        "ueber diese csv",
        "über die daten",
        "ueber die daten",
        "zu dieser datei",
        "anhand dieser datei",
    ]

    data_words = ["csv", "datei", "daten", "datensatz", "tabelle", "dataframe", "dataset"]

    has_overview_word = any(pattern in prompt_lower for pattern in overview_patterns)
    has_data_word = any(word in prompt_lower for word in data_words)

    return has_overview_word and has_data_word


def _safe_markdown_table(dataframe: pd.DataFrame, max_rows: int = 20, index: bool = False) -> str:
    """Markdown-Tabelle mit Fallback, falls tabulate fehlt."""
    try:
        return dataframe.head(max_rows).to_markdown(index=index)
    except Exception:
        return "```text\n" + dataframe.head(max_rows).to_string(index=index) + "\n```"


def _build_direct_dataset_overview(dataframe: pd.DataFrame, source_name: str) -> str:
    """Erstellt eine direkte, robuste pandas-EDA ohne LLM/API-Call.

    Diese Antwort vermeidet AcademicCloud-500-Fehler bei sehr allgemeinen Fragen,
    weil keine Anfrage an das Sprachmodell notwendig ist.
    """
    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    duplicate_count = int(dataframe.duplicated().sum()) if row_count else 0

    dtype_rows = []
    for column_name, dtype in dataframe.dtypes.items():
        missing_count = int(dataframe[column_name].isna().sum())
        unique_count = int(dataframe[column_name].nunique(dropna=True))
        dtype_rows.append(
            {
                "Spalte": str(column_name),
                "Datentyp": str(dtype),
                "Fehlende Werte": missing_count,
                "Fehlend %": f"{missing_count / row_count:.1%}" if row_count else "0.0%",
                "Eindeutige Werte": unique_count,
            }
        )

    dtype_df = pd.DataFrame(dtype_rows)

    missing_sorted = dtype_df.sort_values("Fehlende Werte", ascending=False)
    missing_sorted = missing_sorted[missing_sorted["Fehlende Werte"] > 0]

    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    object_columns = list(dataframe.select_dtypes(include=["object", "category", "bool"]).columns)

    sections: list[str] = []
    sections.append(f"Direkte pandas-Auswertung für **{source_name}**.")
    sections.append(
        "\n".join(
            [
                "## Kurzüberblick",
                f"- **Zeilen:** {row_count}",
                f"- **Spalten:** {column_count}",
                f"- **Doppelte Zeilen:** {duplicate_count}",
                f"- **Numerische Spalten:** {len(numeric_columns)}",
                f"- **Text/Kategorie/Bool-Spalten:** {len(object_columns)}",
            ]
        )
    )

    if not dtype_df.empty:
        sections.append("## Spalten, Datentypen und Datenqualität\n\n" + _safe_markdown_table(dtype_df, max_rows=30, index=False))

    if not missing_sorted.empty:
        sections.append(
            "## Auffälligkeit: fehlende Werte\n\n"
            "Diese Spalten haben fehlende Werte:\n\n"
            + _safe_markdown_table(missing_sorted[["Spalte", "Fehlende Werte", "Fehlend %"]], max_rows=15, index=False)
        )
    else:
        sections.append("## Fehlende Werte\n\nEs wurden keine fehlenden Werte erkannt.")

    if numeric_columns:
        numeric_summary = dataframe[numeric_columns].describe().T.reset_index().rename(columns={"index": "Spalte"})
        # Runde numerische Werte lesbarer.
        for column in numeric_summary.columns:
            if column != "Spalte":
                numeric_summary[column] = pd.to_numeric(numeric_summary[column], errors="coerce").round(3)
        sections.append("## Numerische Spalten: Statistik\n\n" + _safe_markdown_table(numeric_summary, max_rows=20, index=False))

    if object_columns:
        top_value_rows = []
        for column_name in object_columns[:12]:
            value_counts = dataframe[column_name].dropna().astype(str).value_counts().head(3)
            if value_counts.empty:
                top_values = "keine Werte"
            else:
                top_values = ", ".join([f"{idx} ({count})" for idx, count in value_counts.items()])
            top_value_rows.append(
                {
                    "Spalte": str(column_name),
                    "Häufigste Werte": top_values,
                }
            )
        sections.append("## Häufigste Werte in kategorialen/textuellen Spalten\n\n" + _safe_markdown_table(pd.DataFrame(top_value_rows), max_rows=12, index=False))

    # Kleine automatische Interpretation, aber nur aus harten Kennzahlen abgeleitet.
    insights = []
    if duplicate_count:
        insights.append(f"Es gibt **{duplicate_count} doppelte Zeile(n)**; prüfe, ob diese fachlich gewollt sind.")
    if not missing_sorted.empty:
        worst = missing_sorted.iloc[0]
        insights.append(f"Die Spalte mit den meisten fehlenden Werten ist **{worst['Spalte']}** mit **{worst['Fehlende Werte']}** fehlenden Werten.")
    if numeric_columns:
        insights.append("Für numerische Spalten kannst du als Nächstes nach Maximum/Minimum, Ausreißern, Korrelationen oder Gruppenvergleichen fragen.")
    if object_columns:
        insights.append("Für kategoriale Spalten kannst du Häufigkeiten, Gruppierungen und Verteilungen analysieren lassen.")

    if insights:
        sections.append("## Erste ableitbare Hinweise\n\n" + "\n".join(f"- {item}" for item in insights))

    sections.append(
        "## Sinnvolle nächste Fragen\n\n"
        "- `Welche Spalten haben die meisten fehlenden Werte?`\n"
        "- `Gib mir die Zeile mit dem höchsten Wert in <spalte>.`\n"
        "- `Welche Kategorien kommen in <spalte> am häufigsten vor?`\n"
        "- `Gibt es Ausreißer in den numerischen Spalten?`"
    )

    return "\n\n".join(sections)


def _try_direct_pandas_answer(dataframe: pd.DataFrame, source_name: str, user_prompt: str) -> Optional[str]:
    """Beantwortet einfache Datenfragen direkt mit pandas statt über Agent-Schleifen.

    Das verhindert den typischen LangChain-Fehler
    'Agent stopped due to iteration limit or time limit' bei klaren Fragen wie:
    'Gib mir die Zeile mit dem höchsten year_of_study'.
    Zusätzlich werden allgemeine Überblicksfragen direkt ohne LLM beantwortet,
    damit AcademicCloud-500-Fehler bei generischen CSV-Fragen vermieden werden.
    """
    prompt_lower = user_prompt.lower()

    # 0) Allgemeiner CSV-/Datensatz-Überblick wird in v6 bewusst NICHT direkt beantwortet.
    # Solche Fragen gehen an das LLM, aber mit einem vorab berechneten Datenprofil,
    # damit die Antwort wirklich auf der Texteingabe basiert und trotzdem stabil bleibt.

    # 1) Fehlende Werte je Spalte.
    if (
        any(word in prompt_lower for word in ["fehlende", "missing", "null", "nan"])
        and any(word in prompt_lower for word in ["meisten", "höchsten", "highest", "most"])
    ):
        missing = dataframe.isna().sum().sort_values(ascending=False)
        result = pd.DataFrame({
            "Spalte": missing.index,
            "Fehlende Werte": missing.values,
            "Anteil": [f"{value / len(dataframe):.2%}" if len(dataframe) else "0.00%" for value in missing.values],
        })
        return (
            f"Direkte pandas-Auswertung für **{source_name}**.\n\n"
            "Spalten mit den meisten fehlenden Werten:\n\n"
            f"{result.head(20).to_markdown(index=False)}"
        )

    # 2) Zeile(n) mit höchstem/niedrigstem Wert einer Spalte.
    wants_max = any(word in prompt_lower for word in ["höchste", "höchsten", "max", "maximum", "größte", "größten", "highest", "largest"])
    wants_min = any(word in prompt_lower for word in ["niedrigste", "niedrigsten", "min", "minimum", "kleinste", "kleinsten", "lowest", "smallest"])
    wants_row = any(word in prompt_lower for word in ["zeile", "row", "datensatz", "record"] )

    if (wants_max or wants_min) and wants_row:
        column_name = _column_name_in_prompt(user_prompt, dataframe)
        if not column_name:
            return None

        series = dataframe[column_name]
        numeric_series = pd.to_numeric(series, errors="coerce")
        use_numeric = numeric_series.notna().any()
        comparable_series = numeric_series if use_numeric else series
        valid_series = comparable_series.dropna()

        if valid_series.empty:
            return f"Die Spalte `{column_name}` enthält keine auswertbaren Werte."

        target_value = valid_series.max() if wants_max else valid_series.min()
        matching_rows = dataframe.loc[comparable_series == target_value]
        direction_label = "höchsten" if wants_max else "niedrigsten"

        return (
            f"Direkte pandas-Auswertung für **{source_name}**.\n\n"
            f"Die Zeile(n) mit dem **{direction_label} Wert** in `{column_name}` haben den Wert **{target_value}**.\n\n"
            f"{_format_dataframe_result(matching_rows, max_rows=20)}"
        )

    return None


def _dataframe_overview(dataframe: pd.DataFrame) -> str:
    """Erstellt eine kompakte Beschreibung des DataFrames für den Agenten."""
    column_lines = []
    for column_name, dtype in dataframe.dtypes.items():
        column_lines.append(f"- {column_name}: {dtype}")

    return "\n".join(
        [
            f"Zeilen: {len(dataframe)}",
            f"Spalten: {len(dataframe.columns)}",
            "Spalten und Datentypen:",
            *column_lines,
        ]
    )



def _data_profile_for_llm(dataframe: pd.DataFrame, source_name: str) -> str:
    """Erstellt ein kompaktes Datenprofil, das das LLM interpretieren kann.

    Dadurch analysiert das Modell die CSV anhand der Texteingabe, ohne dass ein
    LangChain-Agent in Tool-Schleifen läuft oder die komplette Datei an das LLM
    geschickt werden muss.
    """
    row_count = len(dataframe)
    column_count = len(dataframe.columns)
    duplicate_count = int(dataframe.duplicated().sum()) if row_count else 0

    schema_rows = []
    for column_name, dtype in dataframe.dtypes.items():
        missing_count = int(dataframe[column_name].isna().sum())
        unique_count = int(dataframe[column_name].nunique(dropna=True))
        schema_rows.append({
            "Spalte": str(column_name),
            "Datentyp": str(dtype),
            "Fehlende Werte": missing_count,
            "Fehlend Prozent": round((missing_count / row_count * 100), 2) if row_count else 0.0,
            "Eindeutige Werte": unique_count,
        })
    schema_df = pd.DataFrame(schema_rows)

    parts = [
        f"Datenquelle: {source_name}",
        f"Zeilen: {row_count}",
        f"Spalten: {column_count}",
        f"Doppelte Zeilen: {duplicate_count}",
        "",
        "Schema, Datentypen und Datenqualität:",
        _safe_markdown_table(schema_df, max_rows=80, index=False) if not schema_df.empty else "Keine Spalten.",
    ]

    numeric_columns = list(dataframe.select_dtypes(include="number").columns)
    if numeric_columns:
        numeric_summary = dataframe[numeric_columns].describe().T.reset_index().rename(columns={"index": "Spalte"})
        for column in numeric_summary.columns:
            if column != "Spalte":
                numeric_summary[column] = pd.to_numeric(numeric_summary[column], errors="coerce").round(4)
        parts.extend(["", "Statistik numerischer Spalten:", _safe_markdown_table(numeric_summary, max_rows=80, index=False)])

    categorical_columns = list(dataframe.select_dtypes(include=["object", "category", "bool"]).columns)
    if categorical_columns:
        top_rows = []
        for column_name in categorical_columns[:30]:
            value_counts = dataframe[column_name].dropna().astype(str).value_counts().head(5)
            top_values = "; ".join([f"{idx}: {count}" for idx, count in value_counts.items()]) if not value_counts.empty else "keine Werte"
            top_rows.append({"Spalte": str(column_name), "Top-Werte": top_values})
        parts.extend(["", "Häufigste Werte kategorialer/textueller Spalten:", _safe_markdown_table(pd.DataFrame(top_rows), max_rows=30, index=False)])

    sample = dataframe.head(8).copy()
    parts.extend(["", "Beispielzeilen:", _safe_markdown_table(sample, max_rows=8, index=False)])

    return "\n".join(parts)


def _run_llm_profile_analysis(dataframe: pd.DataFrame, source_name: str, user_prompt: str, api_key: str, api_base: str, model: str, temperature: float) -> str:
    """Lässt das LLM die User-Frage anhand eines pandas-Datenprofils beantworten."""
    llm = _get_llm(api_key=api_key, api_base=api_base, model=model, temperature=temperature)
    data_profile = _data_profile_for_llm(dataframe, source_name)

    prompt = f"""
Du bist ein deutschsprachiger Data-Analysis-Copilot.
Der User stellt eine freie Texteingabe zu einer CSV/DataFrame-Datei.
Beantworte den Analyseauftrag ausschließlich anhand des folgenden Datenprofils.

Wichtige Regeln:
- Erfinde keine Werte.
- Wenn für eine genaue Antwort Rohdaten nötig wären, sage das klar und nenne, welche konkrete Berechnung nötig ist.
- Nutze konkrete Spaltennamen und Zahlen aus dem Datenprofil.
- Antworte praxisnah und strukturiert.
- Schreibe auf Deutsch.

Datenprofil:
{data_profile}

Analyseauftrag des Users:
{user_prompt}
""".strip()

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    return str(content)


def _run_agent(dataframe: pd.DataFrame, source_name: str, user_prompt: str, api_key: str, api_base: str, model: str, temperature: float) -> str:
    """Führt den Analyseauftrag über direkte pandas-Logik oder den LangChain-Agenten aus."""
    direct_answer = _try_direct_pandas_answer(
        dataframe=dataframe,
        source_name=source_name,
        user_prompt=user_prompt,
    )
    if direct_answer:
        return direct_answer

    # v6: Freitextfragen werden vom LLM anhand eines vorab berechneten Datenprofils
    # beantwortet. Der LangChain-Pandas-Agent bleibt als Fallback für komplexe
    # Code-/Berechnungsfragen erhalten, aber allgemeine CSV-Analysen laufen nicht
    # mehr in Agent-Schleifen.
    try:
        return _run_llm_profile_analysis(
            dataframe=dataframe,
            source_name=source_name,
            user_prompt=user_prompt,
            api_key=api_key,
            api_base=api_base,
            model=model,
            temperature=temperature,
        )
    except Exception as llm_error:
        # Wenn der reine LLM-Call fehlschlägt, versuchen wir den LangChain-Agenten
        # als Fallback. Der ursprüngliche Fehler wird danach sichtbar gemacht.
        try:
            agent = _create_dataframe_agent(
                dataframe=dataframe,
                api_key=api_key,
                api_base=api_base,
                model=model,
                temperature=temperature,
            )
            system_context = f"""
Du bist ein deutschsprachiger Data-Analysis-Copilot.
Analysiere ausschließlich den bereitgestellten pandas DataFrame `df`.
Erfinde keine Werte. Wenn eine Aussage nicht aus den Daten ableitbar ist, sage das klar.
Gib konkrete Zahlen, Spaltennamen und kurze Begründungen an.
Antworte direkt und vermeide unnötige Zwischenschritte.

Datenquelle: {source_name}

DataFrame-Übersicht:
{_dataframe_overview(dataframe)}

Analyseauftrag des Users:
{user_prompt}
""".strip()
            response = agent.invoke({"input": system_context})
            if isinstance(response, dict):
                return str(response.get("output", response))
            return str(response)
        except Exception as agent_error:
            raise RuntimeError(
                f"LLM-Profilanalyse fehlgeschlagen: {llm_error}; "
                f"LangChain-Agent-Fallback fehlgeschlagen: {agent_error}"
            )


