"""
Backend-Logik fuer den KI-Chat: LLM, Agent und direkte pandas-Auswertungen.
"""

import re
from typing import Any, Optional

import pandas as pd
import streamlit as st

from sandbox_code_executor import (
    SandboxUnavailableError,
    UnsafeGeneratedCodeError,
    execute_code_in_sandbox,
    strip_code_fences,
    validate_generated_code,
)


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


def _truncate_text(text: str, max_chars: int = 5000) -> str:
    """Verhindert sehr lange Chat-Ausgaben aus Sandbox-Logs."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[gekuerzt]"


def _looks_like_sandboxed_code_request(user_prompt: str) -> bool:
    """Erkennt Fragen, bei denen der Agent wahrscheinlich Python-Code braucht."""
    prompt_lower = user_prompt.lower()
    sandbox_keywords = [
        "ausfuehren",
        "ausführen",
        "berechne",
        "calculate",
        "chart",
        "code",
        "correlation",
        "diagramm",
        "durchschnitt",
        "filtere",
        "grafik",
        "groupby",
        "gruppiere",
        "histogramm",
        "korrelation",
        "median",
        "plot",
        "pivot",
        "python",
        "regression",
        "scatter",
        "sortiere",
        "standardabweichung",
        "verteilung",
        "visualisiere",
        "visualisierung",
    ]
    return any(keyword in prompt_lower for keyword in sandbox_keywords)


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


def _generate_sandbox_code(
    dataframe: pd.DataFrame,
    source_name: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    conversation_context: str = "",
) -> str:
    """Laesst das LLM pandas-Code fuer die Sandbox erzeugen."""
    llm = _get_llm(api_key=api_key, api_base=api_base, model=model, temperature=temperature)
    data_profile = _data_profile_for_llm(dataframe, source_name)
    chat_context = (
        _truncate_text(conversation_context.strip(), max_chars=3000)
        if conversation_context.strip()
        else "Kein bisheriger Chatkontext."
    )

    prompt = f"""
Du erzeugst Python-Code fuer eine isolierte E2B-Sandbox.
Antworte ausschliesslich mit Python-Code, ohne Markdown und ohne Erklaertext.
Nutze keine nummerierten Listen, keine Zeilennummern und keinen Text ausserhalb des Codes.

Verfuegbare Variablen und Bibliotheken:
- df: pandas DataFrame mit den Daten
- pd, np, plt, sns

Regeln:
- Beantworte den Analyseauftrag durch Berechnungen direkt auf df, nicht nur anhand des Datenprofils.
- Fuehre keine Datei-, Netzwerk-, Shell-, Betriebssystem- oder Secret-Zugriffe aus.
- Nutze keine open/eval/exec/compile/globals/locals und keine os/sys/subprocess/socket/requests/urllib/pathlib/shutil APIs.
- Importiere nichts ausser pandas, numpy, matplotlib, seaborn, math oder statistics.
- Veraendere df nur auf Kopien, wenn die Originaldaten erhalten bleiben sollten.
- Drucke eine deutsche Antwort mit den wichtigsten Ergebnissen per print().
- Wenn ein tabellarisches Ergebnis sinnvoll ist, speichere es als pandas DataFrame oder Series in der Variable result.
- Wenn der User ein Diagramm, einen Plot, eine Visualisierung oder "anzeigen" verlangt, muss der Code eine matplotlib/seaborn-Figur erzeugen.
- Bei Follow-up-Fragen wie "das", "den Zusammenhang" oder "als Diagramm" nutze den bisherigen Chatverlauf, um die gemeinten Spalten zu bestimmen.
- Wenn es um den Zusammenhang/Korrelation zwischen zwei numerischen Spalten geht, berechne die Korrelation und erzeuge einen Scatterplot oder Regressionsplot.
- Erzeuge keine Plot-Datei und nutze kein savefig; die aktive Figur wird automatisch gespeichert.
- Halte die Ausgabe kompakt und fachlich direkt.

Datenprofil nur als Orientierung fuer Spalten, Datentypen und Beispielwerte:
{data_profile}

Bisheriger Chatverlauf als Kontext fuer Follow-up-Fragen:
{chat_context}

Analyseauftrag:
{user_prompt}
""".strip()

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    code = strip_code_fences(str(content))

    last_error = ""
    for _ in range(3):
        try:
            validate_generated_code(code)
            return code
        except UnsafeGeneratedCodeError as error:
            last_error = str(error)
            code = _repair_sandbox_code(
                llm=llm,
                broken_code=code,
                validation_error=last_error,
                data_profile=data_profile,
                user_prompt=user_prompt,
                conversation_context=conversation_context,
            )

    fallback_code = _build_safe_fallback_code(user_prompt)
    try:
        validate_generated_code(fallback_code)
        return fallback_code
    except UnsafeGeneratedCodeError as fallback_error:
        raise UnsafeGeneratedCodeError(
            f"Code-Reparatur fehlgeschlagen: {last_error}; Fallback ungueltig: {fallback_error}"
        ) from fallback_error


def _repair_sandbox_code(
    llm,
    broken_code: str,
    validation_error: str,
    data_profile: str,
    user_prompt: str,
    conversation_context: str = "",
) -> str:
    """Laesst das LLM ungueltigen Python-Code reparieren."""
    chat_context = conversation_context.strip() or "Kein bisheriger Chatkontext."
    prompt = f"""
Der folgende Python-Code ist syntaktisch ungueltig oder in der Sandbox nicht erlaubt.
Antworte ausschliesslich mit korrigiertem Python-Code, ohne Markdown, ohne Erklaertext und ohne Zeilennummern.

Strikte Regeln:
- Jeder if/for/while/try/except/with/function/class-Block muss einen eingerueckten Codeblock enthalten.
- Wenn du keinen Block brauchst, verwende keinen Block.
- Fuehre keine Datei-, Netzwerk-, Shell-, Betriebssystem- oder Secret-Zugriffe aus.
- Nutze nur df, pd, np, plt und sns.
- Speichere Tabellen in result und drucke eine kurze deutsche Antwort mit print().
- Wenn der User ein Diagramm/Plot verlangt, erzeuge eine matplotlib/seaborn-Figur ohne savefig.

Datenprofil:
{data_profile}

Bisheriger Chatverlauf:
{chat_context}

Analyseauftrag:
{user_prompt}

Fehler:
{validation_error}

Ungueltiger Code:
{broken_code}
""".strip()

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    return strip_code_fences(str(content))


def _build_safe_fallback_code(user_prompt: str) -> str:
    """Robuster Fallback, wenn das LLM wiederholt ungueltigen Code erzeugt."""
    escaped_prompt = user_prompt.replace("\\", "\\\\").replace('"', '\\"')
    return f'''
print("Ich habe eine robuste Grundanalyse ausgefuehrt.")
print("Analyseauftrag: {escaped_prompt}")
print(f"Datensatz: {{df.shape[0]}} Zeilen x {{df.shape[1]}} Spalten")
print("Spalten:")
print(", ".join(map(str, df.columns)))

summary_rows = []
for column in df.columns:
    series = df[column]
    summary_rows.append({{
        "Spalte": str(column),
        "Datentyp": str(series.dtype),
        "Fehlende Werte": int(series.isna().sum()),
        "Eindeutige Werte": int(series.nunique(dropna=True)),
    }})

result = pd.DataFrame(summary_rows)

numeric_columns = list(df.select_dtypes(include="number").columns)
if numeric_columns:
    print("Numerische Kurzstatistik:")
    print(df[numeric_columns].describe().round(3).to_string())
else:
    print("Es wurden keine numerischen Spalten fuer eine Kurzstatistik gefunden.")
'''.strip()


def _summarize_sandbox_execution(
    execution: dict[str, Any],
    source_name: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    conversation_context: str = "",
) -> str:
    """Erklaert Sandbox-Ergebnisse in natuerlicher Sprache."""
    llm = _get_llm(api_key=api_key, api_base=api_base, model=model, temperature=temperature)
    output = _truncate_text(str(execution.get("output") or "").strip(), max_chars=4000)
    chat_context = (
        _truncate_text(conversation_context.strip(), max_chars=2500)
        if conversation_context.strip()
        else "Kein bisheriger Chatkontext."
    )

    table = execution.get("table")
    if isinstance(table, pd.DataFrame) and not table.empty:
        table_text = _safe_markdown_table(table, max_rows=20, index=False)
    else:
        table_text = "Keine Ergebnis-Tabelle vorhanden."

    figure_note = (
        "Es wurde ein Diagramm erzeugt."
        if execution.get("figure_bytes")
        else "Es wurde kein Diagramm erzeugt."
    )
    warning = _truncate_text(str(execution.get("warning") or "").strip(), max_chars=1500)
    error = _truncate_text(str(execution.get("error") or "").strip(), max_chars=1500)

    prompt = f"""
Du bist ein deutschsprachiger Data-Analysis-Copilot.
Erklaere die Ergebnisse einer Python/pandas-Sandbox-Ausfuehrung in natuerlicher Sprache.

Regeln:
- Antworte auf Deutsch.
- Beziehe dich auf konkrete Zahlen und Spalten aus der Sandbox-Ausgabe.
- Erfinde keine Werte, die nicht in Ausgabe oder Tabelle stehen.
- Wenn ein Diagramm erzeugt wurde, erklaere kurz, was es zeigt.
- Wenn Fehler vorhanden sind, sage klar, was nicht berechnet werden konnte.
- Halte die Antwort gut lesbar und praxisnah.
- Gib keine Python-Code-Erklaerung als Hauptantwort, sondern die fachliche Interpretation.

Datenquelle:
{source_name}

Analyseauftrag:
{user_prompt}

Bisheriger Chatverlauf:
{chat_context}

Sandbox-Ausgabe:
{output if output else "Keine Textausgabe vorhanden."}

Ergebnis-Tabelle:
{table_text}

Diagramm:
{figure_note}

Warnungen:
{warning if warning else "Keine Warnungen."}

Fehler:
{error if error else "Keine Fehler."}
""".strip()

    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    return str(content).strip()


def _run_sandboxed_code_analysis(
    dataframe: pd.DataFrame,
    source_name: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    sandbox_api_key: Optional[str],
    conversation_context: str = "",
) -> dict[str, Any]:
    """Generiert Analyse-Code und fuehrt ihn in E2B aus."""
    if not sandbox_api_key:
        raise SandboxUnavailableError(
            "E2B_API_KEY fehlt. Ohne E2B wird kein vom Agenten erzeugter Code lokal ausgefuehrt."
        )

    code = _generate_sandbox_code(
        dataframe=dataframe,
        source_name=source_name,
        user_prompt=user_prompt,
        api_key=api_key,
        api_base=api_base,
        model=model,
        temperature=temperature,
        conversation_context=conversation_context,
    )
    execution = execute_code_in_sandbox(
        code=code,
        dataframe=dataframe,
        api_key=sandbox_api_key,
    )
    natural_answer = _summarize_sandbox_execution(
        execution=execution,
        source_name=source_name,
        user_prompt=user_prompt,
        api_key=api_key,
        api_base=api_base,
        model=model,
        temperature=temperature,
        conversation_context=conversation_context,
    )
    return {
        "content": natural_answer,
        "figure_bytes": execution.get("figure_bytes"),
        "code": code,
    }


def _run_agent(
    dataframe: pd.DataFrame,
    source_name: str,
    user_prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    temperature: float,
    use_sandbox: bool = False,
    sandbox_api_key: Optional[str] = None,
    conversation_context: str = "",
) -> str | dict[str, Any]:
    """Fuehrt den Analyseauftrag als Sandbox-Code auf dem echten DataFrame aus."""
    if use_sandbox:
        try:
            return _run_sandboxed_code_analysis(
                dataframe=dataframe,
                source_name=source_name,
                user_prompt=user_prompt,
                api_key=api_key,
                api_base=api_base,
                model=model,
                temperature=temperature,
                sandbox_api_key=sandbox_api_key,
                conversation_context=conversation_context,
            )
        except SandboxUnavailableError as sandbox_error:
            return (
                "Die sichere Code-Sandbox konnte nicht genutzt werden: "
                f"{sandbox_error}\n\n"
                "Ich fuehre deshalb keinen generierten Python-Code lokal aus. "
                "Du kannst E2B mit `E2B_API_KEY` konfigurieren oder die Frage ohne Code-Ausfuehrung stellen."
            )
        except UnsafeGeneratedCodeError as code_error:
            return (
                "Der Agent hat keinen gueltigen Python-Code erzeugt: "
                f"{code_error}\n\n"
                "Ich habe den Code nicht ausgefuehrt. Formuliere die Analysefrage bitte etwas konkreter, "
                "z. B. mit dem Spaltennamen und der gewuenschten Berechnung oder Visualisierung."
            )

    # Fallback fuer Tests oder bewusst deaktivierte Sandbox. In der Streamlit-App
    # ist use_sandbox=True fest gesetzt, damit der Agent am echten df arbeitet.
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
        raise RuntimeError(
            f"LLM-Profilanalyse fehlgeschlagen: {llm_error}; "
            "lokale Agent-Code-Ausfuehrung ist aus Sicherheitsgruenden deaktiviert. "
            "Aktiviere die E2B-Sandbox mit E2B_API_KEY, wenn der Agent Python-Code ausfuehren soll."
        ) from llm_error


