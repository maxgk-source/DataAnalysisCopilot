"""
Data Analysis Copilot
Was diese App macht:
1. Sie zeigt eine Streamlit-Webseite an.
2. Sie verbindet sich mit einer PostgreSQL-Serverdatenbank.
3. Man kann CSV-Dateien oder PostgreSQL-Tabellen in den KI-Chat laden.
4. Hochgeladene Dateien können optional als neue PostgreSQL-Tabelle gespeichert werden.
5. Der KI-Agent analysiert die Daten per pandas-Code in einer Sandbox.
"""


# os hilft beim Arbeiten mit Betriebssystem-Funktionen:
import os
# re steht für Regular Expressions.
# Damit kann man Texte bereinigen, z. B. ungültige Zeichen aus Tabellennamen entfernen.
import re
# quote_plus macht Texte sicher für eine Datenbank-URL.
# Beispiel: Ein Passwort mit Sonderzeichen wird dadurch korrekt in die URL eingebaut.
from urllib.parse import quote_plus
# Optional bedeutet: Der Wert kann auch None sein.
# Beispiel: Optional[str] heißt: entweder ein String oder None.
from typing import Optional
# pandas ist eine sehr verbreitete Datenanalyse-Bibliothek.
# Wir nutzen pandas zum Lesen von CSV-Dateien und PostgreSQL-Tabellen.
import pandas as pd
# streamlit baut die Web-Oberfläche.
# Alles mit st. erscheint später auf der Webseite.
import streamlit as st
from chat_memory import render_chat_memory_panel
from langchain_agent import render_langchain_agent

# PostgreSQL verbindt man über SQLAlchemy + psycopg2.
try:
    # create_engine baut die Verbindung zur PostgreSQL-Datenbank.
    # inspect kann Tabellen in der Datenbank finden.
    from sqlalchemy import create_engine, inspect
except ImportError:
    # Wenn der Import fehlschlägt, setzt man die Variablen auf None.
    # Später prüft man das und erklärt dem User, was installiert werden muss.
    create_engine = None
    inspect = None


# STREAMLIT EINSTELLUNGEN
# set_page_config stellt Seitentitel und Layout der App ein.
# layout="wide" bedeutet: Die App nutzt mehr Bildschirmbreite.
st.set_page_config(page_title="Data Analysis Copilot", layout="wide")

# title erzeugt die große Überschrift auf der Webseite.
st.title("Data Analysis Copilot")


#HILFSFUNKTIONEN die man mehrfach benutzen kann.Hauptcode bleibt übersichtlicher.
def sanitize_table_name(name: str) -> str: #macht aus data analysis copilot = data_analysis_copilot. um besser in postgres zu passen.
    # os.path.basename nimmt nur den Dateinamen ohne Ordnerpfad. falls man eine tabelle lokal hochlädt
    # os.path.splitext entfernt die Dateiendung, damit man nur den namen der datei hat dafür sort die 0 (erster teil vom Namen)
    name = os.path.splitext(os.path.basename(name))[0]

    # Diese Zeile ersetzt alles, was NICHT Buchstabe, Zahl oder Unterstrich ist,durch einen Unterstrich. damit man keine Leerzeichen oder Sonderzeichen hat
    # .strip("_") entfernt unnötige Unterstriche am Anfang und Ende
    name = re.sub(r"[^a-zA-Z0-9_]+", "_", name).strip("_").lower()

    # Falls danach nichts mehr übrig ist, nimmt man einen Standardnamen.
    if not name:
        name = "csv_tabelle"

    # SQL-Tabellennamen, die mit einer Zahl beginnen, können problematisch sein. Falls man SQL-Befehle ausführen möchte.
    if name[0].isdigit():
        name = f"t_{name}"
    return name


def quote_identifier(identifier: str) -> str:
    #Setzt doppelte Anführungszeichen um einen SQL-Identifier. Um die Namen eindeutiger zu machen, z. B. wenn sie Sonderzeichen oder reservierte Wörter enthalten.
    return '"' + identifier.replace('"', '""') + '"'


def build_table_reference(table_name: str) -> str:
    # Baut den SQL-Namen einer PostgreSQL-Tabelle.
    # Wir geben hier kein Schema mehr an. PostgreSQL nutzt dadurch automatisch
    # das Standard-Schema der Verbindung, meistens public.
    return quote_identifier(table_name)


# POSTGRESQL-FUNKTIONEN

@st.cache_resource(show_spinner=False)
# PostgreSQL-Verbindungen werden gecacht.
# Sonst würde Streamlit bei jeder kleinen UI-Änderung eine neue Verbindung aufbauen.
def get_postgres_engine(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
):
    # Diese drei Werte sind Pflicht.
    # Ohne sie weiß die App nicht, wohin sie sich verbinden soll.
    if not host or not database or not username: raise ValueError("Host, Datenbankname und Benutzername müssen gesetzt sein.")

    # PostgreSQL-Verbindungs-URL bauen.
    # quote_plus schützt Benutzername, Passwort und Datenbankname vor Sonderzeichenproblemen.
    url = ( "postgresql+psycopg2://"f"{quote_plus(username)}:{quote_plus(password)}"f"@{host}:{int(port)}/{quote_plus(database)}")

    return create_engine(
        url,
        pool_pre_ping=True, #Prüft vor Nutzung die Verbindung, verhindert Fehler durch abgelaufene Verbindungen.
    )


def list_postgres_tables(engine) -> list[str]:
    #Liest alle Tabellen aus PostgreSQL-Verbindung.
    inspector = inspect(engine) #fragt nach Struktur der Datenbank
    tables = inspector.get_table_names() # holt die Namen aller Tabellen in der Datenbank
    return sorted(tables) # sortiert die Tabellennamen alphabetisch


def load_postgres_table( #Lädt eine PostgreSQL-Tabelle als pandas DataFrame.
    engine,
    table_name: str,
    max_rows: Optional[int] = None, #komplette tabelle wird geladen
) -> pd.DataFrame:
    table_reference = build_table_reference(table_name)
    query = f"SELECT * FROM {table_reference}" #SQL Abfrage wird gebaut 

    # Optionales Limit hinzufügen.
    # Das ist bei großen Tabellen wichtig, weil die Daten lokal in pandas geladen werden.
    if max_rows and max_rows > 0:
        query += f" LIMIT {int(max_rows)}"

    return pd.read_sql_query(query, engine)


def save_dataframe_to_postgres( #Speichert ein pandas DataFrame als PostgreSQL-Tabelle.
    dataframe: pd.DataFrame,
    engine,
    table_name: str,
    if_exists: str,
) -> None:
    dataframe.to_sql(
        name=table_name,#Tabellenname
        con=engine,#Datenbankverbindung
        if_exists=if_exists,#Verhalten, wenn die Tabelle schon existiert: "fail", "replace" oder "append".
        index=False,# pandas-Index nicht speichern.
        chunksize=1000,# In Blöcken schreiben, statt alles auf einmal. bessere performance bei großen Daten.
        method="multi",# "multi" schreibt mehrere Zeilen pro INSERT
    )


def build_postgres_config() -> dict:
#Damit man die Werte der PostgreSQL-Verbindung an einemn Ort hat und nicht immer neu eingeben muss
    return {
        "pg_host": pg_host,
        "pg_port": int(pg_port),
        "pg_database": pg_database,
        "pg_username": pg_username,
        "pg_password": pg_password,
    }


def get_engine_from_config(config: dict):
#Baut aus dem Dict die verbindungsdatei zur Datenbank
    return get_postgres_engine(
        host=config["pg_host"],
        port=config["pg_port"],
        database=config["pg_database"],
        username=config["pg_username"],
        password=config["pg_password"],
    )


def list_database_tables(config: dict) -> list[str]:
    #Gibt alle Tabellen aus der verbundenen PostgreSQL-Datenbank zurück.
    engine = get_engine_from_config(config)
    return list_postgres_tables(engine)


def load_table_from_database(
    #Lädt eine PostgreSQL-Tabelle mit eine Dataframe als Ergebnis.
    config: dict,
    table_name: str,
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    
    engine = get_engine_from_config(config)

    return load_postgres_table(
        engine=engine,
        table_name=table_name,
        max_rows=max_rows,
    )


def save_loaded_dataframe_to_database(
    dataframe: pd.DataFrame,
    config: dict,
    table_name: str,
    if_exists: str,
) -> None:
    #Speichert ein bereits geladenes pandas DataFrame aus dem KI-Chat in PostgreSQL.
    engine = get_engine_from_config(config)
    save_dataframe_to_postgres(
        dataframe=dataframe,
        engine=engine,
        table_name=table_name,
        if_exists=if_exists,
    )


def database_label(config: dict) -> str:
#Baut einen kurzen Text, der dem User zeigt, welche PostgreSQL-Verbindung aktiv ist.
    return (
        f"PostgreSQL: {config['pg_username']}@{config['pg_host']}:"
        f"{config['pg_port']}/{config['pg_database']}"
    )


# SIDEBAR Konfiguration für die PostgreSQL-Verbindung Die Sidebar ist der linke Bereich
st.sidebar.header("PostgreSQL-Datenbank")
st.sidebar.caption("Hier die Datenbankverbindung einstellen.")

# Host der PostgreSQL-Datenbank.
# os.getenv("PGHOST", "localhost") bedeutet:
# Nimm die Umgebungsvariable PGHOST, falls sie existiert.
# Sonst nutze "localhost".
pg_host = st.sidebar.text_input(
    "Host",
    value=os.getenv("PGHOST", "localhost"),
)

# Port der PostgreSQL-Datenbank. Step nicht angegeben, weil standardmäßig 1 als Schrittgröße gilt. 
pg_port = st.sidebar.number_input(
    "Port",
    min_value=1,
    max_value=65535,
    value=int(os.getenv("PGPORT", "5433")),
)

# Name der Datenbank, z. B. analytics oder postgres.
pg_database = st.sidebar.text_input(
    "Datenbank",
    value=os.getenv("PGDATABASE", ""),
    placeholder="z. B. postgres",
)

# PostgreSQL-Benutzername.
pg_username = st.sidebar.text_input(
    "Benutzername",
    value=os.getenv("PGUSER", ""),
)

# Passwortfeld.
# type="password" sorgt dafür, dass das Passwort in der UI verdeckt wird.
pg_password = st.sidebar.text_input(
    "Passwort",
    value=os.getenv("PGPASSWORD", ""),
    type="password",
)

# Alle Datenbankeinstellungen werden in einem Dictionary gesammelt.
database_config = build_postgres_config()

# Zeigt unten in der Sidebar, welche Verbindung gerade aktiv ist.
st.sidebar.caption(f"Aktive Verbindung: {database_label(database_config)}")

# Button: Verbindung testen.
if st.sidebar.button("PostgreSQL-Verbindung testen"):
    try:
        # Wenn Tabellen gelesen werden können, funktioniert die Verbindung.
        table_count = len(list_database_tables(database_config))
        st.sidebar.success(f"Verbindung erfolgreich. Gefundene Tabellen: {table_count}")
    except Exception as error:
        # Fehler sichtbar anzeigen, damit man z. B. falsche Zugangsdaten erkennt.
        st.sidebar.error(f"Verbindung fehlgeschlagen: {error}")


# HAUPTBEREICH: KI-CHAT MIT RECHTEM CHAT-SPEICHER
_uploaded_df = st.session_state.get("uploaded_dataframe", None)
_db_df = st.session_state.get("active_dataframe", None)
_db_source = st.session_state.get("active_source_name", "Datenbanktabelle")

chat_column, memory_column = st.columns([0.72, 0.28], gap="large")

with chat_column:
    render_langchain_agent(
        uploaded_df=_uploaded_df,
        db_df=_db_df,
        db_source_name=_db_source,
        database_config=database_config,
        list_database_tables_func=list_database_tables,
        load_table_from_database_func=load_table_from_database,
        save_dataframe_to_database_func=save_loaded_dataframe_to_database,
        sanitize_table_name_func=sanitize_table_name,
    )

with memory_column:
    render_chat_memory_panel()
