"""
Data Analysis Copilot
Was diese App macht:
1. Sie zeigt eine Streamlit-Webseite an.
2. Sie verbindet sich mit einer PostgreSQL-Serverdatenbank.
3. Man kann vorhandene PostgreSQL-Tabellen laden und analysieren und CSV- und JSON-Dateien auf der Webseite hochladen.
4. Hochgeladene CSV- und JSON-Dateien können als neue PostgreSQL-Tabelle gespeichert werden.
5. Die Datenanalyse läuft anschließend über Apache Spark.
"""

# io hilft uns, Bytes wie eine Datei zu behandeln.
# Das brauchen wir, wenn Streamlit eine CSV- oder JSON-Datei hochlädt.
import io
# json hilft uns, JSON-Dateien zu lesen und in Tabellen umzuwandeln.
import json
# os hilft beim Arbeiten mit Betriebssystem-Funktionen:
# z. B. Umgebungsvariablen lesen oder temporäre Dateien löschen.
import os
# re steht für Regular Expressions.
# Damit können wir Texte bereinigen, z. B. ungültige Zeichen aus Tabellennamen entfernen.
import re
# tempfile erstellt temporäre Dateien.
# Spark liest CSV-Dateien am einfachsten über einen Dateipfad.
# Deshalb speichern wir hochgeladene CSVs kurz temporär ab.
import tempfile
# quote_plus macht Texte sicher für eine Datenbank-URL.
# Beispiel: Ein Passwort mit Sonderzeichen wird dadurch korrekt in die URL eingebaut.
from urllib.parse import quote_plus
# Optional bedeutet: Der Wert kann auch None sein.
# Beispiel: Optional[str] heißt: entweder ein String oder None.
from typing import Optional
# pandas ist eine sehr verbreitete Datenanalyse-Bibliothek.
# Wir nutzen pandas zum Lesen von CSV-/JSON-Dateien und PostgreSQL-Tabellen.
import pandas as pd
# streamlit baut die Web-Oberfläche.
# Alles mit st. erscheint später auf der Webseite.
import streamlit as st
# Spark ist gut für größere Datenmengen und DataFrame-Analysen.
from pyspark.sql import SparkSession
# functions enthält Spark-Funktionen wie count, when, isnan, col usw.
# importieren als F, damit der Code kürzer bleibt.
from pyspark.sql import functions as F


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


# SPARK-SESSION
# Eine SparkSession ist wie der Motor von Spark.
# Ohne SparkSession kann Spark keine Daten lesen oder analysieren.

@st.cache_resource
# cache_resource bedeutet:
# Streamlit merkt sich dieses Objekt zwischen Reloads. Ohne Cache würde Spark ständig neu starten.
# Das ist bei Spark wichtig, weil Spark relativ teuer zu starten ist.
def get_spark_session():
    spark = (
        SparkSession.builder
        .appName("DataAnalysisCopilot")# Name der Spark-App. Er erscheint z. B. in Spark-Logs.
        .master("local[*]") # local[*] bedeutet: Spark läuft dem lokalen Rechner und darf alle verfügbaren CPU-Kerne verwenden.
        .getOrCreate() # Wenn schon eine SparkSession existiert, nimm sie. # Sonst erstelle eine neue.
    )
    return spark


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
        name = "daten_tabelle"

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


def dataframe_to_csv_bytes(dataframe: pd.DataFrame) -> bytes:
    #Wandelt ein pandas DataFrame in CSV-Bytes um, weil der download_button Daten als bytes oder string erwartet.
    # index=False entfernt die zeilenummern aus der CSV, damit sie sauberer aussieht.
    return dataframe.to_csv(index=False).encode("utf-8")


def get_uploaded_file_extension(uploaded_file) -> str:
    # Holt die Dateiendung der hochgeladenen Datei, z. B. csv oder json.
    # lower() sorgt dafür, dass CSV, Csv und csv gleich behandelt werden.
    return os.path.splitext(uploaded_file.name)[1].replace(".", "").lower()


def read_json_bytes_to_dataframe(file_bytes: bytes) -> pd.DataFrame:
    # Wandelt JSON-Bytes in ein pandas DataFrame um.
    # JSON kann unterschiedlich aufgebaut sein:
    # 1. Liste von Objekten: [{"name": "Max"}, {"name": "Anna"}]
    # 2. Einzelnes Objekt: {"name": "Max", "alter": 25}
    # 3. JSON Lines: pro Zeile ein JSON-Objekt

    try:
        json_data = json.loads(file_bytes.decode("utf-8"))
    except json.JSONDecodeError:
        # Falls es kein normales JSON ist, versuchen wir JSON Lines.
        # JSON Lines bedeutet: Jede Zeile ist ein eigenes JSON-Objekt.
        return pd.read_json(io.BytesIO(file_bytes), lines=True)

    if isinstance(json_data, list):
        # Eine Liste von Objekten ist der beste Fall für eine Tabelle.
        return pd.json_normalize(json_data)

    if isinstance(json_data, dict):
        # Manche JSON-Dateien haben eine Liste in einem Feld, z. B. {"daten": [...]}
        # Wenn genau eine solche Liste gefunden wird, verwenden wir diese Liste als Tabelle.
        list_values = [value for value in json_data.values() if isinstance(value, list)]

        if len(list_values) == 1 and all(isinstance(item, dict) for item in list_values[0]):
            return pd.json_normalize(list_values[0])

        # Falls alle Werte Listen sind, kann pandas daraus oft direkt eine Tabelle bauen.
        if json_data and all(isinstance(value, list) for value in json_data.values()):
            try:
                return pd.DataFrame(json_data)
            except ValueError:
                pass

        # Sonst wird das einzelne Objekt als eine Zeile gespeichert.
        return pd.json_normalize(json_data)

    raise ValueError("JSON muss entweder ein Objekt, eine Liste oder JSON Lines enthalten.")


def read_uploaded_file_to_dataframe(uploaded_file) -> pd.DataFrame:
    # Liest eine hochgeladene CSV- oder JSON-Datei und gibt immer ein pandas DataFrame zurück.
    file_extension = get_uploaded_file_extension(uploaded_file)
    file_bytes = uploaded_file.getvalue()

    if file_extension == "csv":
        return pd.read_csv(io.BytesIO(file_bytes))

    if file_extension == "json":
        return read_json_bytes_to_dataframe(file_bytes)

    raise ValueError("Nur CSV- und JSON-Dateien werden unterstützt.")


def spark_column(column_name: str):
    #Warum nicht einfach F.col(column_name)? Weil Spalten mit Leerzeichen, Punkte oder Sonderzeichen von Spark sonst falsch intertpretiert werden könnten.
    escaped_column_name = column_name.replace("`", "``")
    return F.col(f"`{escaped_column_name}`")


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


def save_uploaded_file_to_database(
    #Liest eine hochgeladene CSV- oder JSON-Datei und speichert sie als PostgreSQL-Tabelle.
    # Datei als Bytes -> pandas liest CSV/JSON -> pandas speichert DataFrame in PostgreSQL -> Rückgabe des DataFrames für Vorschau
    uploaded_file,
    config: dict,
    table_name: str,
    if_exists: str,
) -> pd.DataFrame:
    dataframe = read_uploaded_file_to_dataframe(uploaded_file)

    engine = get_engine_from_config(config)

    save_dataframe_to_postgres(
        dataframe=dataframe,
        engine=engine,
        table_name=table_name,
        if_exists=if_exists,
    )

    return dataframe


def database_label(config: dict) -> str:
#Baut einen kurzen Text, der dem User zeigt, welche PostgreSQL-Verbindung aktiv ist.
    return (
        f"PostgreSQL: {config['pg_username']}@{config['pg_host']}:"
        f"{config['pg_port']}/{config['pg_database']}"
    )


# Datenanalyse mit Spark

def render_spark_analysis(df_spark, source_name: str):
    #Zeigt eine Spark-Analyse in der Streamlit-Oberfläche.
    #cache merkt sich das DataFrame im Speicher. Dadurch muss Spark es nicht bei jeder folgenden Aktion neu berechnen.
    df_spark = df_spark.cache()

    try:
        # count() zählt die Zeilen.
        # Achtung: Bei sehr großen Daten kann das dauern.
        row_count = df_spark.count()

        # columns ist eine Liste aller Spaltennamen.
        column_count = len(df_spark.columns)

        st.subheader("Spark-Analyse")
        st.caption(f"Datenquelle: {source_name}")

        # Zwei Spalten in der UI für Kennzahlen.
        metric_col_1, metric_col_2 = st.columns(2)
        metric_col_1.metric("Anzahl Zeilen", row_count)
        metric_col_2.metric("Anzahl Spalten", column_count)

        # Datenvorschau anzeigen.
        st.write("Datenvorschau:")
        st.dataframe(df_spark.limit(10).toPandas(), use_container_width=True)

        # Spaltennamen anzeigen.
        st.subheader("Spaltennamen")
        st.write(df_spark.columns)

        # Datentypen anzeigen.
        st.subheader("Datentypen")
        st.dataframe(pd.DataFrame(df_spark.dtypes, columns=["Spalte", "Datentyp"]),use_container_width=True,)

        # describe berechnet einfache Statistik.
        # Bei Textspalten zeigt Spark z. B. count, min, max.
        # Bei Zahlen zusätzlich mean, stddev usw.
        st.subheader("Statistische Analyse")
        st.dataframe(df_spark.describe().toPandas(), use_container_width=True)

        # Fehlende Werte pro Spalte zählen.
        st.subheader("Fehlende Werte")

        missing_value_expressions = []

        for column_name, data_type in df_spark.dtypes:
          column = spark_column(column_name)
          missing_condition = column.isNull()
          # Bei float/double gibt es zusätzlich NaN.
          # NaN bedeutet "Not a Number" und ist nicht dasselbe wie NULL.
          if data_type in ["float", "double"]:
               missing_condition = missing_condition | F.isnan(column)

          missing_value_expressions.append(
               F.count(F.when(missing_condition, column_name)).alias(column_name)
           )

        if missing_value_expressions:
            missing_values = df_spark.select(missing_value_expressions)
            st.dataframe(missing_values.toPandas(), use_container_width=True)
        else:
            st.info("Die Datenquelle enthält keine Spalten.")

    finally:
        # unpersist entfernt das gecachte DataFrame wieder aus Spark.
        # Das spart Speicher.
        df_spark.unpersist()


def analyze_csv_bytes_with_spark(csv_bytes: bytes, source_name: str):
    #Einen temporären Pfad für die CSV-Datei erstellen, damit Spark sie lesen kann. Und danach wieder löschen, damit keine temporären Dateien auf dem Server liegen bleiben.
    spark = get_spark_session()
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(csv_bytes)
            tmp_path = tmp_file.name

        df_spark = (
            spark.read
            # Erste Zeile enthält Spaltennamen.
            .option("header", "true")
            # Spark soll Datentypen automatisch erkennen.
            .option("inferSchema", "true")
            # Erlaubt CSV-Felder mit Zeilenumbrüchen.
            .option("multiLine", "true")
            # Escape-Zeichen für Anführungszeichen.
            .option("escape", '"')
            .csv(tmp_path)
        )

        render_spark_analysis(df_spark, source_name)

    finally:
        # Temporäre Datei immer löschen, auch wenn vorher ein Fehler passiert.
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def analyze_dataframe_with_spark(dataframe: pd.DataFrame, source_name: str):
    # Spark soll weiterhin CSV lesen.
    # Deshalb wandeln wir auch JSON-Daten nach dem Einlesen in eine temporäre CSV-Struktur um.
    # Vorteil: Die eigentliche Spark-Analyse bleibt für CSV und JSON gleich.
    analyze_csv_bytes_with_spark(
        csv_bytes=dataframe_to_csv_bytes(dataframe),
        source_name=source_name,
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


# HAUPTBEREICH: ZWEI TABS
# Tab 1: Tabellen aus PostgreSQL laden und analysieren.
# Tab 2: CSV/JSON hochladen und in PostgreSQL speichern.

tab_database, tab_upload = st.tabs(
    ["Aus PostgreSQL laden & analysieren", "CSV/JSON hochladen & in PostgreSQL speichern"]
)


# TAB 1: AUS POSTGRESQL LADEN UND ANALYSIEREN
with tab_database:
    st.header("Daten aus PostgreSQL analysieren")


    # Eingabe für maximale Zeilen.
    # Bei 0 werden alle Zeilen geladen.
    max_rows = st.number_input(
        "Maximale Zeilen laden, 0 = alle Zeilen", # Ohne name erscheint das feld nicht 
        min_value=0,
        value=0,
        step=1000,
    )

    # Tabellen aus PostgreSQL lesen.
    try:
        tables = list_database_tables(database_config) #liste mit welchen Tabellen in der Datenbank sind
    except Exception as error:
        # Wenn z. B. Zugangsdaten falsch sind, zeigen wir den Fehler in der App.
        st.error(f"PostgreSQL konnte nicht gelesen werden: {error}")
        tables = []

    if not tables:
        st.info(
            "In der verbundenen PostgreSQL-Datenbank gibt es noch keine Tabellen oder sie konnten nicht gelesen werden. "
        )
    else:
        selected_table = st.selectbox( #Auswahlfeld für die Tabellen, die in der Datenbank gefunden wurden.
            "Tabelle auswählen",
            options=tables,
            key="selected_database_table",
        )

        if st.button("Laden/Analysieren"):
            try:
                pandas_dataframe = load_table_from_database( # damit wird die tabelle geladen und in pandas dataframe umgewandelt
                    config=database_config,
                    table_name=selected_table,
                    max_rows=max_rows if max_rows > 0 else None,
                )

                st.success(f"Tabelle '{selected_table}' wurde geladen.")

                st.download_button( #erscheint erst wenn eine Tabelle geladen wurde. 
                    label="Tabelle als CSV herunterladen",
                    data=dataframe_to_csv_bytes(pandas_dataframe),
                    file_name=f"{selected_table}.csv",
                    mime="text/csv",
                )

                analyze_csv_bytes_with_spark(
                    csv_bytes=dataframe_to_csv_bytes(pandas_dataframe),
                    source_name=f"PostgreSQL-Tabelle: {selected_table}",
                )

            except Exception as error:
                st.error(f"Die Tabelle konnte nicht geladen werden: {error}")


# TAB 2: CSV/JSON HOCHLADEN UND ALS POSTGRESQL-TABELLE SPEICHERN
with tab_upload:
    st.header("CSV- oder JSON-Datei hochladen und in PostgreSQL speichern")

    uploaded_file = st.file_uploader( # hochladen einer csv oder json datei 
        "CSV- oder JSON-Datei hochladen",
        type=["csv", "json"],
        key="main_csv_upload",
    )

    if uploaded_file is None:
        st.info("Bitte lade eine CSV- oder JSON-Datei hoch.")
    else:
        # Standard-Tabellenname wird aus dem Dateinamen erstellt.
        default_table_name = sanitize_table_name(uploaded_file.name)

        table_name_input = st.text_input(
            "Name der neuen PostgreSQL-Tabelle",
            value=default_table_name,

        )

        table_name = sanitize_table_name(table_name_input)

        if table_name != table_name_input:
            st.caption(f"Bereinigter Tabellenname: `{table_name}`")

        if_exists_option = st.radio(
            "Was soll passieren, wenn die Tabelle schon existiert?",
            options=[
                "Fehler anzeigen",
                "Tabelle ersetzen",
                "Zeilen anhängen",
            ],
            horizontal=True,
        )

        # pandas.to_sql versteht englische Werte.
        # Diese Mapping-Tabelle übersetzt UI-Text in pandas-Werte.
        if_exists_mapping = {
            "Fehler anzeigen": "fail",
            "Tabelle ersetzen": "replace",
            "Zeilen anhängen": "append",
        }

        analyze_uploaded_file = st.checkbox(
            "Hochgeladene Datei direkt analysieren",
            value=True,
        )

        save_button = st.button("Datei in PostgreSQL speichern")

        if save_button:
            try:
                saved_dataframe = save_uploaded_file_to_database(
                    uploaded_file=uploaded_file,
                    config=database_config,
                    table_name=table_name,
                    if_exists=if_exists_mapping[if_exists_option],
                )

                st.success(
                    f"Datei wurde als Tabelle '{table_name}' in PostgreSQL gespeichert."
                )

                st.write("Vorschau der gespeicherten Daten:")
                st.dataframe(saved_dataframe.head(10), use_container_width=True)

            except ValueError as error:
                # Typischer Fall: Tabelle existiert schon und if_exists="fail".
                st.error(
                    "Die Tabelle existiert bereits oder die Eingabe ist ungültig. "
                    "Wähle entweder 'Tabelle ersetzen' oder 'Zeilen anhängen', falls du die vorhandene Tabelle ändern willst."
                )
                st.exception(error)

            except Exception as error:
                # Alle anderen Fehler, z. B. falsche Datenbankverbindung oder kaputte CSV/JSON-Datei.
                st.error(f"Die Datei konnte nicht gespeichert werden: {error}")

        # Wenn die Checkbox aktiv ist, wird die hochgeladene Datei direkt analysiert.
        # Das passiert unabhängig davon, ob sie schon gespeichert wurde.
        if analyze_uploaded_file:
            try:
                uploaded_dataframe_for_analysis = read_uploaded_file_to_dataframe(uploaded_file)
                analyze_dataframe_with_spark(
                    dataframe=uploaded_dataframe_for_analysis,
                    source_name=f"Hochgeladene Datei: {uploaded_file.name}",
                )
            except Exception as error:
                st.error(f"Die hochgeladene Datei konnte nicht analysiert werden: {error}")
