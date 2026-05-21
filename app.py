#Import von streamlit (Benutzeroberfläche) und pandas zur Datenanalyse
import streamlit as st
import pandas as pd

# Überschrift für die Website
st.title("Data Analysis Copilot")

#Upload Feld für die Dateien
uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])

#Wenn eine Datei hochgeladen wurde, teste Format und führe ... aus. 
if uploaded_file is not None:
    #Den Dateinamen der Datei nehmen und gucken ob richtiges Format hochgeladen wurde. 
    file_name = uploaded_file.name
    #wenn csv,xlsx oder json Datei dann mit pandas lesen
    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)

    elif file_name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)

    elif file_name.endswith(".json"):
        df = pd.read_json(uploaded_file, orient="records")

    #ansosten Fehlermeldung anzeigen, dass der Dateityp nicht unterstützt wird und die Ausführung stoppen.
    else:
        st.error("Dieser Dateityp wird noch nicht unterstützt.(csv, xlsx, json)")
        st.stop()

    #Anzeige das die Datei erfolgreich geladen wurde, damit der Benutzer weiß, dass der Upload funktioniert hat.
    st.success(f"Datei '{file_name}' wurde erfolgreich geladen.")    

    #Erstellen einer Vorschau der importierten Daten
    st.write("Vorschau:")
    st.dataframe(df.head())

    #Durchführen von Analysen. Erstellt eine statistische Zusammenfassung der Daten, zeigt die Spaltennamen, fehlende Werte und Datentypen an. include "all" zeigt alle Spalten an, auch die nicht-numerischen.
    st.write("Statistische Analyse:")
    st.dataframe(df.describe(include="all"))

    #Gibt die Spaltennamen der Daten aus, damit der Benutzer weiß, welche Informationen in der Datei enthalten sind.
    st.subheader("Spaltennamen")
    st.write(df.columns.tolist())

    #Prüft, ob und wie viele fehlende Werte in jeder Spalte sind.
    st.subheader("Fehlende Werte")
    st.write(df.isnull().sum())

    #Gibt die Datentypen der Spalten aus, damit der Benutzer weiß, welche Art von Daten in jeder Spalte enthalten sind.
    st.subheader("Datentypen")
    st.write(df.dtypes)

#Falls keine Datei hochgeladen wurde, wird eine Information angezeigt, die den Benutzer auffordert, eine CSV-Datei hochzuladen.
else:
    st.info("Bitte lade eine CSV-Datei hoch.")

