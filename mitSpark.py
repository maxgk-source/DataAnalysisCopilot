import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


st.title("Data Analysis Copilot")


# SparkSession nur einmal starten
# Das ist wichtig, weil Spark sonst bei jedem Streamlit-Reload neu gestartet wird.
@st.cache_resource
def get_spark_session():
    spark = (
        SparkSession.builder
        .appName("DataAnalysisCopilot")
        .master("local[*]")  # nutzt alle verfügbaren CPU-Kerne lokal
        .getOrCreate()
    )
    return spark


uploaded_file = st.file_uploader(
    "CSV-Datei hochladen",
    type=["csv"],
    key="main_csv_upload"
)

if uploaded_file is not None:
        spark = get_spark_session()

        # Streamlit gibt eine hochgeladene Datei als Objekt zurück.
        # Spark braucht aber meistens einen Dateipfad.
        # Deshalb speichern wir die Datei temporär ab.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # CSV-Datei mit Spark laden
        df_spark = (
            spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(tmp_path)
        )

        st.subheader("Spark-Analyse")

        st.write("Datenvorschau:")

        # Wichtig:
        # Streamlit kann Spark DataFrames nicht direkt schön anzeigen.
        # Deshalb wandeln wir nur die ersten Zeilen in Pandas um.
        st.dataframe(df_spark.limit(10).toPandas())

        st.write("Anzahl Zeilen:")
        st.write(df_spark.count())

        st.write("Anzahl Spalten:")
        st.write(len(df_spark.columns))

        st.subheader("Spaltennamen")
        st.write(df_spark.columns)

        st.subheader("Datentypen")
        st.dataframe(
            pd.DataFrame(df_spark.dtypes, columns=["Spalte", "Datentyp"])
        )

        st.subheader("Statistische Analyse")
        st.dataframe(df_spark.describe().toPandas())

        st.subheader("Fehlende Werte")

        missing_values = df_spark.select([
            F.count(
                F.when(
                    F.col(column).isNull(),
                    column
                )
            ).alias(column)
            for column in df_spark.columns
        ])

        st.dataframe(missing_values.toPandas())

        # Temporäre Datei wieder löschen
        os.remove(tmp_path)

else:
    st.info("Bitte lade eine CSV-Datei hoch.")