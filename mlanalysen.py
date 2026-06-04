"""
1. Clustering
   - findet Gruppen in Daten ohne Zielvariable
   - geeignet für Kundensegmente, Produktgruppen, Mustererkennung
2. Anomalie-Erkennung
   - findet ungewöhnliche Datensätze / Ausreißer
   - geeignet für Betrug, Fehler, Extremwerte

"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Gibt alle numerischen Spalten zurück.
    Clustering und Anomalie-Erkennung funktionieren hier nur mit Zahlenwerten.
    """
    return dataframe.select_dtypes(include=["number"]).columns.tolist()


def clean_numeric_dataframe(
    dataframe: pd.DataFrame,
    selected_columns: list[str],
) -> pd.DataFrame:
    """
    Erstellt ein bereinigtes numerisches DataFrame.

    - nimmt nur ausgewählte numerische Spalten
    - ersetzt unendliche Werte durch NaN
    - entfernt Spalten, die komplett leer sind
    """
    numeric_dataframe = dataframe[selected_columns].copy()

    numeric_dataframe = numeric_dataframe.replace([np.inf, -np.inf], np.nan)

    empty_columns = [
        column for column in numeric_dataframe.columns
        if numeric_dataframe[column].isna().all()
    ]

    if empty_columns:
        numeric_dataframe = numeric_dataframe.drop(columns=empty_columns)

    return numeric_dataframe


def build_preprocessing_pipeline() -> Pipeline:
    """
    Baut eine Preprocessing-Pipeline für ML.

    SimpleImputer:
        ersetzt fehlende Werte durch den Median der jeweiligen Spalte.

    StandardScaler:
        skaliert Zahlenwerte, damit Spalten mit großen Werten
        nicht automatisch wichtiger werden als Spalten mit kleinen Werten.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )


def render_clustering_analysis(
    dataframe: pd.DataFrame,
    source_name: str = "Datenquelle",
):
    """
    Rendert eine Clustering-Analyse in Streamlit.

    Das Clustering nutzt K-Means.
    K-Means gruppiert ähnliche Datensätze anhand numerischer Spalten.
    """

    st.subheader("Clustering")
    st.caption(f"Datenquelle: {source_name}")

    numeric_columns = get_numeric_columns(dataframe)

    if len(numeric_columns) < 2:
        st.warning(
            "Für Clustering werden mindestens zwei numerische Spalten benötigt."
        )
        return

    selected_columns = st.multiselect(
        "Numerische Spalten für Clustering auswählen",
        options=numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
        key="clustering_selected_columns",
    )

    if len(selected_columns) < 2:
        st.info("Bitte wähle mindestens zwei numerische Spalten aus.")
        return

    cluster_count = st.slider(
        "Anzahl Cluster",
        min_value=2,
        max_value=10,
        value=3,
        step=1,
        key="cluster_count",
    )

    run_button = st.button(
        "Clustering berechnen",
        key="run_clustering_button",
    )

    if not run_button:
        return

    numeric_dataframe = clean_numeric_dataframe(
        dataframe=dataframe,
        selected_columns=selected_columns,
    )

    if numeric_dataframe.empty:
        st.error("Nach der Bereinigung sind keine nutzbaren numerischen Daten übrig.")
        return

    if len(numeric_dataframe) < cluster_count:
        st.error(
            "Die Anzahl der Datensätze muss größer oder gleich der Anzahl der Cluster sein."
        )
        return

    try:
        preprocessing_pipeline = build_preprocessing_pipeline()
        processed_values = preprocessing_pipeline.fit_transform(numeric_dataframe)

        kmeans = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        )

        cluster_labels = kmeans.fit_predict(processed_values)

        result_dataframe = dataframe.copy()
        result_dataframe["cluster"] = cluster_labels

        st.success("Clustering wurde erfolgreich berechnet.")

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

        metric_col_1.metric("Datensätze", len(result_dataframe))
        metric_col_2.metric("Cluster", cluster_count)

        if cluster_count > 1 and len(set(cluster_labels)) > 1:
            score = silhouette_score(processed_values, cluster_labels)
            metric_col_3.metric("Silhouette Score", round(score, 3))
        else:
            metric_col_3.metric("Silhouette Score", "nicht verfügbar")

        st.write("Cluster-Größen")
        cluster_sizes = (
            result_dataframe["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Anzahl Datensätze")
        )
        st.dataframe(cluster_sizes, use_container_width=True)

        st.write("Durchschnittswerte je Cluster")
        cluster_profile = (
            result_dataframe
            .groupby("cluster")[numeric_dataframe.columns]
            .mean()
            .round(3)
            .reset_index()
        )
        st.dataframe(cluster_profile, use_container_width=True)

        st.write("Daten mit Cluster-Zuordnung")
        st.dataframe(result_dataframe.head(100), use_container_width=True)

        csv_bytes = result_dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Clustering-Ergebnis als CSV herunterladen",
            data=csv_bytes,
            file_name="clustering_ergebnis.csv",
            mime="text/csv",
        )

        render_cluster_plot(
            processed_values=processed_values,
            cluster_labels=cluster_labels,
        )

    except Exception as error:
        st.error(f"Clustering konnte nicht berechnet werden: {error}")


def render_cluster_plot(
    processed_values: np.ndarray,
    cluster_labels: np.ndarray,
):
    """
    Visualisiert Cluster in 2D.

    Wenn mehr als zwei Spalten genutzt wurden, reduziert PCA die Daten auf zwei Dimensionen.
    Dadurch kann man Cluster grafisch darstellen.
    """

    st.write("Cluster-Visualisierung")

    if processed_values.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        plot_values = pca.fit_transform(processed_values)

        explained_variance = pca.explained_variance_ratio_.sum()
        st.caption(
            f"PCA wurde genutzt. Erklärte Varianz der 2D-Darstellung: "
            f"{explained_variance:.2%}"
        )
    else:
        plot_values = processed_values

    fig, ax = plt.subplots(figsize=(8, 5))

    scatter = ax.scatter(
        plot_values[:, 0],
        plot_values[:, 1],
        c=cluster_labels,
        alpha=0.75,
    )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Cluster-Visualisierung")

    legend = ax.legend(
        *scatter.legend_elements(),
        title="Cluster",
        loc="best",
    )
    ax.add_artist(legend)

    st.pyplot(fig)


def render_anomaly_detection(
    dataframe: pd.DataFrame,
    source_name: str = "Datenquelle",
):
    """
    Rendert eine Anomalie-Erkennung in Streamlit.

    Die Methode nutzt Isolation Forest.
    Isolation Forest erkennt Datensätze, die im Vergleich zum Rest ungewöhnlich sind.
    """

    st.subheader("Anomalie-Erkennung")
    st.caption(f"Datenquelle: {source_name}")

    numeric_columns = get_numeric_columns(dataframe)

    if len(numeric_columns) < 1:
        st.warning(
            "Für die Anomalie-Erkennung wird mindestens eine numerische Spalte benötigt."
        )
        return

    selected_columns = st.multiselect(
        "Numerische Spalten für Anomalie-Erkennung auswählen",
        options=numeric_columns,
        default=numeric_columns[: min(5, len(numeric_columns))],
        key="anomaly_selected_columns",
    )

    if len(selected_columns) < 1:
        st.info("Bitte wähle mindestens eine numerische Spalte aus.")
        return

    contamination = st.slider(
        "Erwarteter Anteil an Anomalien",
        min_value=0.01,
        max_value=0.30,
        value=0.05,
        step=0.01,
        key="anomaly_contamination",
    )

    run_button = st.button(
        "Anomalien erkennen",
        key="run_anomaly_button",
    )

    if not run_button:
        return

    numeric_dataframe = clean_numeric_dataframe(
        dataframe=dataframe,
        selected_columns=selected_columns,
    )

    if numeric_dataframe.empty:
        st.error("Nach der Bereinigung sind keine nutzbaren numerischen Daten übrig.")
        return

    try:
        preprocessing_pipeline = build_preprocessing_pipeline()
        processed_values = preprocessing_pipeline.fit_transform(numeric_dataframe)

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
        )

        predictions = model.fit_predict(processed_values)

        anomaly_scores = model.decision_function(processed_values)

        result_dataframe = dataframe.copy()

        result_dataframe["anomaly_prediction"] = predictions
        result_dataframe["is_anomaly"] = result_dataframe["anomaly_prediction"] == -1
        result_dataframe["anomaly_score"] = anomaly_scores

        anomalies = result_dataframe[result_dataframe["is_anomaly"]].copy()

        st.success("Anomalie-Erkennung wurde erfolgreich berechnet.")

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)

        metric_col_1.metric("Datensätze", len(result_dataframe))
        metric_col_2.metric("Gefundene Anomalien", len(anomalies))
        metric_col_3.metric(
            "Anomalie-Anteil",
            f"{len(anomalies) / len(result_dataframe):.2%}",
        )

        st.write("Auffällige Datensätze")
        st.dataframe(
            anomalies.sort_values("anomaly_score").head(100),
            use_container_width=True,
        )

        st.write("Alle Daten mit Anomalie-Markierung")
        st.dataframe(result_dataframe.head(100), use_container_width=True)

        csv_bytes = result_dataframe.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Anomalie-Ergebnis als CSV herunterladen",
            data=csv_bytes,
            file_name="anomalie_ergebnis.csv",
            mime="text/csv",
        )

        render_anomaly_plot(
            processed_values=processed_values,
            is_anomaly=result_dataframe["is_anomaly"].to_numpy(),
        )

    except Exception as error:
        st.error(f"Anomalie-Erkennung konnte nicht berechnet werden: {error}")


def render_anomaly_plot(
    processed_values: np.ndarray,
    is_anomaly: np.ndarray,
):
    """
    Visualisiert normale und auffällige Datensätze in 2D.

    Wenn mehr als zwei Spalten genutzt wurden, reduziert PCA die Daten auf zwei Dimensionen.
    """

    st.write("Anomalie-Visualisierung")

    if processed_values.shape[1] > 2:
        pca = PCA(n_components=2, random_state=42)
        plot_values = pca.fit_transform(processed_values)

        explained_variance = pca.explained_variance_ratio_.sum()
        st.caption(
            f"PCA wurde genutzt. Erklärte Varianz der 2D-Darstellung: "
            f"{explained_variance:.2%}"
        )
    else:
        plot_values = processed_values

    fig, ax = plt.subplots(figsize=(8, 5))

    normal_values = plot_values[~is_anomaly]
    anomaly_values = plot_values[is_anomaly]

    ax.scatter(
        normal_values[:, 0],
        normal_values[:, 1],
        alpha=0.5,
        label="Normal",
    )

    ax.scatter(
        anomaly_values[:, 0],
        anomaly_values[:, 1],
        alpha=0.9,
        label="Anomalie",
        marker="x",
    )

    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Anomalie-Visualisierung")
    ax.legend()

    st.pyplot(fig)


def render_ml_analysis(
    dataframe: pd.DataFrame,
    source_name: str = "Datenquelle",
):
    """
    Hauptfunktion für deine main-Datei.

    Diese Funktion kannst du per Button aus deiner Streamlit-App aufrufen.
    Sie zeigt dem Nutzer eine Auswahl zwischen:
    - Clustering
    - Anomalie-Erkennung
    """

    st.header("ML-basierte Analysen")

    if dataframe is None or dataframe.empty:
        st.warning("Es wurden keine Daten für die ML-Analyse übergeben.")
        return

    st.caption(f"Aktuelle Datenquelle: {source_name}")

    analysis_type = st.radio(
        "Welche ML-Analyse möchtest du ausführen?",
        options=[
            "Clustering",
            "Anomalie-Erkennung",
        ],
        horizontal=True,
        key="ml_analysis_type",
    )

    if analysis_type == "Clustering":
        render_clustering_analysis(
            dataframe=dataframe,
            source_name=source_name,
        )

    elif analysis_type == "Anomalie-Erkennung":
        render_anomaly_detection(
            dataframe=dataframe,
            source_name=source_name,
        )