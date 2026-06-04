"""
1. Clustering
   - findet Gruppen in Daten ohne Zielvariable
   - geeignet für Kundensegmente, Produktgruppen, Mustererkennung
2. Anomalie-Erkennung
   - findet ungewöhnliche Datensätze / Ausreißer
   - geeignet für Betrug, Fehler, Extremwerte
"""
import numpy as np #Zur hilfe bei Zahlenwerten, z.B. unendliche Werte oder NaN
import pandas as pd #um tabellen in Datenframes zu verarbeiten
import streamlit as st # Öberflächer UI
import matplotlib.pyplot as plt #für die Visualisierung der ML Analysen

from sklearn.cluster import KMeans#für das Clustering
from sklearn.ensemble import IsolationForest#für die Anomalie-Erkennung
from sklearn.impute import SimpleImputer#um fehlende Werte zu ersetzen
from sklearn.metrics import silhouette_score#um die Qualität der Cluster zu bewerten
from sklearn.pipeline import Pipeline# um die Schritte der Datenvorbereitung und Modellierung zu verbinden
from sklearn.preprocessing import StandardScaler#um die Daten zu skalieren, damit alle Spalten gleich wichtig sind
from sklearn.decomposition import PCA#um die Daten für die Visualisierung auf 2D zu reduzieren, wenn mehr als 2 Spalten genutzt werden

#prüft welche Spalten im DataFrame Zahlenwerte enthalten, da ML-Analysen nur mit numerischen Daten funktionieren
def get_numeric_columns(dataframe: pd.DataFrame) -> list[str]:
    return dataframe.select_dtypes(include=["number"]).columns.tolist()

#    Erstellt ein bereinigtes numerisches DataFrame.
def clean_numeric_dataframe(
    dataframe: pd.DataFrame,
    selected_columns: list[str],
) -> pd.DataFrame:
    
    numeric_dataframe = dataframe[selected_columns].copy()# erstellt eine Kopie des DataFrames mit nur den ausgewählten numerischen Spalten

    numeric_dataframe = numeric_dataframe.replace([np.inf, -np.inf], np.nan)#ersetzt unendliche Werte durch NaN, da unendliche Werte die ML-Modelle stören können

    empty_columns = [# überprüft, ob eine Spalte komplett leer ist
        column for column in numeric_dataframe.columns
        if numeric_dataframe[column].isna().all()
    ]

    if empty_columns:# wenn Collum empty ist, wird es gelöscht, da es für die Analyse nicht nutzbar ist
        numeric_dataframe = numeric_dataframe.drop(columns=empty_columns)

    return numeric_dataframe#gibt den bereinigten DataFrame zurück

#Baut eine Preprocessing-Pipeline für ML.
def build_preprocessing_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),#SimpleImputer:ersetzt fehlende Werte durch den Median der jeweiligen Spalte.
            ("scaler", StandardScaler()),#StandardScaler: skaliert die Daten, damit alle Spalten den gleichen Einfluss auf die ML-Modelle haben, unabhängig von ihren ursprünglichen Wertebereichen.
        ]
    )

#
#K-Means gruppiert ähnliche Datensätze anhand numerischer Spalten.
def render_clustering_analysis(#zeigt die Clustering-Analyse in Streamlit an
    dataframe: pd.DataFrame,
    source_name: str = "Datenquelle",
):
    st.subheader("Clustering")#Überschrift für den Clustering-Abschnitt
    st.caption(f"Datenquelle: {source_name}")#was analysiert wird, damit der Nutzer den Kontext hat

    numeric_columns = get_numeric_columns(dataframe)#prüft, welche Spalten im DataFrame Zahlenwerte enthalten, da ML-Analysen nur mit numerischen Daten funktionieren

    if len(numeric_columns) < 2:#wenn es weniger als 2 numerische Spalten gibt, kann kein Clustering durchgeführt werden, da Clustering die Ähnlichkeit zwischen Datensätzen basierend auf mehreren Merkmalen erfordert.
        st.warning(
            "Für Clustering werden mindestens zwei numerische Spalten benötigt."#ausgabe das es zu wenig spalten gibt 
        )
        return

    selected_columns = st.multiselect(#erstellt eine Mehrfachauswahl für die numerischen Spalten, die in das Clustering einbezogen werden sollen,damit mehr analysiert werden können. Mehr abhängigkeiten
        "Numerische Spalten für Clustering auswählen",
        options=numeric_columns,#die Optionen für die Auswahl sind die numerischen Spalten im DataFrame
        default=numeric_columns[: min(3, len(numeric_columns))],#standardmäßig werden die ersten 3 numerischen Spalten ausgewählt, oder alle, wenn es weniger als 3 gibt
        key="clustering_selected_columns",#ein eindeutiger Schlüssel für die Streamlit-Komponente, damit der Zustand der Auswahl gespeichert wird
    )

    if len(selected_columns) < 2:#wenn der Nutzer weniger als 2 Spalten auswählt, wird eine Information ausgegeben, dass mindestens 2 Spalten ausgewählt werden müssen, da Clustering die Ähnlichkeit zwischen Datensätzen basierend auf mehreren Merkmalen erfordert.
        st.info("Bitte wähle mindestens zwei numerische Spalten aus.")
        return

    cluster_count = st.slider(#erstellt einen Schieberegler für die Anzahl der Cluster, die K-Means bilden soll, damit der Nutzer die Anzahl der Cluster anpassen kann
        "Anzahl Cluster",
        min_value=2,
        max_value=12,
        value=3,#standardauswahl ist 3 
        step=1,
        key="cluster_count",#ein eindeutiger Schlüssel für die Streamlit-Komponente, damit der Zustand des Schiebereglers gespeichert wird
    )

    run_button = st.button(#erstellt einen Button, um das Clustering zu starten, damit der Nutzer die Analyse manuell auslösen kann, nachdem er die gewünschten Einstellungen vorgenommen hat
        "Clustering berechnen",
        key="run_clustering_button",#ein eindeutiger Schlüssel für die Streamlit-Komponente, damit der Zustand des Buttons gespeichert wird
    )

    if not run_button:#wenn der Button nicht gedrückt wurde, wird die Funktion verlassen, damit die Analyse erst gestartet wird, wenn der Nutzer bereit ist
        return

    numeric_dataframe = clean_numeric_dataframe(#erstellt ein bereinigtes numerisches DataFrame basierend auf den ausgewählten Spalten, damit die ML-Modelle mit sauberen Daten arbeiten können
        dataframe=dataframe,#das ursprüngliche DataFrame wird übergeben, damit die Bereinigung auf den richtigen Daten durchgeführt wird
        selected_columns=selected_columns,#die ausgewählten Spalten werden übergeben, damit nur diese Spalten bereinigt und für die Analyse verwendet werden
    )

    if numeric_dataframe.empty:#wenn nach der Bereinigung keine nutzbaren numerischen Daten übrig sind, wird eine Fehlermeldung ausgegeben, damit der Nutzer weiß, dass die Analyse
        st.error("Nach der Bereinigung sind keine nutzbaren numerischen Daten übrig.")
        return

    if len(numeric_dataframe) < cluster_count:#wenn die Anzahl der Datensätze kleiner als die Anzahl der Cluster ist, wird eine Fehlermeldung ausgegeben, damit der Nutzer weiß, dass es nicht genug Daten gibt, um die gewünschte Anzahl von Clustern zu bilden
        st.error(
            "Die Anzahl der Datensätze muss größer oder gleich der Anzahl der Cluster sein."
        )
        return

    try:#versucht, die Clustering-Analyse durchzuführen, und fängt Fehler ab, um eine Fehlermeldung an den Nutzer auszugeben, falls etwas schiefgeht
        preprocessing_pipeline = build_preprocessing_pipeline()#baut eine Preprocessing-Pipeline für ML, damit die Daten vor der Analyse bereinigt und skaliert werden
        processed_values = preprocessing_pipeline.fit_transform(numeric_dataframe)#
        #preprcessing istt eine Pipeline, die fehlende Werte ersetzt und die Daten skaliert. fit_transform wendet diese Schritte auf die numerischen Daten an und gibt die verarbeiteten Werte zurück, die für das Clustering verwendet werden können
        kmeans = KMeans(#erstellt ein KMeans-Modell mit der angegebenen Anzahl von Clustern, einem festen Zufallszustand für Reproduzierbarkeit und einer bestimmten Anzahl von Initialisierungen, damit das Modell stabilere Ergebnisse liefert
            n_clusters=cluster_count,
            random_state=42,#42 ist die zahl für alles was mit ML zu tun hat, damit die Ergebnisse reproduzierbar sind
            n_init=10,#die Anzahl der Initialisierungen gibt an, wie oft K-Means mit unterschiedlichen Startpunkten ausgeführt wird, um die beste Lösung zu finden. Standardmäßig ist es 10
        )

        cluster_labels = kmeans.fit_predict(processed_values)#fitbredict führt das K-Means-Clustering durch und gibt die Cluster-Labels für jeden Datensatz zurück, damit man weiß, zu welchem Cluster jeder Datensatz gehört

        result_dataframe = dataframe.copy()#erstellt eine Kopie des ursprünglichen DataFrames, damit die Cluster-Labels hinzugefügt werden können, ohne die Originaldaten zu verändern
        result_dataframe["cluster"] = cluster_labels#fügt eine neue Spalte "cluster" zum resultierenden DataFrame hinzu, die die Cluster-Labels enthält

        st.success("Clustering wurde erfolgreich berechnet.")#ausgabe falls es geklappt hat 

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)#erstellt drei Spalten für die Metriken, damit die wichtigsten Informationen übersichtlich dargestellt werden können

        metric_col_1.metric("Datensätze", len(result_dataframe))#zeigt die Anzahl der Datensätze im resultierenden DataFrame an, damit der Nutzer weiß, wie viele Datensätze in die Analyse einbezogen wurden
        metric_col_2.metric("Cluster", cluster_count)#zeigt die Anzahl der Cluster an, die gebildet wurden, damit der Nutzer die gewählte Anzahl von Clustern sehen kann

        if cluster_count > 1 and len(set(cluster_labels)) > 1: #überprüft, ob es mehr als einen Cluster gibt und ob die Cluster-Labels tatsächlich mehr als eine Gruppe enthalten, damit die Silhouette Score nur berechnet wird, wenn es sinnvoll ist
            score = silhouette_score(processed_values, cluster_labels)#der silhoutte_score bewertet die Qualität der Cluster, indem er misst, wie ähnlich ein Datensatz zu seinem eigenen Cluster im Vergleich zu anderen Clustern ist. Ein höherer Wert bedeutet bessere Cluster
            metric_col_3.metric("Silhouette Score", round(score, 3))
        else:
            metric_col_3.metric("Silhouette Score", "nicht verfügbar")

        st.write("Cluster-Größen") #zeigt die Anzahl der Datensätze in jedem Cluster an, damit der Nutzer sehen kann, wie die Daten auf die Cluster verteilt sind
        cluster_sizes = (
            result_dataframe["cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")#gibt der Serie einen Namen "Cluster", damit die Cluster-Labels als Index angezeigt werden
            .reset_index(name="Anzahl Datensätze")
        )
        st.dataframe(cluster_sizes, use_container_width=True)#zeigt die Cluster-Größen in einem DataFrame an, damit der Nutzer die Verteilung der Datensätze auf die Cluster sehen kann

        st.write("Durchschnittswerte je Cluster")
        cluster_profile = (
            result_dataframe
            .groupby("cluster")[numeric_dataframe.columns]#gruppiert die Daten nach Cluster und berechnet den Durchschnitt für die ausgewählten numerischen Spalten
            .mean()
            .round(3)#berechnet den Durchschnitt für jede numerische Spalte in jedem Cluster und rundet die Werte auf 3 Dezimalstellen, damit die Ergebnisse übersichtlich dargestellt werden können
            .reset_index()#setzt den Cluster-Label als normale Spalte zurück, damit sie im DataFrame angezeigt werden kann
        )
        st.dataframe(cluster_profile, use_container_width=True)#zeigt die Durchschnittswerte je Cluster in einem DataFrame an, damit der Nutzer die charakteristischen Merkmale jedes Clusters sehen kann

        st.write("Daten mit Cluster-Zuordnung")#ausgabe
        st.dataframe(result_dataframe.head(100), use_container_width=True)#zeigt die ersten 100 Zeilen des resultierenden DataFrames mit der Cluster-Zuordnung an

        csv_bytes = result_dataframe.to_csv(index=False).encode("utf-8")#konvertiert den resultierenden DataFrame in eine CSV-Datei im Speicher, damit der Nutzer die Ergebnisse herunterladen kann. index=False bedeutet, dass der DataFrame-Index nicht in die CSV-Datei aufgenommen wird, und encode("utf-8") stellt sicher, dass die Datei im UTF-8-Format kodiert ist.

        st.download_button(#um die ergebnisse als csv herunterzuladen
            label="Clustering-Ergebnis als CSV herunterladen",
            data=csv_bytes,
            file_name="clustering_ergebnis.csv",
            mime="text/csv",
        )

        render_cluster_plot(#visualisiert die Cluster in 2D, damit der Nutzer die Cluster grafisch sehen kann
            processed_values=processed_values,
            cluster_labels=cluster_labels,
        )

    except Exception as error:#ausgabe falls es nicht geklappt hat, damit der Nutzer weiß, dass ein Fehler aufgetreten ist und was das Problem war
        st.error(f"Clustering konnte nicht berechnet werden: {error}")


def render_cluster_plot(#Visualisiert Cluster in 2D, damit der Nutzer die Cluster grafisch sehen kann
    processed_values: np.ndarray,
    cluster_labels: np.ndarray,
):
    st.write("Cluster-Visualisierung")
    if processed_values.shape[1] > 2:#wenn mehr als zwei Spalten genutzt wurden, reduziert PCA die Daten auf zwei Dimensionen, damit sie in einem 2D-Plot dargestellt werden können
        pca = PCA(n_components=2, random_state=42)#n_components=2 bedeutet, dass die Daten auf 2 Dimensionen reduziert werden, damit sie in einem 2D-Plot dargestellt werden können. random_state=42 sorgt für Reproduzierbarkeit der Ergebnisse
        plot_values = pca.fit_transform(processed_values)#fit_transform wendet PCA auf die verarbeiteten Werte an und gibt die reduzierten Werte zurück, die für die Visualisierung verwendet werden können

        explained_variance = pca.explained_variance_ratio_.sum()#pca.explained_variance_ratio_ gibt an, wie viel der ursprünglichen Varianz in den Daten durch die reduzierten Dimensionen erklärt wird. sum() berechnet die Gesamtvarianz, die durch die 2D-Darstellung erklärt wird, damit der Nutzer sehen kann, wie gut die Visualisierung die ursprünglichen Daten repräsentiert
        st.caption(
            f"PCA wurde genutzt. Erklärte Varianz der 2D-Darstellung: "
            f"{explained_variance:.2%}"#zeigt die erklärte Varianz in Prozent an, damit der Nutzer eine Vorstellung davon hat, wie viel der ursprünglichen Daten durch die 2D-Darstellung repräsentiert wird
        )
    else:
        plot_values = processed_values#wenn nur zwei Spalten genutzt wurden, können diese direkt für die Visualisierung verwendet werden, damit die Daten nicht unnötig reduziert werden und die volle Information erhalten bleibt

    fig, ax = plt.subplots(figsize=(8, 5))#plt.subplots erstellt eine neue Figur und Achse für die Visualisierung, damit die Cluster in einem separaten Plot dargestellt werden können. figsize=(8, 5) legt die Größe des Plots fest

    scatter = ax.scatter(#ax.scatter erstellt einen Streudiagramm-Plot, um die Cluster zu visualisieren. plot_values[:, 0] und plot_values[:, 1] geben die x- und y-Koordinaten der Punkte an, c=cluster_labels färbt die Punkte basierend auf ihren Cluster-Labels, und alpha=0.75 macht die Punkte leicht transparent, damit überlappende Punkte besser sichtbar sind
        plot_values[:, 0],#die Werte der ersten Dimension für die x-Achse
        plot_values[:, 1],#die Werte der zweiten Dimension für die y-Achse
        c=cluster_labels,#die Cluster-Labels bestimmen die Farbe der Punkte, damit die verschiedenen Cluster visuell unterschieden werden können
        alpha=0.75,#die Transparenz der Punkte wird auf 0.75 gesetzt, damit überlappende Punkte besser sichtbar sind und die Visualisierung klarer wird
    )

    ax.set_xlabel("Dimension 1")#beschriftet die x-Achse als "Dimension 1"
    ax.set_ylabel("Dimension 2")#beschriftet die y-Achse als "Dimension 2"
    ax.set_title("Cluster-Visualisierung")#Überschrift für den Plot

    legend = ax.legend(#ax.legend erstellt eine Legende für den Plot, damit der Nutzer sehen kann, welche Farben zu welchen Clustern gehören. 
        *scatter.legend_elements(),#scatter.legend_elements() generiert die Legenden-Handles und -Labels basierend auf den Farben der Punkte im Scatter-Plot, damit die Legende automatisch die richtigen Farben und Cluster-Labels anzeigt
        title="Cluster",
        loc="best",#die Legende wird an der besten Position im Plot platziert, damit sie nicht wichtige Informationen überdeckt und gut lesbar ist
    )
    ax.add_artist(legend)#fügt die Legende zum Plot hinzu, damit sie angezeigt wird

    st.pyplot(fig)#st.pyplot(fig) zeigt die erstellte Figur in Streamlit an, damit der Nutzer die Cluster-Visualisierung sehen kann


def render_anomaly_detection(#IsolationForrest erkennt Datensätze, die im Vergleich zum Rest ungewöhnlich sind, damit der Nutzer auffällige Datensätze identifizieren kann
    dataframe: pd.DataFrame,
    source_name: str = "Datenquelle",
):
    st.subheader("Anomalie-Erkennung")
    st.caption(f"Datenquelle: {source_name}")

    numeric_columns = get_numeric_columns(dataframe)#prüft, welche Spalten im DataFrame Zahlenwerte enthalten, da ML-Analysen nur mit numerischen Daten funktionieren. 
    if len(numeric_columns) < 1:#muss mindestens eine spalte haben
        st.warning(
            "Für die Anomalie-Erkennung wird mindestens eine numerische Spalte benötigt."#ansosnten warnung 
        )
        return

    selected_columns = st.multiselect(#auswahl der spalten für die anomalieerkennung 
        "Numerische Spalten für Anomalie-Erkennung auswählen",
        options=numeric_columns,
        default=numeric_columns[: min(3, len(numeric_columns))],
        key="anomaly_selected_columns",
    )

    if len(selected_columns) < 1:
        st.info("Bitte wähle mindestens eine numerische Spalte aus.")# falls man keine ausgewählt hat 
        return

    contamination = st.slider(#contamination gibt an, welcher Anteil der Daten als Anomalien betrachtet werden soll. 
        "Erwarteter Anteil an Anomalien",#Es ist ein wichtiger Parameter für die IsolationForest-Methode, da er dem Modell hilft zu bestimmen, wie viele Datensätze als Anomalien markiert werden sollen. Ein zu hoher Wert könnte zu vielen falsch positiven Anomalien führen, während ein zu niedriger Wert möglicherweise echte Anomalien übersieht.
        min_value=0.01,#mindestens 1%
        max_value=0.5,#maximal 50%
        value=0.06,
        step=0.02,
        key="anomaly_contamination",
    )

    run_button = st.button(#um die Anomalieerkennung zu starten
        "Anomalien erkennen",
        key="run_anomaly_button",#ein eindeutiger Schlüssel für die Streamlit-Komponente, damit der Zustand des Buttons gespeichert wird
    )

    if not run_button:#wenn der Button nicht gedrückt wurde, wird die Funktion verlassen, damit die Analyse erst gestartet wird, wenn der Nutzer bereit ist
        return

    numeric_dataframe = clean_numeric_dataframe(#clean_numeric_dataframe erstellt ein bereinigtes numerisches DataFrame basierend auf den ausgewählten Spalten, damit die ML-Modelle mit sauberen Daten arbeiten können
        dataframe=dataframe,
        selected_columns=selected_columns,
    )

    if numeric_dataframe.empty:
        st.error("Nach der Bereinigung sind keine nutzbaren numerischen Daten übrig.")#fehlermeldung wenn keine Daten zur analyse übrig bleiben
        return

    try:#versucht die Anomalie-Erkennung durchzuführen
        preprocessing_pipeline = build_preprocessing_pipeline()#build_preprocessing_pipeline baut eine Preprocessing-Pipeline für ML, damit die Daten vor der Analyse bereinigt und skaliert werden
        processed_values = preprocessing_pipeline.fit_transform(numeric_dataframe)#fit_transform wendet die Preprocessing-Pipeline auf die numerischen Daten an und gibt die verarbeiteten Werte zurück, die für die Anomalie-Erkennung verwendet werden können

        model = IsolationForest(#IsolationForest ist ein Algorithmus zur Anomalie-Erkennung, der ungewöhnliche Datensätze identifiziert, indem er sie isoliert. Er funktioniert gut bei hochdimensionalen Daten und benötigt keine Annahmen über die Verteilung der Daten.
            contamination=contamination,#contamination gibt an, welcher Anteil der Daten als Anomalien betrachtet werden soll. Ein zu hoher Wert könnte zu vielen falsch positiven Anomalien führen, während ein zu niedriger Wert möglicherweise echte Anomalien übersieht.
            random_state=42,#42 ist die zahl für alles was mit ML zu tun hat, damit die Ergebnisse reproduzierbar sind
        )

        predictions = model.fit_predict(processed_values)#fit_predict führt die Anomalie-Erkennung durch und gibt für jeden Datensatz eine Vorhersage zurück, wobei -1 für Anomalien und 1 für normale Datensätze

        anomaly_scores = model.decision_function(processed_values)#model.decision_function gibt für jeden Datensatz einen Anomalie-Score zurück. Ein negativer Score deutet auf eine Anomalie hin, während ein positiver Score auf einen normalen Datensatz hindeutet.

        result_dataframe = dataframe.copy()#erstellt eine Kopie des ursprünglichen DataFrames, damit die Anomalie-Vorhersagen und -Scores hinzugefügt werden können, ohne die Originaldaten zu verändern

        result_dataframe["anomaly_prediction"] = predictions#fügt eine neue Spalte "anomaly_prediction" zum resultierenden DataFrame hinzu, die die Vorhersagen der Anomalie-Erkennung enthält
        result_dataframe["is_anomaly"] = result_dataframe["anomaly_prediction"] == -1#fügt eine neue Spalte "is_anomaly" hinzu, die auf True gesetzt wird, wenn die Vorhersage -1 (Anomalie) ist, und auf False, wenn die Vorhersage 1 (normaler Datensatz) ist.
        result_dataframe["anomaly_score"] = anomaly_scores#fügt eine neue Spalte "anomaly_score" hinzu, die die Anomalie-Scores enthält
        #der anomaly_score gibt an, wie stark ein Datensatz als Anomalie betrachtet wird. Ein negativer Score deutet auf eine Anomalie hin, während ein positiver Score auf einen normalen Datensatz hindeutet. Je niedriger der Score, desto stärker ist die Anomalie.

        anomalies = result_dataframe[result_dataframe["is_anomaly"]].copy()#erstellt einen neuen DataFrame "anomalies", der nur die Datensätze enthält, die als Anomalien identifiziert wurden, damit der Nutzer diese auffälligen Datensätze separat betrachten kann

        st.success("Anomalie-Erkennung wurde erfolgreich berechnet.")#wenn es geklappt hat

        st.write("Alle Daten mit Anomalie-Markierung")
        st.dataframe(result_dataframe.head(100), use_container_width=True)#zeigt die ersten 100 Zeilen des resultierenden DataFrames mit der Anomalie-Markierung an, damit der Nutzer die Ergebnisse der Anomalie-Erkennung sehen kann

        csv_bytes = result_dataframe.to_csv(index=False).encode("utf-8")#konvertiert den resultierenden DataFrame in eine CSV-Datei im Speicher, damit der Nutzer die Ergebnisse herunterladen kann. index=False bedeutet, dass der DataFrame-Index nicht in die CSV-Datei aufgenommen wird, und encode("utf-8") stellt sicher, dass die Datei im UTF-8-Format kodiert ist.

        st.download_button(#zum herunterladen der ergebnisse als csv
            label="Anomalie-Ergebnis als CSV herunterladen",
            data=csv_bytes,
            file_name="anomalie_ergebnis.csv",
            mime="text/csv",
        )

        render_anomaly_plot(#visualisiert normale und auffällige Datensätze in 2D, damit der Nutzer die Anomalien grafisch sehen kann
            processed_values=processed_values,
            is_anomaly=result_dataframe["is_anomaly"].to_numpy(),
        )

    except Exception as error:
        st.error(f"Anomalie-Erkennung konnte nicht berechnet werden: {error}")#fehlerausgabe wenn es nicht geklappt hat


def render_anomaly_plot(#render_anomaly_plot visualisiert normale und auffällige Datensätze in 2D, damit der Nutzer die Anomalien grafisch sehen kann. Wenn mehr als zwei Spalten genutzt wurden, reduziert PCA die Daten auf zwei Dimensionen, damit sie in einem 2D-Plot dargestellt werden können.
    processed_values: np.ndarray,#processed_values sind die verarbeiteten Werte der numerischen Daten, die für die Anomalie-Erkennung verwendet wurden. Sie werden für die Visualisierung genutzt, damit der Nutzer die Anomalien in Bezug auf die ursprünglichen Daten sehen kann.
    is_anomaly: np.ndarray,#is_anomaly ist ein Array, das angibt, welche Datensätze als Anomalien identifiziert wurden (True für Anomalien und False für normale Datensätze). Es wird verwendet, um die Punkte im Plot entsprechend zu färben, damit der Nutzer die Anomalien visuell unterscheiden kann.
):
    st.write("Anomalie-Visualisierung")

    if processed_values.shape[1] > 2:#wenn mehr als zwei Spalten genutzt wurden, reduziert PCA die Daten auf zwei Dimensionen, damit sie in einem 2D-Plot dargestellt werden können. PCA (Principal Component Analysis) ist eine Technik zur Dimensionsreduktion, die die wichtigsten Informationen in den Daten beibehält, während sie die Anzahl der Dimensionen reduziert.
        pca = PCA(n_components=2, random_state=42)#n_components=2 bedeutet, dass die Daten auf 2 Dimensionen reduziert werden, damit sie in einem 2D-Plot dargestellt werden können. random_state=42 sorgt für Reproduzierbarkeit der Ergebnisse
        plot_values = pca.fit_transform(processed_values)#pca.fit_transform wendet PCA auf die verarbeiteten Werte an und gibt die reduzierten Werte zurück, die für die Visualisierung verwendet werden können

        explained_variance = pca.explained_variance_ratio_.sum()#pca.explained_variance_ratio_ gibt an, wie viel der ursprünglichen Varianz in den Daten durch die reduzierten Dimensionen erklärt wird. sum() berechnet die Gesamtvarianz, die durch die 2D-Darstellung erklärt wird, damit der Nutzer sehen kann, wie gut die Visualisierung die ursprünglichen Daten repräsentiert
        st.caption(
            f"Erklärte Varianz als 2D-Darstellung: "
            f"{explained_variance:.2%}"
        )
    else:
        plot_values = processed_values#zwei spalten werden sofort visualisiert

    fig, ax = plt.subplots(figsize=(8, 5))#plt.subplots erstellt eine neue Figur und Achse für die Visualisierung, damit die Anomalien in einem separaten Plot dargestellt werden können. figsize=(8, 5) legt die Größe des Plots fest

    normal_values = plot_values[~is_anomaly]#normal_values enthält die Werte der Datensätze, die als normal (nicht anomal) identifiziert wurden
    anomaly_values = plot_values[is_anomaly]#anomaly_values enthält die Werte der Datensätze, die als Anomalien identifiziert wurden

    ax.scatter(#ax.scatter erstellt ein Streudiagramm für die normalen
        normal_values[:, 0],#X-Koordinaten der normalen Datensätze
        normal_values[:, 1],#Y_Koordinaten der normalen Datensätze
        alpha=0.5,#die Transparenz der Punkte wird auf 0.5 gesetzt, damit überlappende Punkte besser sichtbar sind und die Visualisierung klarer wird
        label="Normal",#die normalen Datensätze werden mit dem Label "Normal" in der Legende angezeigt
    )

    ax.scatter(#ax.scatter erstellt ein Streudiagramm für die Anomalien
        anomaly_values[:, 0],#X-Koordinaten der Anomalien
        anomaly_values[:, 1],#Y-Koordinaten der Anomalien
        alpha=0.9,  #Transparenz
        label="Anomalie",
        marker="x",#die anomalie werden mit einem x dargestellt 
    )

    #Beschriftungen und Legende für den Plot
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title("Anomalie-Visualisierung")
    ax.legend()

    st.pyplot(fig)#darstellung der Visualisierung in Streamlit


def render_ml_analysis(#hautpfunktion für die ML-Analysen, die in der Streamlit-App aufgerufen wird. Sie zeigt dem Nutzer eine Auswahl zwischen Clustering und Anomalie
    dataframe: pd.DataFrame, #das DataFrame, das für die ML-Analysen verwendet werden soll
    source_name: str = "Datenquelle",
):
    st.header("ML-basierte Analysen")

    if dataframe is None or dataframe.empty:
        st.warning("Es wurden keine Daten für die ML-Analyse übergeben.")#wenn es kein dataframe gibt 
        return

    st.caption(f"Aktuelle Datenquelle: {source_name}") #zeigt den namen der aktuellen Quelle

    analysis_type = st.radio( #Auswahl welche ML-Methode man auswählen möchte 
        "Welche ML-Analyse möchtest du ausführen?",
        options=[
            "Clustering",
            "Anomalie-Erkennung",
        ],
        horizontal=True,
        key="ml_analysis_type",
    )

    if analysis_type == "Clustering": #wenn Clustering ausgewählt wird dann führe das aus 
        render_clustering_analysis(
            dataframe=dataframe,
            source_name=source_name,
        )

    elif analysis_type == "Anomalie-Erkennung": #wenn Anomalie ausgewählt wurde dann führe das aus 
        render_anomaly_detection(
            dataframe=dataframe,
            source_name=source_name,
        )
