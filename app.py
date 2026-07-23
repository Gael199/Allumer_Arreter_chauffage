"""
Application de décision de chauffage — basée sur l'analyse DJU / Consommation.

Lancement :
    streamlit run app.py

Dépendances (voir requirements.txt) :
    pip install streamlit pandas numpy plotly openpyxl scikit-learn requests
"""

import io
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# CONFIGURATION GÉNÉRALE

st.set_page_config(
    page_title="Décision Chauffage — Analyse DJU",
    page_icon="🌡️",
    layout="wide",
)

COL_SITE = "Site"
COL_EQUIP = "Équipement"
COL_DATE = "Date (Jour)"
COL_CONSO = "Consommation (kWhef)"
COL_DJU = "DJU Chauds"

REQUIRED_COLS = [COL_SITE, COL_EQUIP, COL_DATE, COL_CONSO, COL_DJU]



# FONCTIONS DE CHARGEMENT ET DE NETTOYAGE

@st.cache_data(show_spinner=False)
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Charge et nettoie le fichier Excel de consommation."""
    df = pd.read_excel(io.BytesIO(file_bytes))

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes manquantes dans le fichier : {missing}. "
            f"Colonnes attendues : {REQUIRED_COLS}"
        )

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], errors="coerce")

    # Imputation par médiane groupée (Site, Équipement) — cohérente avec le
    # notebook d'analyse KMeans du projet.
    for col in [COL_CONSO, COL_DJU]:
        df[col] = df.groupby([COL_SITE, COL_EQUIP])[col].transform(
            lambda s: s.fillna(s.median())
        )
        df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=REQUIRED_COLS)
    df = df.sort_values(COL_DATE)
    return df



# MODÈLE DE RUPTURE (CHANGEPOINT) — cœur de la décision chauffage

def fit_changepoint_model(dju: np.ndarray, conso: np.ndarray, n_candidates: int = 80) -> dict:
    """
    Ajuste un modèle de régression linéaire par morceaux (continu, à un seul
    point de rupture) entre le DJU (variable explicative) et la consommation.

    Modèle :
        conso = a + b1 * dju + b2 * max(0, dju - seuil)

    Le seuil optimal est celui qui minimise la somme des carrés des résidus
    (recherche par grille sur les percentiles de DJU observés).
    """
    dju = np.asarray(dju, dtype=float)
    conso = np.asarray(conso, dtype=float)

    lo, hi = np.percentile(dju, [5, 95])
    if lo >= hi:
        lo, hi = dju.min(), dju.max()
    candidates = np.linspace(lo, hi, n_candidates)

    sst = np.sum((conso - conso.mean()) ** 2)
    best = None

    for t in candidates:
        hinge = np.maximum(0, dju - t)
        X = np.column_stack([np.ones_like(dju), dju, hinge])
        beta, _, _, _ = np.linalg.lstsq(X, conso, rcond=None)
        pred = X @ beta
        sse = np.sum((conso - pred) ** 2)
        if best is None or sse < best["sse"]:
            r2 = 1 - sse / sst if sst > 0 else 0.0
            best = {"sse": sse, "threshold": float(t), "beta": beta, "r2": float(r2)}

    intercept, slope_avant, slope_extra = best["beta"]
    best.update(
        intercept=float(intercept),
        slope_avant=float(slope_avant),
        slope_apres=float(slope_avant + slope_extra),
    )
    return best


def predict_conso(model: dict, dju_values: np.ndarray) -> np.ndarray:
    dju_values = np.asarray(dju_values, dtype=float)
    hinge = np.maximum(0, dju_values - model["threshold"])
    return model["intercept"] + model["slope_avant"] * dju_values + (
        model["beta"][2] * hinge
    )


def recommandation(dju_value: float, model: dict, marge: float = 1.0) -> str:
    seuil = model["threshold"]
    if dju_value >= seuil + marge:
        return "🔥 Chauffage recommandé"
    elif dju_value <= seuil - marge:
        return "☀️ Chauffage non nécessaire"
    else:
        return "🤔 Zone limite — à surveiller"



# GÉOCODAGE ET PRÉVISIONS MÉTÉO (Open-Meteo — gratuit, sans clé API)

@st.cache_data(show_spinner=False, ttl=3600)
def geocode_ville(nom_ville: str):
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": nom_ville, "count": 1, "language": "fr"},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        if not data.get("results"):
            return None
        res = data["results"][0]
        return {
            "lat": res["latitude"],
            "lon": res["longitude"],
            "nom": res.get("name", nom_ville),
            "pays": res.get("country", ""),
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=1800)
def previsions_meteo(lat: float, lon: float, jours: int = 7):
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max",
                "timezone": "auto",
                "forecast_days": jours,
            },
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        daily = data["daily"]
        return pd.DataFrame(
            {
                "date": pd.to_datetime(daily["time"]),
                "temp_moy": daily["temperature_2m_mean"],
                "temp_min": daily["temperature_2m_min"],
                "temp_max": daily["temperature_2m_max"],
            }
        )
    except Exception:
        return None


def dju_depuis_temperature(temp_moy: np.ndarray, base_ref: float = 18.0) -> np.ndarray:
    """Calcule le DJU Chauds à partir d'une température moyenne journalière,
    selon la même convention que la colonne 'DJU Chauds' du jeu de données
    (référence par défaut 18°C, méthode des DJU unifiés en France)."""
    return np.maximum(0, base_ref - np.asarray(temp_moy, dtype=float))



# SIDEBAR — CHARGEMENT DES DONNÉES

st.sidebar.title("🌡️ Décision Chauffage")
st.sidebar.caption(
    "Analyse du point de bascule chauffage à partir de l'historique "
    "Consommation / DJU Chauds."
)

uploaded_file = st.sidebar.file_uploader(
    "Charger le fichier de consommation (.xlsx)", type=["xlsx"]
)

if uploaded_file is None:
    st.title("🌡️ Outil de décision : quand allumer / arrêter le chauffage ?")
    st.info(
        "👈 Chargez votre fichier `donnees_regroupees.xlsx` dans la barre latérale "
        "pour commencer.\n\n"
        f"Colonnes attendues : `{'`, `'.join(REQUIRED_COLS)}`."
    )
    st.markdown(
        """
        ### Principe de l'application

        Pour chaque site et équipement, l'application ajuste un **modèle de
        régression à point de rupture** entre le DJU Chauds (indicateur du
        besoin de chauffage lié au climat) et la consommation énergétique
        réelle.

        Ce modèle détecte automatiquement le **DJU à partir duquel la
        consommation commence réellement à augmenter** pour ce site précis
        — c'est-à-dire le seuil climatique à partir duquel le chauffage doit
        être activé, déterminé empiriquement plutôt que fixé arbitrairement.

        L'application permet ensuite de :
        - visualiser ce point de bascule site par site,
        - comparer tous les sites dans un tableau récapitulatif,
        - simuler une décision "allumer / arrêter" à partir d'une valeur de
          DJU ou d'une prévision météo réelle (via Open-Meteo),
        - situer chaque site dans une segmentation KMeans de profils de
          consommation (cohérente avec le notebook d'analyse du projet).
        """
    )
    st.stop()

try:
    df = load_data(uploaded_file.getvalue())
except ValueError as e:
    st.error(str(e))
    st.stop()

sites = sorted(df[COL_SITE].unique())


# ONGLET STRUCTURE

tab_analyse, tab_comparatif, tab_meteo, tab_segmentation = st.tabs(
    ["📈 Analyse par site", "📊 Tableau comparatif", "🌦️ Décision météo", "🧩 Segmentation"]
)


# ONGLET 1 : ANALYSE PAR SITE

with tab_analyse:
    col1, col2 = st.columns(2)
    with col1:
        site_choisi = st.selectbox("Site", sites, key="site_analyse")
    with col2:
        equips_dispo = sorted(df.loc[df[COL_SITE] == site_choisi, COL_EQUIP].unique())
        equip_choisi = st.selectbox("Équipement", equips_dispo, key="equip_analyse")

    sous_df = df[(df[COL_SITE] == site_choisi) & (df[COL_EQUIP] == equip_choisi)]

    if len(sous_df) < 30:
        st.warning(
            "Moins de 30 observations disponibles pour ce couple Site/Équipement : "
            "le modèle de rupture sera peu fiable."
        )

    model = fit_changepoint_model(sous_df[COL_DJU].values, sous_df[COL_CONSO].values)
    base_ref = st.slider(
        "Température de référence utilisée pour le calcul des DJU (°C)",
        14.0, 20.0, 18.0, 0.5,
        help="Référence standard des DJU unifiés en France. À ajuster si votre "
             "fournisseur de données utilise une autre convention.",
    )
    temp_bascule = base_ref - model["threshold"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Seuil DJU de bascule", f"{model['threshold']:.1f}")
    k2.metric("Température de bascule ≈", f"{temp_bascule:.1f} °C")
    k3.metric("Pente après seuil (kWh/DJU)", f"{model['slope_apres']:.1f}")
    k4.metric("Qualité du modèle (R²)", f"{model['r2']:.2f}")

    st.caption(
        "💡 La température de bascule est une estimation indicative, obtenue en "
        "supposant que les DJU du fichier ont été calculés avec la température de "
        "référence choisie ci-dessus. Le seuil DJU lui-même reste la valeur la "
        "plus fiable, car directement estimée sur les données."
    )

    # --- Graphique de dispersion + modèle de rupture ---
    dju_range = np.linspace(sous_df[COL_DJU].min(), sous_df[COL_DJU].max(), 200)
    conso_pred = predict_conso(model, dju_range)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sous_df[COL_DJU], y=sous_df[COL_CONSO],
            mode="markers", name="Observations",
            marker=dict(size=5, opacity=0.4, color="#4C78A8"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dju_range, y=conso_pred, mode="lines",
            name="Modèle à point de rupture", line=dict(color="#F58518", width=3),
        )
    )
    fig.add_vline(
        x=model["threshold"], line_dash="dash", line_color="red",
        annotation_text="Seuil de bascule", annotation_position="top",
    )
    fig.update_layout(
        title=f"Consommation vs DJU — {site_choisi} / {equip_choisi}",
        xaxis_title="DJU Chauds",
        yaxis_title="Consommation (kWhef)",
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

    # --- Simulateur ponctuel ---
    st.subheader("Simuler une décision")
    dju_simule = st.slider(
        "Valeur de DJU (climat du jour à évaluer)",
        float(df[COL_DJU].min()), float(df[COL_DJU].max()),
        float(model["threshold"]), 0.5,
    )
    st.markdown(f"### {recommandation(dju_simule, model)}")

    # --- Série temporelle ---
    with st.expander("Voir l'historique temporel (consommation & DJU)"):
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=sous_df[COL_DATE], y=sous_df[COL_CONSO],
                       name="Consommation (kWhef)", yaxis="y1")
        )
        fig2.add_trace(
            go.Scatter(x=sous_df[COL_DATE], y=sous_df[COL_DJU],
                       name="DJU Chauds", yaxis="y2", line=dict(color="firebrick"))
        )
        fig2.update_layout(
            yaxis=dict(title="Consommation (kWhef)"),
            yaxis2=dict(title="DJU Chauds", overlaying="y", side="right"),
            height=400,
        )
        st.plotly_chart(fig2, width="stretch")


# ONGLET 2 : TABLEAU COMPARATIF DE TOUS LES SITES

with tab_comparatif:
    st.subheader("Seuils de bascule chauffage — tous sites / équipements")

    @st.cache_data(show_spinner="Calcul des seuils pour tous les sites…")
    def calculer_tous_les_seuils(df_: pd.DataFrame) -> pd.DataFrame:
        lignes = []
        for (site, equip), g in df_.groupby([COL_SITE, COL_EQUIP]):
            if len(g) < 10:
                continue
            m = fit_changepoint_model(g[COL_DJU].values, g[COL_CONSO].values)
            lignes.append(
                {
                    COL_SITE: site,
                    COL_EQUIP: equip,
                    "n_observations": len(g),
                    "seuil_dju": round(m["threshold"], 2),
                    "pente_avant": round(m["slope_avant"], 2),
                    "pente_apres": round(m["slope_apres"], 2),
                    "r2": round(m["r2"], 3),
                    "conso_moyenne": round(g[COL_CONSO].mean(), 1),
                }
            )
        return pd.DataFrame(lignes)

    tableau = calculer_tous_les_seuils(df)
    tableau = tableau.sort_values("seuil_dju")

    st.dataframe(tableau, width="stretch", hide_index=True)

    csv = tableau.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Télécharger le tableau (CSV)", csv,
        "seuils_chauffage_par_site.csv", "text/csv",
    )

    fig3 = px.bar(
        tableau.sort_values("seuil_dju"),
        x="seuil_dju", y=[f"{s} / {e}" for s, e in zip(tableau[COL_SITE], tableau[COL_EQUIP])],
        orientation="h", labels={"y": "Site / Équipement", "seuil_dju": "Seuil DJU"},
        title="Comparaison des seuils de bascule par site",
    )
    fig3.update_layout(height=max(400, 25 * len(tableau)))
    st.plotly_chart(fig3, width="stretch")


# ONGLET 3 : DÉCISION À PARTIR DE LA MÉTÉO RÉELLE

with tab_meteo:
    st.subheader("Décision chauffage à partir des prévisions météo (Open-Meteo)")
    st.caption(
        "Nécessite un accès internet actif sur la machine exécutant l'application "
        "(API publique Open-Meteo, sans clé requise)."
    )

    col1, col2 = st.columns(2)
    with col1:
        site_meteo = st.selectbox("Site", sites, key="site_meteo")
    with col2:
        equips_meteo = sorted(df.loc[df[COL_SITE] == site_meteo, COL_EQUIP].unique())
        equip_meteo = st.selectbox("Équipement", equips_meteo, key="equip_meteo")

    ville = st.text_input(
        "Ville ou localisation du site (pour récupérer la météo)",
        placeholder="ex : Lyon, Grenoble, Marseille…",
    )
    base_ref_meteo = st.slider(
        "Référence DJU (°C) utilisée pour convertir la température en DJU",
        14.0, 20.0, 18.0, 0.5, key="base_ref_meteo",
    )

    if ville:
        geo = geocode_ville(ville)
        if geo is None:
            st.error("Ville introuvable ou API injoignable (vérifiez la connexion internet).")
        else:
            st.success(f"Localisation trouvée : {geo['nom']}, {geo['pays']} "
                       f"({geo['lat']:.2f}, {geo['lon']:.2f})")
            meteo = previsions_meteo(geo["lat"], geo["lon"], jours=7)
            if meteo is None:
                st.error("Impossible de récupérer les prévisions météo (API injoignable).")
            else:
                sous_df2 = df[(df[COL_SITE] == site_meteo) & (df[COL_EQUIP] == equip_meteo)]
                model2 = fit_changepoint_model(sous_df2[COL_DJU].values, sous_df2[COL_CONSO].values)

                meteo["dju_prevu"] = dju_depuis_temperature(meteo["temp_moy"], base_ref_meteo)
                meteo["decision"] = meteo["dju_prevu"].apply(
                    lambda v: recommandation(v, model2)
                )

                st.dataframe(
                    meteo.rename(columns={
                        "date": "Date", "temp_moy": "Temp. moyenne (°C)",
                        "temp_min": "Temp. min (°C)", "temp_max": "Temp. max (°C)",
                        "dju_prevu": "DJU prévu", "decision": "Décision",
                    }),
                    width="stretch", hide_index=True,
                )

                fig4 = go.Figure()
                fig4.add_trace(go.Bar(x=meteo["date"], y=meteo["dju_prevu"], name="DJU prévu"))
                fig4.add_hline(
                    y=model2["threshold"], line_dash="dash", line_color="red",
                    annotation_text="Seuil de bascule du site",
                )
                fig4.update_layout(
                    title=f"DJU prévu à 7 jours — {site_meteo} / {equip_meteo}",
                    yaxis_title="DJU", height=400,
                )
                st.plotly_chart(fig4, width="stretch")
    else:
        st.info("Renseignez une ville pour obtenir une recommandation basée sur la météo réelle.")


# ONGLET 4 : SEGMENTATION KMEANS (cohérence avec le notebook du projet)

with tab_segmentation:
    st.subheader("Segmentation des profils de consommation (KMeans)")
    st.caption(
        "Reprend la logique du notebook d'analyse KMeans du projet, pour situer "
        "chaque site dans une famille de comportement de consommation."
    )

    k = st.slider("Nombre de clusters (k)", 2, 8, 4)

    @st.cache_data(show_spinner="Calcul des profils et du clustering…")
    def calculer_profils_et_clusters(df_: pd.DataFrame, k_: int):
        profils = df_.groupby([COL_SITE, COL_EQUIP]).agg(
            conso_moyenne=(COL_CONSO, "mean"),
            conso_ecart_type=(COL_CONSO, "std"),
            dju_moyen=(COL_DJU, "mean"),
        ).reset_index()
        profils["coeff_variation"] = (
            profils["conso_ecart_type"] / profils["conso_moyenne"]
        )
        profils = profils.fillna(profils.median(numeric_only=True))

        features = ["conso_moyenne", "coeff_variation", "dju_moyen"]
        X_scaled = StandardScaler().fit_transform(profils[features])

        km = KMeans(n_clusters=k_, random_state=42, n_init=10)
        profils["cluster"] = km.fit_predict(X_scaled).astype(str)
        return profils

    profils = calculer_profils_et_clusters(df, k)

    fig5 = px.scatter(
        profils, x="dju_moyen", y="conso_moyenne", color="cluster",
        size=900, hover_data=[COL_SITE, COL_EQUIP],
        title="Segmentation des sites par profil de consommation",
        labels={"dju_moyen": "DJU moyen", "conso_moyenne": "Consommation moyenne (kWhef)"},
    )
    st.plotly_chart(fig5, width="stretch")

    st.dataframe(
        profils.sort_values("cluster"), width="stretch", hide_index=True
    )
