import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Groundwater Dashboard", layout="wide")

# ============================================================
# CLEANING FUNCTION
# ============================================================

def clean_column(col):
    return pd.to_numeric(
        col.astype(str)
           .str.replace(",", ".", regex=False)
           .str.strip()
           .replace(["", " ", "nan", "None"], np.nan)
           .str.replace(r"^[<>].*", "", regex=True),
        errors="coerce"
    )

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def lade_daten(datei, blatt):
    raw = pd.read_excel(datei, sheet_name=blatt, header=None)

    headers = raw.iloc[0].fillna("").astype(str).str.strip()
    units_row = raw.iloc[1].fillna("").astype(str).str.strip()

    df = raw.iloc[2:].copy()
    df.columns = headers
    df = df.dropna(axis=1, how="all")

    units = {}
    for col in df.columns:
        idx = list(headers).index(col)
        unit = units_row.iloc[idx]
        units[col] = unit if str(unit).strip() != "" else "-"

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

    for col in df.columns:
        if col != "Date":
            df[col] = clean_column(df[col])

    df = df.dropna(axis=1, how="all")
    return df, units

# ============================================================
# CONFIG
# ============================================================

WELL_SHEET_MAP = {
    "GCW1": "GCW1",
    "DRW3/4": "DRW3&4"
}

TARGET_VALUES = {
    "Ammonium": 0.5,
    "Chloride": 250,
    "Fluoride": 1.5,
    "Nitrate": 50,
    "Nitrite": 0.5,
    "Sodium": 200,
    "Sulfate": 250,
    "Conductivity at 25 °C": 2500,
    "Conductivity at 25 ° C": 2500,
    "pH": None,
    "Iron": 200,
    "Manganese": 50
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_numeric_parameters(df):
    numeric_cols = []

    for col in df.columns:
        if col == "Date":
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
            numeric_cols.append(col)

    return numeric_cols

def plot_time_series(df, parameter, well_name, units):
    df = df.sort_values("Date")
    df_plot = df[["Date", parameter]].dropna().copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    if df_plot.empty:
        st.warning(f"No valid data available for {parameter}.")
        return fig

    unit = units.get(parameter, "-")

    ax.plot(
        df_plot["Date"],
        df_plot[parameter],
        marker="o",
        linestyle="-",
        linewidth=1.5,
        markersize=6,
        label=parameter
    )

    if len(df_plot) >= 2:
        x = mdates.date2num(df_plot["Date"])
        y = df_plot[parameter].values

        coeffs = np.polyfit(x, y, 1)
        y_trend = np.polyval(coeffs, x)

        ss_res = np.sum((y - y_trend) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

        trend_label = f"Trend ↓ (r²={r2:.2f})" if y_trend[-1] < y_trend[0] else f"Trend ↑ (r²={r2:.2f})"

        ax.plot(
            df_plot["Date"],
            y_trend,
            linestyle="--",
            linewidth=2,
            color="orange",
            label=trend_label
        )

    ax.set_title(f"Zeitreihe: {parameter}", fontsize=20, fontweight="bold", pad=42)

    ax.text(
        0.5, 1.03,
        f"Messstelle: {well_name}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13
    )

    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(f"Concentration ({unit})", fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)

    if parameter in TARGET_VALUES and TARGET_VALUES[parameter] is not None:
        target = TARGET_VALUES[parameter]
        ax.axhline(
            target,
            linestyle=":",
            linewidth=2,
            color="red",
            label=f"Grenzwert {target} {unit}"
        )

    ax.legend()
    plt.tight_layout()
    return fig

def plot_triangular_correlation(df, cols, title):
    if len(cols) < 2:
        st.warning("At least two numeric parameters are needed for correlation.")
        return None

    df_corr = df[cols].copy()
    df_corr = df_corr.dropna(axis=1, how="all")
    df_corr = df_corr.loc[:, df_corr.notna().sum() > 1]

    if df_corr.shape[1] < 2:
        st.warning("Not enough valid numeric data for correlation.")
        return None

    corr = df_corr.corr(method="pearson")
    mask = np.triu(np.ones_like(corr, dtype=bool))

    n = len(corr.columns)
    fig_size = max(8, min(18, n * 0.7))

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=-1,
        vmax=1,
        linewidths=0.4,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 8},
        ax=ax
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    plt.tight_layout()
    return fig

def prepare_cross_well_dataset(df1, df2, parameters, well1_name, well2_name):
    common_params = [p for p in parameters if p in df1.columns and p in df2.columns]

    if not common_params:
        return None, []

    left = df1[["Date"] + common_params].copy()
    right = df2[["Date"] + common_params].copy()

    left = left.rename(columns={p: f"{well1_name} | {p}" for p in common_params})
    right = right.rename(columns={p: f"{well2_name} | {p}" for p in common_params})

    merged = pd.merge(left, right, on="Date", how="inner")
    corr_cols = [c for c in merged.columns if c != "Date"]

    return merged, corr_cols

# ============================================================
# UI
# ============================================================

st.title("🌍 Groundwater Dashboard")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    st.header("Well selection")

    selected_well = st.selectbox("Select well", list(WELL_SHEET_MAP.keys()), key="main_well")
    selected_sheet = WELL_SHEET_MAP[selected_well]

    df_well, units_well = lade_daten(uploaded_file, selected_sheet)

    st.subheader("Groundwater parameters")

    well_parameters = get_numeric_parameters(df_well)

    if not well_parameters:
        st.error("No matching numeric groundwater parameters were found in this sheet.")
    else:
        selected_parameter = st.selectbox(
            "Select parameter",
            well_parameters,
            key="main_parameter"
        )

        fig_ts = plot_time_series(df_well, selected_parameter, selected_well, units_well)
        st.pyplot(fig_ts, use_container_width=True)

        with st.expander("Show raw data for selected well"):
            st.dataframe(df_well)

    st.header("Correlation")

    tab1, tab2 = st.tabs(["Within one well", "Between two wells"])

    with tab1:
        corr_well = st.selectbox(
            "Select well for internal correlation",
            list(WELL_SHEET_MAP.keys()),
            key="corr_single_well"
        )

        df_corr_single, _ = lade_daten(uploaded_file, WELL_SHEET_MAP[corr_well])
        corr_single_params = get_numeric_parameters(df_corr_single)

        if len(corr_single_params) < 2:
            st.warning("Not enough numeric parameters available for correlation.")
        else:
            selected_corr_params = st.multiselect(
                "Select parameters for internal correlation",
                corr_single_params,
                default=corr_single_params[:8] if len(corr_single_params) > 8 else corr_single_params,
                key="corr_single_params"
            )

            if len(selected_corr_params) < 2:
                st.warning("Please select at least 2 parameters.")
            else:
                fig_corr_single = plot_triangular_correlation(
                    df_corr_single,
                    selected_corr_params,
                    f"Correlation matrix – {corr_well} (Pearson r)"
                )
                if fig_corr_single is not None:
                    st.pyplot(fig_corr_single, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            well_a = st.selectbox(
                "Select first well",
                list(WELL_SHEET_MAP.keys()),
                key="well_a"
            )

        with col2:
            well_b = st.selectbox(
                "Select second well",
                list(WELL_SHEET_MAP.keys()),
                key="well_b"
            )

        df_a, _ = lade_daten(uploaded_file, WELL_SHEET_MAP[well_a])
        df_b, _ = lade_daten(uploaded_file, WELL_SHEET_MAP[well_b])

        common_parameters = sorted(
            list(set(get_numeric_parameters(df_a)).intersection(set(get_numeric_parameters(df_b))))
        )

        if len(common_parameters) < 2:
            st.warning("Not enough common parameters were found between the selected wells.")
        else:
            selected_cross_params = st.multiselect(
                "Select common parameters for cross-well correlation",
                common_parameters,
                default=common_parameters[:6] if len(common_parameters) > 6 else common_parameters,
                key="cross_corr_params"
            )

            if len(selected_cross_params) < 2:
                st.warning("Please select at least 2 parameters.")
            else:
                merged_df, merged_cols = prepare_cross_well_dataset(
                    df_a, df_b, selected_cross_params, well_a, well_b
                )

                if merged_df is None or len(merged_cols) < 2:
                    st.warning("No valid common parameters were found between the selected wells.")
                else:
                    fig_corr_cross = plot_triangular_correlation(
                        merged_df,
                        merged_cols,
                        f"Cross-well correlation – {well_a} vs {well_b} (Pearson r)"
                    )
                    if fig_corr_cross is not None:
                        st.pyplot(fig_corr_cross, use_container_width=True)

                    with st.expander("Show merged data used for cross-well correlation"):
                        st.dataframe(merged_df)