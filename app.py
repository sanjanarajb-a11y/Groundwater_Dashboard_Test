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
# CONFIG
# ============================================================

EXCEL_FILE = "grundwasser_daten.xlsx"

WELL_SHEET_MAP = {
    "GCW1": "GCW1",
    "GCW2": "GCW2",
    "GCW3": "GCW3",
    "GCW4": "GCW4",
    "GCW5": "GCW5",
    "DRW3/4": "DRW3&4",
}

FILTER_COLUMNS = ["Date", "Aquifer", "Well", "Screen section", "_SheetWell"]

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
    "Manganese": 50,
}

CONTAMINANTS_SHEET = "Contaminants"

# Edit Pathway 1 and Pathway 2 here if their exact compounds are different.
# The code automatically skips compounds that are not present in the Excel sheet.
PATHWAYS = {
    # Change only the compound lists below if your exact pathway definitions differ.
    # The dashboard automatically uses only compounds available in the Contaminants sheet.
    "Pathway 1": {
        "description": "Anaerobic microbial degradation",
        "compounds": ["PCE", "TCE", "cis-1,2-DCE", "VC", "Ethen"],
    },
    "Pathway 2": {
        "description": "Anaerobic microbial degradation",
        "compounds": ["TCE", "cis-1,2-DCE", "VC", "Ethen"],
    },
    "Pathway 3": {
        "description": "Anaerobic microbial degradation",
        "compounds": ["PCE", "TCE", "1,1-DCE", "VC", "Ethen"],
    },
    "Pathway 4": {
        "description": "Anaerobic microbial degradation",
        "compounds": ["PCE", "TCE", "trans-1,2-DCE", "VC", "Ethen"],
    },
}

COMPOUND_ALIASES = {
    "PCE": ["PCE", "Tetrachloroethene", "Tetrachloroethylene"],
    "TCE": ["TCE", "Trichloroethene", "Trichloroethylene"],
    "cis-1,2-DCE": ["cis-1,2-DCE", "cis-1,2 DCE", "cis- Dichloroethylene", "cis-DCE"],
    "trans-1,2-DCE": ["trans-1,2-DCE", "trans-1,2 DCE", "trans- Dichloroethylene", "trans-DCE"],
    "1,1-DCE": ["1,1-DCE", "1,1 DCE", "1,1-Dichloroethene", "1,1-Dichloroethylene"],
    "VC": ["VC", "Vinyl chloride", "Vinylchloride"],
    "Ethen": ["Ethen", "Ethene"],
}

# ============================================================
# CLEANING + COLUMN HELPERS
# ============================================================

def clean_numeric_column(col):
    """Convert Excel values to numbers.

    Handles German decimal commas and non-detect values like <1, <0.5, < 10.
    For non-detects, the plotted value is half the detection limit:
    <1 -> 0.5, <5 -> 2.5.
    """
    def convert_value(value):
        if pd.isna(value):
            return np.nan

        text = str(value).strip().replace(",", ".")
        if text in ["", " ", "nan", "None", "-"]:
            return np.nan

        is_less_than = text.startswith("<")
        text = text.replace("<", "").replace(">", "").strip()

        number = pd.to_numeric(text, errors="coerce")
        if pd.isna(number):
            return np.nan

        if is_less_than:
            return float(number) / 2
        return float(number)

    return col.apply(convert_value)


def find_column(df, possible_names):
    """Find a column even if spacing/capital letters are slightly different."""
    normalized = {
        str(c).strip().lower().replace("_", " "): c
        for c in df.columns
    }

    for name in possible_names:
        key = name.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]

    return None


def format_option(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def natural_sort(values):
    def sort_key(x):
        text = str(x)
        number = "".join(ch for ch in text if ch.isdigit())
        return (text.rstrip(number), int(number) if number else 9999, text)
    return sorted(values, key=sort_key)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def lade_daten(datei, blatt, well_label):
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

    date_col = find_column(df, ["Date"])
    aquifer_col = find_column(df, ["Aquifer"])
    well_col = find_column(df, ["Well", "Wells"])
    screen_col = find_column(df, ["Screen section", "Screen Section", "Screen"])

    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})

    if aquifer_col and aquifer_col != "Aquifer":
        df = df.rename(columns={aquifer_col: "Aquifer"})

    if well_col and well_col != "Well":
        df = df.rename(columns={well_col: "Well"})

    if screen_col and screen_col != "Screen section":
        df = df.rename(columns={screen_col: "Screen section"})

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])

    # Keep these as filter columns, not graph parameters
    for col in ["Aquifer", "Well", "Screen section"]:
        if col in df.columns:
            df[col] = df[col].apply(format_option)

    for col in df.columns:
        if col not in ["Date", "Aquifer", "Well", "Screen section"]:
            df[col] = clean_numeric_column(df[col])

    df["_SheetWell"] = well_label

    df = df.dropna(axis=1, how="all")
    return df, units


@st.cache_data
def load_selected_wells(datei, selected_well):
    if selected_well == "All":
        all_dfs = []
        all_units = {}

        for well_label, sheet_name in WELL_SHEET_MAP.items():
            df, units = lade_daten(datei, sheet_name, well_label)
            all_dfs.append(df)

            for key, value in units.items():
                if key not in all_units:
                    all_units[key] = value

        return pd.concat(all_dfs, ignore_index=True), all_units

    sheet_name = WELL_SHEET_MAP[selected_well]
    return lade_daten(datei, sheet_name, selected_well)

# ============================================================
# FILTER HELPERS
# ============================================================

def get_filter_options(df, column):
    if column not in df.columns:
        return ["All"]

    values = [
        format_option(v)
        for v in df[column].dropna().unique()
        if format_option(v) not in ["", None]
    ]

    return ["All"] + natural_sort(list(set(values)))


def apply_filter(df, column, selected_value):
    if selected_value == "All" or column not in df.columns:
        return df
    return df[df[column].astype(str) == str(selected_value)]


def safe_dataframe(df):
    """Display dataframe safely in Streamlit.

    Streamlit/pyarrow can fail when Excel creates non-string or NumPy-type
    column/index metadata. This converts visible column names to normal strings
    and resets the index before display.
    """
    df_show = df.copy()
    df_show.columns = [str(c) for c in df_show.columns]
    df_show = df_show.reset_index(drop=True)
    st.dataframe(df_show)

# ============================================================
# PLOTTING HELPERS
# ============================================================

def get_numeric_parameters(df):
    numeric_cols = []

    for col in df.columns:
        if col in FILTER_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().sum() > 1:
            numeric_cols.append(col)

    return numeric_cols


def plot_time_series(df, parameter, well_name, units):
    df = df.sort_values("Date")
    df_plot = df[["Date", "_SheetWell", parameter]].dropna().copy()

    fig, ax = plt.subplots(figsize=(12, 5))

    if df_plot.empty:
        st.warning(f"No valid data available for {parameter}.")
        return fig

    unit = units.get(parameter, "-")

    # If "All" wells are selected, plot one line per well
    for sheet_well, group in df_plot.groupby("_SheetWell"):
        ax.plot(
            group["Date"],
            group[parameter],
            marker="o",
            linestyle="-",
            linewidth=1.5,
            markersize=5,
            label=f"{sheet_well} | {parameter}"
        )

    # One overall trendline for the currently filtered data
    if len(df_plot) >= 2:
        x = mdates.date2num(df_plot["Date"])
        y = df_plot[parameter].values

        coeffs = np.polyfit(x, y, 1)
        y_trend = np.polyval(coeffs, x)

        ss_res = np.sum((y - y_trend) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 0 if ss_tot == 0 else 1 - (ss_res / ss_tot)

        trend_label = f"Overall trend ↓ (r²={r2:.2f})" if y_trend[-1] < y_trend[0] else f"Overall trend ↑ (r²={r2:.2f})"

        trend_df = pd.DataFrame({"Date": df_plot["Date"], "Trend": y_trend}).sort_values("Date")

        ax.plot(
            trend_df["Date"],
            trend_df["Trend"],
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
# CONTAMINANT PATHWAY HELPERS
# ============================================================

def normalize_name(name):
    return (
        str(name).strip().lower()
        .replace("_", " ")
        .replace("-", "")
        .replace(",", "")
        .replace(".", "")
        .replace(" ", "")
    )


def find_header_row(raw):
    for i in range(min(10, len(raw))):
        row_values = [normalize_name(v) for v in raw.iloc[i].fillna("").tolist()]
        joined = " ".join(row_values)
        if "date" in joined and "aquifer" in joined:
            return i
    return 0


@st.cache_data
def load_contaminants_sheet(datei):
    raw = pd.read_excel(datei, sheet_name=CONTAMINANTS_SHEET, header=None)
    raw = raw.dropna(how="all")

    header_idx = find_header_row(raw)
    headers = raw.iloc[header_idx].fillna("").astype(str).str.strip()
    units_row = raw.iloc[header_idx + 1].fillna("").astype(str).str.strip() if header_idx + 1 < len(raw) else pd.Series([])

    df = raw.iloc[header_idx + 2:].copy()
    df.columns = headers
    df = df.dropna(axis=1, how="all")
    df = df.dropna(how="all")

    # Remove repeated header rows inside the table, if present.
    df = df[~df.apply(lambda r: any(str(v).strip().lower() == "aquifer" for v in r.values), axis=1)]

    units = {}
    for idx, col in enumerate(df.columns):
        unit = units_row.iloc[idx] if idx < len(units_row) else "-"
        units[col] = unit if str(unit).strip() != "" else "-"

    date_col = find_column(df, ["Date", "Datum", "Sampling date"])
    aquifer_col = find_column(df, ["Aquifer", "Aquifer section"])
    well_col = find_column(df, ["Well", "Messstelle", "Wells"])
    screen_col = find_column(df, ["Screen section", "Screen Section", "Screen", "Screen section and Well"])

    rename_map = {}
    if date_col: rename_map[date_col] = "Date"
    if aquifer_col: rename_map[aquifer_col] = "Aquifer"
    if well_col: rename_map[well_col] = "Well"
    if screen_col: rename_map[screen_col] = "Screen section"
    df = df.rename(columns=rename_map)

    if "Date" not in df.columns:
        st.error("The Contaminants sheet needs a Date column.")
        return pd.DataFrame(), units

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date"])

    for col in ["Aquifer", "Well", "Screen section"]:
        if col in df.columns:
            df[col] = df[col].apply(format_option)

    for col in df.columns:
        if col not in ["Date", "Aquifer", "Well", "Screen section"]:
            df[col] = clean_numeric_column(df[col])

    return df.dropna(axis=1, how="all"), units


def resolve_compound_columns(df, compounds):
    resolved = []
    for compound in compounds:
        candidates = COMPOUND_ALIASES.get(compound, [compound])
        col = find_column(df, candidates)
        if col is None:
            normalized_cols = {normalize_name(c): c for c in df.columns}
            for candidate in candidates:
                key = normalize_name(candidate)
                if key in normalized_cols:
                    col = normalized_cols[key]
                    break
        if col is not None and col not in resolved:
            resolved.append(col)
    return resolved


def evaluate_pathway_activity(df, compounds):
    existing = [c for c in compounds if c in df.columns]
    if len(existing) < 2:
        return "🔴 No pathway detected", "Not enough compounds from this pathway are present in the selected data."

    detected = []
    increasing_products = 0
    decreasing_parent = 0

    for c in existing:
        values = df[["Date", c]].dropna().sort_values("Date")
        if values[c].gt(0).any():
            detected.append(c)
        if len(values) >= 2:
            first = values[c].iloc[0]
            last = values[c].iloc[-1]
            if c == existing[0] and last < first:
                decreasing_parent += 1
            if c != existing[0] and last > first:
                increasing_products += 1

    detection_ratio = len(detected) / len(existing)

    if detection_ratio >= 0.75 and (decreasing_parent >= 1 or increasing_products >= 2):
        return "🟢 Strong evidence of reductive dechlorination", f"Detected {len(detected)} of {len(existing)} pathway compounds with supportive trend behaviour."
    if detection_ratio >= 0.40:
        return "🟡 Partial pathway observed", f"Detected {len(detected)} of {len(existing)} pathway compounds."
    return "🔴 No pathway detected", f"Detected only {len(detected)} of {len(existing)} pathway compounds."


def plot_pathway_time_series(df, pathway_name, pathway_info, y_scale="linear"):
    compounds = resolve_compound_columns(df, pathway_info["compounds"])
    fig, ax = plt.subplots(figsize=(14, 6))

    if len(compounds) == 0:
        st.warning("None of the selected pathway compounds were found in the Contaminants sheet.")
        return fig, []

    df = df.sort_values("Date")

    for compound in compounds:
        group = df[["Date", compound]].dropna().copy()
        if y_scale == "log":
            group = group[group[compound] > 0]
        if not group.empty:
            ax.plot(group["Date"], group[compound], marker="o", linewidth=2, label=compound)

    title_parts = [pathway_name]
    for col in ["Aquifer", "Well", "Screen section"]:
        if col in df.columns and df[col].nunique() == 1:
            title_parts.append(f"{col}: {df[col].iloc[0]}")

    ax.set_title(" - ".join(title_parts), fontsize=18, fontweight="bold")
    ax.set_xlabel("Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Concentration [µg/L]", fontsize=12, fontweight="bold")
    ax.set_yscale(y_scale)
    ax.grid(True, which="both", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    plt.xticks(rotation=45, ha="right")
    ax.legend(loc="best")

    pathway_text = " - ".join(pathway_info["compounds"])
    fig.text(0.02, 0.02, f"{pathway_name}\n{pathway_text}\n{pathway_info['description']}", fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return fig, compounds

# ============================================================
# UI
# ============================================================

st.title("🌍 Groundwater Dashboard")

# ============================================================
# CONTAMINANT PATHWAY ANALYSIS
# ============================================================

st.header("🧪 Contaminant pathway analysis")

try:
    df_cont, units_cont = load_contaminants_sheet(EXCEL_FILE)
except Exception as e:
    df_cont = pd.DataFrame()
    st.info(f"Contaminants sheet not loaded yet: {e}")

if not df_cont.empty:
    p1, p2, p3, p4 = st.columns(4)

    with p1:
        pathway_name = st.selectbox("Select pathway", list(PATHWAYS.keys()), key="pathway_select")

    with p2:
        selected_cont_aquifer = st.selectbox(
            "Select aquifer",
            get_filter_options(df_cont, "Aquifer"),
            key="pathway_aquifer"
        )

    df_path = apply_filter(df_cont, "Aquifer", selected_cont_aquifer)

    with p3:
        selected_cont_well = st.selectbox(
            "Select well",
            get_filter_options(df_path, "Well"),
            key="pathway_well"
        )

    df_path = apply_filter(df_path, "Well", selected_cont_well)

    with p4:
        selected_cont_screen = st.selectbox(
            "Select screen section",
            get_filter_options(df_path, "Screen section"),
            key="pathway_screen"
        )

    df_path = apply_filter(df_path, "Screen section", selected_cont_screen)

    y_scale = st.radio(
        "Y-axis scale",
        ["linear", "log"],
        horizontal=True,
        key="pathway_y_scale"
    )

    pathway_info = PATHWAYS[pathway_name]
    fig_path, used_compounds = plot_pathway_time_series(df_path, pathway_name, pathway_info, y_scale)
    st.pyplot(fig_path, use_container_width=True)

    status, explanation = evaluate_pathway_activity(df_path, used_compounds)
    if status.startswith("🟢"):
        st.success(status + " — " + explanation)
    elif status.startswith("🟡"):
        st.warning(status + " — " + explanation)
    else:
        st.error(status + " — " + explanation)

    with st.expander("Show contaminant data used for this pathway"):
        visible_cols = [c for c in ["Date", "Aquifer", "Well", "Screen section"] + used_compounds if c in df_path.columns]
        safe_dataframe(df_path[visible_cols].sort_values("Date"))

st.divider()

uploaded_file = EXCEL_FILE

if uploaded_file:
    st.header("Filter selection")

    col1, col2, col3 = st.columns(3)

    with col2:
        selected_well = st.selectbox(
            "Select well",
            ["All"] + list(WELL_SHEET_MAP.keys()),
            key="main_well"
        )

    df_well, units_well = load_selected_wells(uploaded_file, selected_well)

    with col1:
        selected_aquifer = st.selectbox(
            "Select aquifer",
            get_filter_options(df_well, "Aquifer"),
            key="main_aquifer"
        )

    df_filtered = apply_filter(df_well, "Aquifer", selected_aquifer)

    with col3:
        selected_screen = st.selectbox(
            "Select screen section",
            get_filter_options(df_filtered, "Screen section"),
            key="main_screen"
        )

    df_filtered = apply_filter(df_filtered, "Screen section", selected_screen)

    st.subheader("Groundwater parameters")

    well_parameters = get_numeric_parameters(df_filtered)

    if not well_parameters:
        st.error("No matching numeric groundwater parameters were found for the selected filters.")
    else:
        selected_parameter = st.selectbox(
            "Select parameter",
            well_parameters,
            key="main_parameter"
        )

        filter_label = (
            f"Well: {selected_well} | "
            f"Aquifer: {selected_aquifer} | "
            f"Screen section: {selected_screen}"
        )

        fig_ts = plot_time_series(df_filtered, selected_parameter, filter_label, units_well)
        st.pyplot(fig_ts, use_container_width=True)

        with st.expander("Show raw data for selected filters"):
            safe_dataframe(df_filtered)

    st.header("Correlation")

    tab1, tab2 = st.tabs(["Within one well", "Between two wells"])

    with tab1:
        corr_well = st.selectbox(
            "Select well for internal correlation",
            list(WELL_SHEET_MAP.keys()),
            key="corr_single_well"
        )

        df_corr_single, _ = load_selected_wells(uploaded_file, corr_well)

        c1, c2 = st.columns(2)
        with c1:
            corr_aquifer = st.selectbox(
                "Select aquifer for correlation",
                get_filter_options(df_corr_single, "Aquifer"),
                key="corr_aquifer"
            )

        df_corr_single = apply_filter(df_corr_single, "Aquifer", corr_aquifer)

        with c2:
            corr_screen = st.selectbox(
                "Select screen section for correlation",
                get_filter_options(df_corr_single, "Screen section"),
                key="corr_screen"
            )

        df_corr_single = apply_filter(df_corr_single, "Screen section", corr_screen)

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
                    f"Correlation matrix – {corr_well}, Aquifer {corr_aquifer}, Screen {corr_screen} (Pearson r)"
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

        df_a, _ = load_selected_wells(uploaded_file, well_a)
        df_b, _ = load_selected_wells(uploaded_file, well_b)

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
                        safe_dataframe(merged_df)
