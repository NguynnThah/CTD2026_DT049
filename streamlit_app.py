
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


st.set_page_config(
    page_title="Seller ESG Assessment Prototype",
    layout="wide",
)


# ============================================================
# 1. PROJECT PATH CONFIGURATION
# ============================================================

def find_project_root(start: Path) -> Path:
    """
    Find the project root by looking for both:
    - Dataset/
    - notebooks/

    This lets the app work whether it is placed in the project root
    or inside another subfolder.
    """
    candidates = [start] + list(start.parents)

    for candidate in candidates:
        if (candidate / "Dataset").exists() and (candidate / "notebooks").exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the project root. "
        "Place this file somewhere inside the CTD2026_DT049 project."
    )


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(APP_DIR)

FEATURE_DATASET_PATH = (
    PROJECT_ROOT
    / "Dataset"
    / "processed"
    / "seller_esg_feature_dataset.csv"
)

ESG_MODELING_DIR = PROJECT_ROOT / "ESG_Modeling"

MAPPING_PATH = ESG_MODELING_DIR / "ahp_indicator_mapping.csv"
WEIGHT_PATH = ESG_MODELING_DIR / "indicator_weight.csv"
REFERENCE_PATH = ESG_MODELING_DIR / "seller_esg_reference_dataset.csv"
PERFORMANCE_PATH = ESG_MODELING_DIR / "model_performance.csv"
IMPORTANCE_PATH = ESG_MODELING_DIR / "feature_importance.csv"
MODEL_PATH = ESG_MODELING_DIR / "seller_esg_model.pkl"

# ============================================================
# 2. EXPECTED ESG STRUCTURE
# ============================================================

DEFAULT_MAPPING = pd.DataFrame(
    [
        {
            "feature": "sustainable_material_count",
            "indicator": "Sustainable Material Adoption",
            "dimension": "Environmental",
            "dimension_code": "E",
        },
        {
            "feature": "eco_label_count",
            "indicator": "Eco Label Adoption",
            "dimension": "Environmental",
            "dimension_code": "E",
        },
        {
            "feature": "environmental_commitment_score",
            "indicator": "Environmental Commitment",
            "dimension": "Environmental",
            "dimension_code": "E",
        },
        {
            "feature": "product_quality_score",
            "indicator": "Product Quality Responsibility",
            "dimension": "Social",
            "dimension_code": "S",
        },
        {
            "feature": "product_durability_score",
            "indicator": "Product Durability",
            "dimension": "Social",
            "dimension_code": "S",
        },
        {
            "feature": "product_safety_score",
            "indicator": "Product Safety Responsibility",
            "dimension": "Social",
            "dimension_code": "S",
        },
        {
            "feature": "customer_relationship_score",
            "indicator": "Customer Relationship Management",
            "dimension": "Governance",
            "dimension_code": "G",
        },
        {
            "feature": "business_integrity_score",
            "indicator": "Business Integrity",
            "dimension": "Governance",
            "dimension_code": "G",
        },
        {
            "feature": "transparency_score",
            "indicator": "Transparency & Responsibility",
            "dimension": "Governance",
            "dimension_code": "G",
        },
    ]
)

SELLER_COLUMN_CANDIDATES = [
    "seller",
    "seller_name",
    "seller_id",
    "shop_name",
    "shop_id",
    "brand",
]

DIMENSION_NAMES = {
    "E": "Environmental",
    "S": "Social",
    "G": "Governance",
}

SCORE_LEVELS = [
    (80, "Excellent"),
    (65, "Good"),
    (50, "Moderate"),
    (35, "Weak"),
    (0, "Very Weak"),
]


# ============================================================
# 3. DATA VALIDATION AND LOADING
# ============================================================

def find_first_existing_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    normalized = {str(column).strip().lower(): column for column in columns}

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def score_level(score: float) -> str:
    for threshold, label in SCORE_LEVELS:
        if score >= threshold:
            return label
    return "Very Weak"


@st.cache_data
def load_mapping() -> pd.DataFrame:
    if MAPPING_PATH.exists():
        mapping = pd.read_csv(MAPPING_PATH)
    else:
        mapping = DEFAULT_MAPPING.copy()

    required_columns = {
        "feature",
        "indicator",
        "dimension",
        "dimension_code",
    }

    missing_columns = required_columns - set(mapping.columns)
    if missing_columns:
        raise ValueError(
            "ahp_indicator_mapping.csv is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    return mapping


@st.cache_data
def load_weights() -> pd.DataFrame:
    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(
            "indicator_weight.csv was not found.\n\n"
            f"Expected path: {WEIGHT_PATH}\n\n"
            "Run Part 3 of Notebook 06 first so that the AHP weights "
            "are exported to Dataset/ESG_Modeling/indicator_weight.csv."
        )

    weights = pd.read_csv(WEIGHT_PATH)

    required = {
        "feature",
        "indicator",
        "dimension",
        "dimension_code",
        "local_weight",
        "dimension_weight",
        "global_weight",
    }

    missing = required - set(weights.columns)
    if missing:
        raise ValueError(
            "indicator_weight.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    return weights


@st.cache_data
def load_feature_dataset() -> pd.DataFrame:
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "seller_esg_feature_dataset.csv was not found.\n\n"
            f"Expected path: {FEATURE_DATASET_PATH}\n\n"
            "Run Notebook 05 first or check the file path."
        )

    return pd.read_csv(FEATURE_DATASET_PATH)


def validate_features(
    seller_df: pd.DataFrame,
    mapping: pd.DataFrame,
) -> list[str]:
    required_features = mapping["feature"].tolist()

    missing_features = [
        feature
        for feature in required_features
        if feature not in seller_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following ESG features are missing from "
            "seller_esg_feature_dataset.csv:\n"
            + "\n".join(f"- {feature}" for feature in missing_features)
        )

    return required_features


def detect_seller_column(df: pd.DataFrame) -> str:
    seller_column = find_first_existing_column(
        df.columns,
        SELLER_COLUMN_CANDIDATES,
    )

    if seller_column is None:
        raise ValueError(
            "No seller identifier column was found. "
            "Expected one of: "
            + ", ".join(SELLER_COLUMN_CANDIDATES)
        )

    return seller_column


def build_reference_dataset(
    raw_df: pd.DataFrame,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Rebuild the same scoring flow used in Notebook 06:

    raw features
        -> numeric conversion
        -> median imputation
        -> Min-Max normalization
        -> dimension scores
        -> overall ESG reference score
    """
    seller_column = detect_seller_column(raw_df)
    esg_features = weights["feature"].tolist()

    missing_features = [
        feature
        for feature in esg_features
        if feature not in raw_df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following weighted ESG features are missing:\n"
            + "\n".join(f"- {feature}" for feature in missing_features)
        )

    working = raw_df[[seller_column] + esg_features].copy()

    for feature in esg_features:
        working[feature] = pd.to_numeric(
            working[feature],
            errors="coerce",
        )

    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(working[esg_features])

    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(imputed_values)

    normalized_df = pd.DataFrame(
        normalized_values,
        columns=esg_features,
        index=working.index,
    )

    # Convert normalized values from 0-1 to 0-100 for presentation.
    normalized_100_df = normalized_df * 100
    normalized_100_df.insert(
        0,
        seller_column,
        working[seller_column].astype(str).values,
    )

    reference_df = normalized_100_df.copy()

    for dimension_code in ["E", "S", "G"]:
        dimension_rows = weights[
            weights["dimension_code"] == dimension_code
        ].copy()

        dimension_score = np.zeros(len(reference_df))

        for _, row in dimension_rows.iterrows():
            dimension_score += (
                reference_df[row["feature"]].to_numpy()
                * float(row["local_weight"])
            )

        reference_df[f"{dimension_code}_score"] = dimension_score

    dimension_weight_map = (
        weights[["dimension_code", "dimension_weight"]]
        .drop_duplicates("dimension_code")
        .set_index("dimension_code")["dimension_weight"]
        .to_dict()
    )

    reference_df["ESG_reference_score"] = (
        reference_df["E_score"]
        * float(dimension_weight_map["E"])
        + reference_df["S_score"]
        * float(dimension_weight_map["S"])
        + reference_df["G_score"]
        * float(dimension_weight_map["G"])
    )

    reference_df["ESG_level"] = (
        reference_df["ESG_reference_score"]
        .apply(score_level)
    )

    reference_df["ESG_rank"] = (
        reference_df["ESG_reference_score"]
        .rank(
            ascending=False,
            method="dense",
        )
        .astype(int)
    )

    indicator_long_rows = []

    for _, seller_row in reference_df.iterrows():
        seller_value = seller_row[seller_column]

        for _, weight_row in weights.iterrows():
            feature = weight_row["feature"]
            normalized_score = float(seller_row[feature])

            indicator_long_rows.append(
                {
                    seller_column: seller_value,
                    "feature": feature,
                    "indicator": weight_row["indicator"],
                    "dimension": weight_row["dimension"],
                    "dimension_code": weight_row["dimension_code"],
                    "normalized_score": normalized_score,
                    "local_weight": float(weight_row["local_weight"]),
                    "dimension_weight": float(
                        weight_row["dimension_weight"]
                    ),
                    "global_weight": float(weight_row["global_weight"]),
                    "dimension_contribution": (
                        normalized_score
                        * float(weight_row["local_weight"])
                    ),
                    "esg_contribution": (
                        normalized_score
                        * float(weight_row["global_weight"])
                    ),
                }
            )

    indicator_long_df = pd.DataFrame(indicator_long_rows)

    return reference_df, indicator_long_df, seller_column


@st.cache_data
def load_all_data():
    mapping = load_mapping()
    weights = load_weights()
    raw_df = load_feature_dataset()

    validate_features(raw_df, mapping)

    if REFERENCE_PATH.exists():
        reference_df = pd.read_csv(REFERENCE_PATH)
        seller_column = detect_seller_column(reference_df)

        required_scores = {
            "E_score",
            "S_score",
            "G_score",
            "ESG_reference_score",
        }

        if not required_scores.issubset(reference_df.columns):
            reference_df, indicator_long_df, seller_column = (
                build_reference_dataset(raw_df, weights)
            )
        else:
            # Build long indicator table from the saved reference dataset.
            _, indicator_long_df, _ = build_reference_dataset(
                raw_df,
                weights,
            )
    else:
        reference_df, indicator_long_df, seller_column = (
            build_reference_dataset(raw_df, weights)
        )

    return {
        "mapping": mapping,
        "weights": weights,
        "raw": raw_df,
        "reference": reference_df,
        "indicator_long": indicator_long_df,
        "seller_column": seller_column,
    }


# ============================================================
# 4. EXPLANATION LOGIC
# ============================================================

def benchmark_label(
    score: float,
    benchmark: float,
) -> str:
    gap = score - benchmark

    if gap >= 15:
        return "cao hơn đáng kể so với mức trung bình"
    if gap >= 5:
        return "cao hơn mức trung bình"
    if gap <= -15:
        return "thấp hơn đáng kể so với mức trung bình"
    if gap <= -5:
        return "thấp hơn mức trung bình"
    return "xấp xỉ mức trung bình"


def build_indicator_reason(
    selected_row: pd.Series,
    all_indicator_rows: pd.DataFrame,
) -> str:
    same_indicator = all_indicator_rows[
        all_indicator_rows["feature"] == selected_row["feature"]
    ]

    benchmark = same_indicator["normalized_score"].mean()
    position = benchmark_label(
        selected_row["normalized_score"],
        benchmark,
    )

    return (
        f"{selected_row['indicator']} đạt "
        f"{selected_row['normalized_score']:.1f}/100, "
        f"{position}. "
        f"Chỉ báo này có trọng số cục bộ "
        f"{selected_row['local_weight']:.4f} trong chiều "
        f"{selected_row['dimension']} và trọng số toàn cục "
        f"{selected_row['global_weight']:.4f} trong tổng điểm ESG."
    )


def build_dimension_reason(
    dimension_code: str,
    dimension_score: float,
    seller_rows: pd.DataFrame,
    all_rows: pd.DataFrame,
) -> str:
    dimension_rows = seller_rows[
        seller_rows["dimension_code"] == dimension_code
    ].copy()

    strongest = dimension_rows.sort_values(
        "dimension_contribution",
        ascending=False,
    ).iloc[0]

    weakest = dimension_rows.sort_values(
        "normalized_score",
        ascending=True,
    ).iloc[0]

    dimension_name = DIMENSION_NAMES[dimension_code]

    return (
        f"Điểm {dimension_name} đạt {dimension_score:.1f}/100. "
        f"{strongest['indicator']} là yếu tố đóng góp lớn nhất do kết hợp "
        f"giữa điểm chỉ báo và trọng số cục bộ. "
        f"{weakest['indicator']} là chỉ báo có điểm thấp nhất và là khu vực "
        f"cần ưu tiên cải thiện."
    )


# ============================================================
# 5. CHARTS
# ============================================================

def draw_dimension_bar(
    e_score: float,
    s_score: float,
    g_score: float,
):
    labels = ["Environmental", "Social", "Governance"]
    values = [e_score, s_score, g_score]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Score")
    ax.set_title("E, S and G Scores")

    for index, value in enumerate(values):
        ax.text(
            index,
            value + 2,
            f"{value:.1f}",
            ha="center",
        )

    plt.tight_layout()
    return fig


def draw_radar(
    e_score: float,
    s_score: float,
    g_score: float,
):
    labels = ["Environmental", "Social", "Governance"]
    values = [e_score, s_score, g_score]

    angles = np.linspace(
        0,
        2 * np.pi,
        len(labels),
        endpoint=False,
    ).tolist()

    values = values + values[:1]
    angles = angles + angles[:1]

    fig = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title("ESG Dimension Profile", pad=20)

    return fig


# ============================================================
# 6. VIEW HELPERS
# ============================================================

def seller_selector(
    reference_df: pd.DataFrame,
    seller_column: str,
) -> str:
    options = (
        reference_df[seller_column]
        .astype(str)
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    return st.selectbox(
        "Select seller",
        options,
    )


def get_selected_seller(
    reference_df: pd.DataFrame,
    seller_column: str,
    selected_seller: str,
) -> pd.Series:
    return reference_df[
        reference_df[seller_column].astype(str) == str(selected_seller)
    ].iloc[0]


# ============================================================
# 7. STREAMLIT PAGES
# ============================================================

def show_home(data):
    reference_df = data["reference"]

    st.title("Seller ESG Assessment Prototype")
    st.write(
        "Hệ thống sử dụng dữ liệu thực tế từ quy trình xử lý của dự án "
        "để mô phỏng việc chấm điểm ESG cho nhà bán hàng thời trang "
        "trên sàn thương mại điện tử."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Number of sellers", len(reference_df))
    col2.metric("ESG dimensions", 3)
    col3.metric("Operational indicators", 9)

    st.subheader("Data pipeline")
    st.code(
        "Dataset/processed/ESG_Features/seller_esg_feature_dataset.csv\n"
        "        ↓\n"
        "Dataset/ESG_Modeling/indicator_weight.csv\n"
        "        ↓\n"
        "E, S, G dimension scores\n"
        "        ↓\n"
        "ESG reference score"
    )

    st.subheader("Current input paths")
    st.write(f"Feature dataset: `{FEATURE_DATASET_PATH}`")
    st.write(f"Weight dataset: `{WEIGHT_PATH}`")
    st.write(f"Reference dataset: `{REFERENCE_PATH}`")


def show_seller_overview(data):
    reference_df = data["reference"]
    seller_column = data["seller_column"]

    st.title("Seller Overview")

    selected = seller_selector(
        reference_df,
        seller_column,
    )

    seller = get_selected_seller(
        reference_df,
        seller_column,
        selected,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ESG Score",
        f"{seller['ESG_reference_score']:.1f}",
    )
    c2.metric(
        "Environmental",
        f"{seller['E_score']:.1f}",
    )
    c3.metric(
        "Social",
        f"{seller['S_score']:.1f}",
    )
    c4.metric(
        "Governance",
        f"{seller['G_score']:.1f}",
    )

    st.write(
        f"Nhà bán hàng đạt điểm ESG tổng hợp "
        f"{seller['ESG_reference_score']:.1f}/100, "
        f"thuộc nhóm {score_level(seller['ESG_reference_score'])}."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.pyplot(
            draw_dimension_bar(
                seller["E_score"],
                seller["S_score"],
                seller["G_score"],
            )
        )

    with col2:
        st.pyplot(
            draw_radar(
                seller["E_score"],
                seller["S_score"],
                seller["G_score"],
            )
        )


def show_seller_score(data):
    reference_df = data["reference"]
    indicator_long = data["indicator_long"]
    weights = data["weights"]
    seller_column = data["seller_column"]

    st.title("Seller ESG Score")

    selected = seller_selector(
        reference_df,
        seller_column,
    )

    seller = get_selected_seller(
        reference_df,
        seller_column,
        selected,
    )

    seller_rows = indicator_long[
        indicator_long[seller_column].astype(str) == str(selected)
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "ESG Score",
        f"{seller['ESG_reference_score']:.1f} / 100",
    )
    c2.metric(
        "Rating",
        score_level(seller["ESG_reference_score"]),
    )
    c3.metric(
        "Rank",
        int(seller["ESG_rank"]),
    )

    dimension_weight_df = (
        weights[
            [
                "dimension_code",
                "dimension",
                "dimension_weight",
            ]
        ]
        .drop_duplicates("dimension_code")
        .copy()
    )

    dimension_score_map = {
        "E": seller["E_score"],
        "S": seller["S_score"],
        "G": seller["G_score"],
    }

    dimension_weight_df["score"] = (
        dimension_weight_df["dimension_code"]
        .map(dimension_score_map)
    )

    dimension_weight_df["weighted_contribution"] = (
        dimension_weight_df["score"]
        * dimension_weight_df["dimension_weight"]
    )

    st.subheader("Overall score calculation")
    st.dataframe(
        dimension_weight_df[
            [
                "dimension",
                "score",
                "dimension_weight",
                "weighted_contribution",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.latex(
        r"ESG_i = W_E E_i + W_S S_i + W_G G_i"
    )

    for dimension_code in ["E", "S", "G"]:
        st.divider()

        dimension_name = DIMENSION_NAMES[dimension_code]
        dimension_score = seller[f"{dimension_code}_score"]

        dimension_rows = seller_rows[
            seller_rows["dimension_code"] == dimension_code
        ].copy()

        st.subheader(f"{dimension_name} Score")
        st.write(
            f"{dimension_name} đạt {dimension_score:.1f}/100, "
            f"mức đánh giá {score_level(dimension_score)}."
        )

        table = dimension_rows[
            [
                "indicator",
                "normalized_score",
                "local_weight",
                "dimension_contribution",
            ]
        ].copy()

        table.columns = [
            "Indicator",
            "Score",
            "Local weight",
            "Contribution",
        ]

        st.dataframe(
            table.round(4),
            use_container_width=True,
            hide_index=True,
        )

        st.write(
            build_dimension_reason(
                dimension_code,
                dimension_score,
                seller_rows,
                indicator_long,
            )
        )


def show_indicator_analysis(data):
    reference_df = data["reference"]
    indicator_long = data["indicator_long"]
    seller_column = data["seller_column"]

    st.title("Indicator Analysis")

    selected = seller_selector(
        reference_df,
        seller_column,
    )

    seller_rows = indicator_long[
        indicator_long[seller_column].astype(str) == str(selected)
    ].copy()

    display_df = seller_rows[
        [
            "feature",
            "indicator",
            "dimension",
            "normalized_score",
            "local_weight",
            "global_weight",
            "dimension_contribution",
            "esg_contribution",
        ]
    ].copy()

    st.dataframe(
        display_df.round(4),
        use_container_width=True,
        hide_index=True,
    )

    selected_indicator = st.selectbox(
        "Select indicator",
        seller_rows["indicator"].tolist(),
    )

    selected_row = seller_rows[
        seller_rows["indicator"] == selected_indicator
    ].iloc[0]

    st.subheader(selected_indicator)
    st.write(
        build_indicator_reason(
            selected_row,
            indicator_long,
        )
    )

    st.write(
        f"Điểm đóng góp vào chiều "
        f"{selected_row['dimension']}: "
        f"{selected_row['dimension_contribution']:.2f}."
    )

    st.write(
        f"Điểm đóng góp trực tiếp vào ESG tổng hợp: "
        f"{selected_row['esg_contribution']:.2f}."
    )


def show_benchmark(data):
    reference_df = data["reference"]
    seller_column = data["seller_column"]

    st.title("Benchmark")

    selected = seller_selector(
        reference_df,
        seller_column,
    )

    seller = get_selected_seller(
        reference_df,
        seller_column,
        selected,
    )

    score_columns = [
        "ESG_reference_score",
        "E_score",
        "S_score",
        "G_score",
    ]

    average_scores = reference_df[score_columns].mean()

    top_cutoff = reference_df[
        "ESG_reference_score"
    ].quantile(0.90)

    top_group = reference_df[
        reference_df["ESG_reference_score"] >= top_cutoff
    ]

    top_average = top_group[score_columns].mean()

    benchmark_df = pd.DataFrame(
        {
            "Dimension": [
                "ESG",
                "Environmental",
                "Social",
                "Governance",
            ],
            "Selected seller": [
                seller["ESG_reference_score"],
                seller["E_score"],
                seller["S_score"],
                seller["G_score"],
            ],
            "Dataset average": [
                average_scores["ESG_reference_score"],
                average_scores["E_score"],
                average_scores["S_score"],
                average_scores["G_score"],
            ],
            "Top 10% average": [
                top_average["ESG_reference_score"],
                top_average["E_score"],
                top_average["S_score"],
                top_average["G_score"],
            ],
        }
    )

    st.dataframe(
        benchmark_df.round(2),
        use_container_width=True,
        hide_index=True,
    )


def show_recommendations(data):
    reference_df = data["reference"]
    indicator_long = data["indicator_long"]
    seller_column = data["seller_column"]

    st.title("Recommendations")

    selected = seller_selector(
        reference_df,
        seller_column,
    )

    seller_rows = indicator_long[
        indicator_long[seller_column].astype(str) == str(selected)
    ].copy()

    target_score = st.slider(
        "Target indicator score",
        60,
        100,
        85,
    )

    seller_rows["score_gap"] = np.maximum(
        target_score - seller_rows["normalized_score"],
        0,
    )

    seller_rows["priority_score"] = (
        seller_rows["global_weight"]
        * seller_rows["score_gap"]
    )

    recommendation_df = (
        seller_rows.sort_values(
            "priority_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    recommendation_df["Priority"] = (
        recommendation_df.index + 1
    )

    st.dataframe(
        recommendation_df[
            [
                "Priority",
                "indicator",
                "normalized_score",
                "global_weight",
                "score_gap",
                "priority_score",
            ]
        ].round(4),
        use_container_width=True,
        hide_index=True,
    )

    top = recommendation_df.iloc[0]

    st.write(
        f"Chỉ báo nên ưu tiên cải thiện là "
        f"{top['indicator']}. "
        f"Điểm hiện tại là {top['normalized_score']:.1f}/100, "
        f"khoảng cách tới mục tiêu là {top['score_gap']:.1f} điểm."
    )


def show_buyer_discovery(data):
    reference_df = data["reference"]
    seller_column = data["seller_column"]

    st.title("Buyer Seller Discovery")

    min_esg = st.slider(
        "Minimum ESG score",
        0,
        100,
        60,
    )

    min_e = st.slider(
        "Minimum Environmental score",
        0,
        100,
        0,
    )

    min_s = st.slider(
        "Minimum Social score",
        0,
        100,
        0,
    )

    min_g = st.slider(
        "Minimum Governance score",
        0,
        100,
        0,
    )

    filtered = reference_df[
        (reference_df["ESG_reference_score"] >= min_esg)
        & (reference_df["E_score"] >= min_e)
        & (reference_df["S_score"] >= min_s)
        & (reference_df["G_score"] >= min_g)
    ].copy()

    filtered["Rating"] = (
        filtered["ESG_reference_score"]
        .apply(score_level)
    )

    st.dataframe(
        filtered[
            [
                seller_column,
                "ESG_reference_score",
                "E_score",
                "S_score",
                "G_score",
                "Rating",
            ]
        ].sort_values(
            "ESG_reference_score",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )


def show_buyer_profile(data):
    reference_df = data["reference"]
    indicator_long = data["indicator_long"]
    seller_column = data["seller_column"]

    st.title("Buyer Seller Profile")

    selected = seller_selector(
        reference_df,
        seller_column,
    )

    seller = get_selected_seller(
        reference_df,
        seller_column,
        selected,
    )

    seller_rows = indicator_long[
        indicator_long[seller_column].astype(str) == str(selected)
    ].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "ESG",
        f"{seller['ESG_reference_score']:.1f}",
    )
    c2.metric(
        "Environmental",
        f"{seller['E_score']:.1f}",
    )
    c3.metric(
        "Social",
        f"{seller['S_score']:.1f}",
    )
    c4.metric(
        "Governance",
        f"{seller['G_score']:.1f}",
    )

    for dimension_code in ["E", "S", "G"]:
        st.subheader(
            DIMENSION_NAMES[dimension_code]
        )

        st.write(
            build_dimension_reason(
                dimension_code,
                seller[f"{dimension_code}_score"],
                seller_rows,
                indicator_long,
            )
        )

    st.caption(
        "Điểm ESG hiển thị là điểm đánh giá nhà bán hàng "
        "trong mô hình nghiên cứu, không phải chứng nhận "
        "ESG chính thức cho từng sản phẩm."
    )


def show_buyer_comparison(data):
    reference_df = data["reference"]
    seller_column = data["seller_column"]

    st.title("Buyer Seller Comparison")

    seller_options = (
        reference_df[seller_column]
        .astype(str)
        .sort_values()
        .unique()
        .tolist()
    )

    default_selection = seller_options[:3]

    selected = st.multiselect(
        "Select 2 to 4 sellers",
        seller_options,
        default=default_selection,
        max_selections=4,
    )

    if len(selected) < 2:
        st.info(
            "Please select at least two sellers."
        )
        return

    selected_df = reference_df[
        reference_df[seller_column].astype(str).isin(selected)
    ].copy()

    comparison = (
        selected_df.set_index(seller_column)[
            [
                "ESG_reference_score",
                "E_score",
                "S_score",
                "G_score",
            ]
        ]
        .T
    )

    comparison.index = [
        "ESG",
        "Environmental",
        "Social",
        "Governance",
    ]

    st.dataframe(
        comparison.round(2),
        use_container_width=True,
    )

    priority = st.selectbox(
        "Buyer priority",
        [
            "Balanced ESG",
            "Environmental priority",
            "Product quality and safety priority",
            "Transparency and integrity priority",
        ],
    )

    priority_column_map = {
        "Balanced ESG": "ESG_reference_score",
        "Environmental priority": "E_score",
        "Product quality and safety priority": "S_score",
        "Transparency and integrity priority": "G_score",
    }

    score_column = priority_column_map[priority]

    best_seller = selected_df.sort_values(
        score_column,
        ascending=False,
    ).iloc[0]

    st.write(
        f"Với ưu tiên hiện tại, "
        f"{best_seller[seller_column]} có điểm cao nhất "
        f"trong nhóm được chọn."
    )


def show_methodology(data):
    weights = data["weights"]

    st.title("Methodology")

    st.subheader("Research framework")
    st.code(
        "Seller ESG feature dataset\n"
        "        ↓\n"
        "Feature validation\n"
        "        ↓\n"
        "Median imputation\n"
        "        ↓\n"
        "Min-Max normalization\n"
        "        ↓\n"
        "AHP local and global weights\n"
        "        ↓\n"
        "E, S, G scores\n"
        "        ↓\n"
        "ESG reference score"
    )

    st.subheader("Dimension score")
    st.latex(
        r"D_i = \sum_{j=1}^{m} w_{j|D}x_{ij}"
    )

    st.subheader("Overall ESG score")
    st.latex(
        r"ESG_i = W_EE_i + W_SS_i + W_GG_i"
    )

    st.latex(
        r"ESG_i = \sum_{j=1}^{9}GW_jx_{ij}"
    )

    st.subheader("AHP weights")
    st.dataframe(
        weights,
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "Prototype sử dụng trực tiếp các file đầu ra của "
        "Notebook 05 và Notebook 06. "
        "Kết quả phục vụ mục đích nghiên cứu và trình diễn, "
        "không thay thế chứng nhận ESG hoặc kiểm toán độc lập."
    )


# ============================================================
# 8. APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    """
    Main entry point of the Streamlit application.
    """

    try:
        data = load_all_data()

    except FileNotFoundError as error:
        st.error("Không tìm thấy file dữ liệu đầu vào.")
        st.code(str(error))
        st.stop()
        return

    except ValueError as error:
        st.error("Dữ liệu đầu vào chưa đúng cấu trúc yêu cầu.")
        st.code(str(error))
        st.stop()
        return

    except Exception as error:
        st.error("Không thể khởi tạo ứng dụng.")
        st.exception(error)
        st.stop()
        return

    page = st.sidebar.radio(
        "Navigation",
        [
            "Home",
            "Seller Overview",
            "Seller ESG Score",
            "Indicator Analysis",
            "Benchmark",
            "Recommendations",
            "Buyer Discovery",
            "Buyer Seller Profile",
            "Buyer Comparison",
            "Methodology",
        ],
    )

    if page == "Home":
        show_home(data)

    elif page == "Seller Overview":
        show_seller_overview(data)

    elif page == "Seller ESG Score":
        show_seller_score(data)

    elif page == "Indicator Analysis":
        show_indicator_analysis(data)

    elif page == "Benchmark":
        show_benchmark(data)

    elif page == "Recommendations":
        show_recommendations(data)

    elif page == "Buyer Discovery":
        show_buyer_discovery(data)

    elif page == "Buyer Seller Profile":
        show_buyer_profile(data)

    elif page == "Buyer Comparison":
        show_buyer_comparison(data)

    elif page == "Methodology":
        show_methodology(data)


if __name__ == "__main__":
    main()