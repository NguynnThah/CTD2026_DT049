from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Seller ESG Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1220px;
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }

        [data-testid="stMetric"] {
            background: rgba(248, 250, 252, 0.92);
            border: 1px solid #e5e7eb;
            border-radius: 14px;
            padding: 14px 16px;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid #e5e7eb;
        }

        .esg-note {
            padding: 14px 16px;
            border-radius: 12px;
            background: #f6f8f7;
            border: 1px solid #e2e8e4;
            margin-bottom: 1rem;
        }

        .small-muted {
            color: #667085;
            font-size: 0.92rem;
        }

        h1, h2, h3 {
            letter-spacing: -0.02em;
        }

        #MainMenu {
            visibility: hidden;
        }

        footer {
            visibility: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

def find_project_root(start: Path) -> Path:
    """
    Locate the project root by finding Dataset/processed.
    The app may be placed in the root or any subfolder.
    """
    candidates = [start] + list(start.parents)

    for candidate in candidates:
        if (candidate / "Dataset" / "processed").exists():
            return candidate

    raise FileNotFoundError(
        "Không tìm thấy thư mục Dataset/processed. "
        "Hãy đặt streamlit_app.py bên trong project CTD2026_DT049."
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
PERFORMANCE_PATH = ESG_MODELING_DIR / "model_performance.csv"
IMPORTANCE_PATH = ESG_MODELING_DIR / "feature_importance.csv"
METADATA_PATH = ESG_MODELING_DIR / "model_metadata.json"


# ============================================================
# 3. ESG SCORING CONFIGURATION
# ============================================================

DIMENSION_WEIGHTS = {
    "E": 0.655,
    "S": 0.290,
    "G": 0.055,
}

DIMENSION_NAMES = {
    "E": "Môi trường",
    "S": "Xã hội",
    "G": "Quản trị",
}

DIMENSION_SHORT_NAMES = {
    "E": "Environmental",
    "S": "Social",
    "G": "Governance",
}

INDICATOR_MAPPING = pd.DataFrame(
    [
        {
            "feature": "sustainable_material_count",
            "indicator": "Sử dụng vật liệu bền vững",
            "dimension_code": "E",
            "direction": "positive",
        },
        {
            "feature": "eco_label_count",
            "indicator": "Áp dụng nhãn sinh thái",
            "dimension_code": "E",
            "direction": "positive",
        },
        {
            "feature": "environmental_keyword_count",
            "indicator": "Cam kết môi trường",
            "dimension_code": "E",
            "direction": "positive",
        },
        {
            "feature": "product_quality_complaint_count",
            "indicator": "Trách nhiệm chất lượng sản phẩm",
            "dimension_code": "S",
            "direction": "negative",
        },
        {
            "feature": "product_damage_complaint_count",
            "indicator": "Độ bền sản phẩm",
            "dimension_code": "S",
            "direction": "negative",
        },
        {
            "feature": "product_safety_complaint_count",
            "indicator": "An toàn sản phẩm",
            "dimension_code": "S",
            "direction": "negative",
        },
        {
            "feature": "customer_service_complaint_count",
            "indicator": "Quản trị quan hệ khách hàng",
            "dimension_code": "G",
            "direction": "negative",
        },
        {
            "feature": "counterfeit_complaint_count",
            "indicator": "Liêm chính trong kinh doanh",
            "dimension_code": "G",
            "direction": "negative",
        },
        {
            "feature": "governance_keyword_count",
            "indicator": "Minh bạch và trách nhiệm",
            "dimension_code": "G",
            "direction": "positive",
        },
    ]
)

INDICATOR_MAPPING["dimension"] = (
    INDICATOR_MAPPING["dimension_code"]
    .map(DIMENSION_NAMES)
)

ESG_FEATURES = INDICATOR_MAPPING["feature"].tolist()

NEGATIVE_FEATURES = (
    INDICATOR_MAPPING.loc[
        INDICATOR_MAPPING["direction"] == "negative",
        "feature",
    ]
    .tolist()
)

SELLER_COLUMN_CANDIDATES = [
    "seller_name",
    "seller",
    "seller_id",
    "shop_name",
    "shop_id",
    "brand",
]

SCORE_LEVELS = [
    (80, "Xuất sắc"),
    (65, "Tốt"),
    (50, "Trung bình"),
    (35, "Cần cải thiện"),
    (0, "Yếu"),
]


# ============================================================
# 4. DATA HELPERS
# ============================================================

def find_first_existing_column(
    columns: Iterable[str],
    candidates: Iterable[str],
) -> str | None:
    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    return None


def detect_seller_column(df: pd.DataFrame) -> str:
    seller_column = find_first_existing_column(
        df.columns,
        SELLER_COLUMN_CANDIDATES,
    )

    if seller_column is None:
        raise ValueError(
            "Không tìm thấy cột định danh seller. "
            "Cần một trong các cột: "
            + ", ".join(SELLER_COLUMN_CANDIDATES)
        )

    return seller_column


def score_level(score: float) -> str:
    for threshold, label in SCORE_LEVELS:
        if score >= threshold:
            return label

    return "Yếu"


def score_status(score: float) -> str:
    if score >= 70:
        return "Tốt"
    if score >= 50:
        return "Trung bình"
    return "Cần cải thiện"


@st.cache_data
def load_feature_dataset() -> pd.DataFrame:
    if not FEATURE_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Không tìm thấy seller_esg_feature_dataset.csv.\n"
            f"Đường dẫn mong đợi: {FEATURE_DATASET_PATH}"
        )

    return pd.read_csv(FEATURE_DATASET_PATH)


@st.cache_data
def load_optional_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    return pd.read_csv(path)


@st.cache_data
def build_dashboard_data(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Scoring logic synchronized with Notebook 06:

    1. Median imputation
    2. Min-Max normalization
    3. Reverse negative indicators
    4. Mean of 3 indicators within each dimension
    5. ESG = 0.655*E + 0.290*S + 0.055*G
    """
    seller_column = detect_seller_column(raw_df)

    missing_features = [
        feature
        for feature in ESG_FEATURES
        if feature not in raw_df.columns
    ]

    if missing_features:
        raise ValueError(
            "Dataset đang thiếu các feature sau:\n- "
            + "\n- ".join(missing_features)
        )

    working = raw_df[
        [seller_column] + ESG_FEATURES
    ].copy()

    for feature in ESG_FEATURES:
        working[feature] = pd.to_numeric(
            working[feature],
            errors="coerce",
        )

    imputer = SimpleImputer(strategy="median")
    imputed_values = imputer.fit_transform(
        working[ESG_FEATURES]
    )

    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(
        imputed_values
    )

    normalized_df = pd.DataFrame(
        normalized_values,
        columns=ESG_FEATURES,
        index=working.index,
    )

    for feature in NEGATIVE_FEATURES:
        normalized_df[feature] = (
            1.0 - normalized_df[feature]
        )

    reference_df = normalized_df.copy()
    reference_df.insert(
        0,
        seller_column,
        working[seller_column].astype(str).values,
    )

    dimension_feature_map = {
        dimension_code: (
            INDICATOR_MAPPING.loc[
                INDICATOR_MAPPING["dimension_code"]
                == dimension_code,
                "feature",
            ]
            .tolist()
        )
        for dimension_code in ["E", "S", "G"]
    }

    for dimension_code, features in dimension_feature_map.items():
        reference_df[f"{dimension_code}_score"] = (
            reference_df[features]
            .mean(axis=1)
            * 100
        )

    reference_df["ESG_reference_score"] = (
        reference_df["E_score"]
        * DIMENSION_WEIGHTS["E"]
        + reference_df["S_score"]
        * DIMENSION_WEIGHTS["S"]
        + reference_df["G_score"]
        * DIMENSION_WEIGHTS["G"]
    )

    reference_df["ESG_level"] = (
        reference_df["ESG_reference_score"]
        .apply(score_level)
    )

    reference_df["ESG_rank"] = (
        reference_df["ESG_reference_score"]
        .rank(
            method="dense",
            ascending=False,
        )
        .astype(int)
    )

    indicator_rows = []

    for _, seller_row in reference_df.iterrows():
        seller_value = seller_row[seller_column]

        for _, mapping_row in INDICATOR_MAPPING.iterrows():
            feature = mapping_row["feature"]
            dimension_code = mapping_row["dimension_code"]
            normalized_score = (
                float(seller_row[feature]) * 100
            )

            local_weight = 1.0 / 3.0
            dimension_weight = DIMENSION_WEIGHTS[
                dimension_code
            ]
            global_weight = (
                dimension_weight * local_weight
            )

            indicator_rows.append(
                {
                    seller_column: seller_value,
                    "feature": feature,
                    "indicator": mapping_row["indicator"],
                    "dimension_code": dimension_code,
                    "dimension": mapping_row["dimension"],
                    "normalized_score": normalized_score,
                    "local_weight": local_weight,
                    "dimension_weight": dimension_weight,
                    "global_weight": global_weight,
                    "dimension_contribution": (
                        normalized_score
                        * local_weight
                    ),
                    "esg_contribution": (
                        normalized_score
                        * global_weight
                    ),
                }
            )

    indicator_long_df = pd.DataFrame(
        indicator_rows
    )

    return (
        reference_df,
        indicator_long_df,
        seller_column,
    )


@st.cache_data
def load_all_data() -> dict:
    raw_df = load_feature_dataset()

    (
        reference_df,
        indicator_long_df,
        seller_column,
    ) = build_dashboard_data(raw_df)

    return {
        "raw": raw_df,
        "reference": reference_df,
        "indicator_long": indicator_long_df,
        "seller_column": seller_column,
        "performance": load_optional_csv(
            PERFORMANCE_PATH
        ),
        "importance": load_optional_csv(
            IMPORTANCE_PATH
        ),
    }


# ============================================================
# 5. UI HELPERS
# ============================================================

def page_header(
    title: str,
    description: str,
) -> None:
    st.title(title)
    st.markdown(
        f'<div class="small-muted">{description}</div>',
        unsafe_allow_html=True,
    )
    st.write("")


def seller_selector(
    reference_df: pd.DataFrame,
    seller_column: str,
    key: str,
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
        "Chọn nhà bán hàng",
        options,
        key=key,
    )


def selected_seller_row(
    reference_df: pd.DataFrame,
    seller_column: str,
    seller: str,
) -> pd.Series:
    return reference_df.loc[
        reference_df[seller_column].astype(str)
        == str(seller)
    ].iloc[0]


def format_score(value: float) -> str:
    return f"{value:.1f}"


def show_dimension_progress(
    label: str,
    score: float,
) -> None:
    st.markdown(
        f"**{label}** · {score:.1f}/100"
    )
    st.progress(
        min(
            max(
                float(score) / 100.0,
                0.0,
            ),
            1.0,
        )
    )


def indicator_table_for_seller(
    indicator_long: pd.DataFrame,
    seller_column: str,
    seller: str,
) -> pd.DataFrame:
    table = indicator_long.loc[
        indicator_long[seller_column].astype(str)
        == str(seller)
    ].copy()

    table["Đánh giá"] = (
        table["normalized_score"]
        .apply(score_status)
    )

    return (
        table[
            [
                "dimension",
                "indicator",
                "normalized_score",
                "Đánh giá",
            ]
        ]
        .rename(
            columns={
                "dimension": "Chiều",
                "indicator": "Chỉ báo",
                "normalized_score": "Điểm",
            }
        )
        .sort_values(
            ["Chiều", "Điểm"],
            ascending=[True, False],
        )
    )


def recommendations_for_seller(
    indicator_long: pd.DataFrame,
    seller_column: str,
    seller: str,
) -> pd.DataFrame:
    rows = indicator_long.loc[
        indicator_long[seller_column].astype(str)
        == str(seller)
    ].copy()

    rows["score_gap"] = (
        100 - rows["normalized_score"]
    ).clip(lower=0)

    rows["priority_score"] = (
        rows["score_gap"]
        * rows["global_weight"]
    )

    return (
        rows.sort_values(
            "priority_score",
            ascending=False,
        )
        .head(3)
        .reset_index(drop=True)
    )


# ============================================================
# 6. PAGES
# ============================================================

def show_overview(data: dict) -> None:
    reference_df = data["reference"]
    seller_column = data["seller_column"]

    page_header(
        "Tổng quan ESG",
        "Bức tranh tổng thể về điểm ESG của các nhà bán hàng trong bộ dữ liệu.",
    )

    average_score = (
        reference_df["ESG_reference_score"]
        .mean()
    )

    top_row = (
        reference_df
        .sort_values(
            "ESG_reference_score",
            ascending=False,
        )
        .iloc[0]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Số nhà bán hàng",
        f"{len(reference_df):,}",
    )
    col2.metric(
        "Điểm ESG trung bình",
        format_score(average_score),
    )
    col3.metric(
        "Điểm cao nhất",
        format_score(
            top_row["ESG_reference_score"]
        ),
    )
    col4.metric(
        "Số chỉ báo",
        "9",
    )

    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Phân bố mức đánh giá")

        level_order = [
            "Xuất sắc",
            "Tốt",
            "Trung bình",
            "Cần cải thiện",
            "Yếu",
        ]

        level_counts = (
            reference_df["ESG_level"]
            .value_counts()
            .reindex(
                level_order,
                fill_value=0,
            )
            .rename_axis("Mức đánh giá")
            .to_frame("Số seller")
        )

        st.bar_chart(
            level_counts,
            height=320,
        )

    with right:
        st.subheader("Top 10 nhà bán hàng")

        ranking = (
            reference_df[
                [
                    seller_column,
                    "ESG_reference_score",
                    "E_score",
                    "S_score",
                    "G_score",
                    "ESG_rank",
                ]
            ]
            .sort_values("ESG_rank")
            .head(10)
            .rename(
                columns={
                    seller_column: "Nhà bán hàng",
                    "ESG_reference_score": "ESG",
                    "E_score": "E",
                    "S_score": "S",
                    "G_score": "G",
                    "ESG_rank": "Hạng",
                }
            )
        )

        st.dataframe(
            ranking.round(1),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown(
        """
        <div class="esg-note">
            <strong>Công thức đang sử dụng:</strong>
            điểm từng chiều là trung bình của 3 chỉ báo;
            điểm ESG tổng hợp = 0,655 × E + 0,290 × S + 0,055 × G.
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_seller_profile(data: dict) -> None:
    reference_df = data["reference"]
    indicator_long = data["indicator_long"]
    seller_column = data["seller_column"]

    page_header(
        "Hồ sơ nhà bán hàng",
        "Xem điểm ESG, chỉ báo thành phần và các ưu tiên cải thiện.",
    )

    selected = seller_selector(
        reference_df,
        seller_column,
        key="profile_seller",
    )

    seller = selected_seller_row(
        reference_df,
        seller_column,
        selected,
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "ESG",
        format_score(
            seller["ESG_reference_score"]
        ),
    )
    c2.metric(
        "Môi trường",
        format_score(
            seller["E_score"]
        ),
    )
    c3.metric(
        "Xã hội",
        format_score(
            seller["S_score"]
        ),
    )
    c4.metric(
        "Quản trị",
        format_score(
            seller["G_score"]
        ),
    )
    c5.metric(
        "Xếp hạng",
        f"#{int(seller['ESG_rank'])}",
    )

    st.markdown(
        f"""
        <div class="esg-note">
            <strong>{selected}</strong> đạt
            <strong>{seller['ESG_reference_score']:.1f}/100</strong>,
            thuộc mức <strong>{seller['ESG_level']}</strong>.
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Hồ sơ E · S · G")

        show_dimension_progress(
            "Môi trường",
            seller["E_score"],
        )
        show_dimension_progress(
            "Xã hội",
            seller["S_score"],
        )
        show_dimension_progress(
            "Quản trị",
            seller["G_score"],
        )

        dimension_chart = pd.DataFrame(
            {
                "Điểm": [
                    seller["E_score"],
                    seller["S_score"],
                    seller["G_score"],
                ]
            },
            index=[
                "Môi trường",
                "Xã hội",
                "Quản trị",
            ],
        )

        st.bar_chart(
            dimension_chart,
            height=280,
        )

    with right:
        st.subheader("Chi tiết chỉ báo")

        indicator_table = indicator_table_for_seller(
            indicator_long,
            seller_column,
            selected,
        )

        st.dataframe(
            indicator_table.round(
                {"Điểm": 1}
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Ưu tiên cải thiện")

    recommendations = recommendations_for_seller(
        indicator_long,
        seller_column,
        selected,
    )

    recommendation_columns = st.columns(3)

    for index, (_, row) in enumerate(
        recommendations.iterrows()
    ):
        with recommendation_columns[index]:
            st.markdown(
                f"""
                <div class="esg-note">
                    <strong>{index + 1}. {row['indicator']}</strong><br>
                    Điểm hiện tại: {row['normalized_score']:.1f}/100<br>
                    Chiều: {row['dimension']}
                </div>
                """,
                unsafe_allow_html=True,
            )


def show_comparison(data: dict) -> None:
    reference_df = data["reference"]
    seller_column = data["seller_column"]

    page_header(
        "So sánh nhà bán hàng",
        "So sánh từ 2 đến 4 nhà bán hàng theo ESG và ba chiều thành phần.",
    )

    options = (
        reference_df[seller_column]
        .astype(str)
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )

    selected = st.multiselect(
        "Chọn nhà bán hàng",
        options,
        default=options[:3],
        max_selections=4,
    )

    if len(selected) < 2:
        st.info(
            "Hãy chọn ít nhất 2 nhà bán hàng."
        )
        return

    comparison_df = reference_df.loc[
        reference_df[seller_column]
        .astype(str)
        .isin(selected),
        [
            seller_column,
            "ESG_reference_score",
            "E_score",
            "S_score",
            "G_score",
            "ESG_rank",
        ],
    ].copy()

    comparison_df = comparison_df.rename(
        columns={
            seller_column: "Nhà bán hàng",
            "ESG_reference_score": "ESG",
            "E_score": "Môi trường",
            "S_score": "Xã hội",
            "G_score": "Quản trị",
            "ESG_rank": "Hạng",
        }
    )

    st.dataframe(
        comparison_df.round(1),
        use_container_width=True,
        hide_index=True,
    )

    chart_df = (
        comparison_df
        .set_index("Nhà bán hàng")[
            [
                "ESG",
                "Môi trường",
                "Xã hội",
                "Quản trị",
            ]
        ]
    )

    st.bar_chart(
        chart_df,
        height=420,
    )

    priority = st.selectbox(
        "Tiêu chí ưu tiên",
        [
            "ESG",
            "Môi trường",
            "Xã hội",
            "Quản trị",
        ],
    )

    best_row = (
        comparison_df
        .sort_values(
            priority,
            ascending=False,
        )
        .iloc[0]
    )

    st.success(
        f"Theo tiêu chí {priority}, "
        f"{best_row['Nhà bán hàng']} đang có điểm cao nhất "
        f"({best_row[priority]:.1f}/100)."
    )


def show_model_results(data: dict) -> None:
    performance_df = data["performance"]
    importance_df = data["importance"]

    page_header(
        "Kết quả mô hình",
        "Tóm tắt hiệu quả của 5 mô hình và mức độ ảnh hưởng của các feature.",
    )

    if performance_df is None:
        st.info(
            "Chưa có model_performance.csv. "
            "Hãy chạy Notebook 06 trước."
        )
    else:
        columns_to_show = [
            column
            for column in [
                "model",
                "test_RMSE",
                "test_R2",
                "CV_RMSE_mean",
                "CV_R2_mean",
            ]
            if column in performance_df.columns
        ]

        display_performance = (
            performance_df[columns_to_show]
            .copy()
        )

        display_performance = (
            display_performance.rename(
                columns={
                    "model": "Mô hình",
                    "test_RMSE": "Test RMSE",
                    "test_R2": "Test R²",
                    "CV_RMSE_mean": "CV RMSE",
                    "CV_R2_mean": "CV R²",
                }
            )
        )

        st.dataframe(
            display_performance.round(6),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            <div class="esg-note">
                <strong>Cách đọc kết quả:</strong>
                Linear Regression là mô hình baseline tái tạo công thức điểm.
                Ridge Regression phù hợp để làm mô hình chính vì đơn giản,
                ổn định và dễ giải thích.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Mức độ ảnh hưởng của feature")

    if importance_df is None:
        st.info(
            "Chưa có feature_importance.csv. "
            "Hãy chạy Notebook 06 trước."
        )
        return

    if (
        "indicator" in importance_df.columns
        and "importance_normalized"
        in importance_df.columns
    ):
        importance_chart = (
            importance_df[
                [
                    "indicator",
                    "importance_normalized",
                ]
            ]
            .dropna()
            .sort_values(
                "importance_normalized",
                ascending=True,
            )
            .set_index("indicator")
            * 100
        )

        importance_chart.columns = [
            "Mức ảnh hưởng (%)"
        ]

        st.bar_chart(
            importance_chart,
            height=420,
        )

        st.dataframe(
            importance_df[
                [
                    column
                    for column in [
                        "importance_rank",
                        "indicator",
                        "dimension",
                        "importance_normalized",
                    ]
                    if column
                    in importance_df.columns
                ]
            ]
            .rename(
                columns={
                    "importance_rank": "Hạng",
                    "indicator": "Chỉ báo",
                    "dimension": "Chiều",
                    "importance_normalized": "Mức ảnh hưởng",
                }
            )
            .round(4),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.dataframe(
            importance_df,
            use_container_width=True,
            hide_index=True,
        )


def show_methodology(data: dict) -> None:
    page_header(
        "Phương pháp",
        "Logic chấm điểm đang được Streamlit sử dụng.",
    )

    st.subheader("Quy trình")

    st.code(
        "Seller-level ESG dataset\n"
        "        ↓\n"
        "Median imputation\n"
        "        ↓\n"
        "Min-Max normalization\n"
        "        ↓\n"
        "Reverse negative complaint indicators\n"
        "        ↓\n"
        "Mean of 3 indicators within E, S and G\n"
        "        ↓\n"
        "ESG = 0.655 × E + 0.290 × S + 0.055 × G"
    )

    st.subheader("Trọng số ba chiều")

    dimension_table = pd.DataFrame(
        {
            "Chiều": [
                "Môi trường",
                "Xã hội",
                "Quản trị",
            ],
            "Mã": [
                "E",
                "S",
                "G",
            ],
            "Trọng số": [
                DIMENSION_WEIGHTS["E"],
                DIMENSION_WEIGHTS["S"],
                DIMENSION_WEIGHTS["G"],
            ],
            "Trọng số tương đương mỗi feature": [
                DIMENSION_WEIGHTS["E"] / 3,
                DIMENSION_WEIGHTS["S"] / 3,
                DIMENSION_WEIGHTS["G"] / 3,
            ],
        }
    )

    st.dataframe(
        dimension_table.round(6),
        use_container_width=True,
        hide_index=True,
    )

    st.latex(
        r"""
        E_i = \frac{E_{1i}+E_{2i}+E_{3i}}{3},
        \quad
        S_i = \frac{S_{1i}+S_{2i}+S_{3i}}{3},
        \quad
        G_i = \frac{G_{1i}+G_{2i}+G_{3i}}{3}
        """
    )

    st.latex(
        r"""
        ESG_i = 0.655E_i + 0.290S_i + 0.055G_i
        """
    )

    st.warning(
        "ESG_reference_score là điểm tham chiếu nội sinh "
        "được xây dựng từ chính các feature đầu vào. "
        "Các mô hình Machine Learning được dùng để kiểm tra "
        "khả năng tái tạo và tính ổn định của hệ thống điểm, "
        "không phải để xác thực một nhãn ESG độc lập."
    )

    st.caption(
        f"Nguồn dữ liệu đang dùng: {FEATURE_DATASET_PATH}"
    )


# ============================================================
# 7. APPLICATION ENTRY POINT
# ============================================================

def main() -> None:
    try:
        data = load_all_data()

    except FileNotFoundError as error:
        st.error(
            "Không tìm thấy dữ liệu đầu vào."
        )
        st.code(str(error))
        st.stop()

    except ValueError as error:
        st.error(
            "Dữ liệu đầu vào chưa đúng cấu trúc."
        )
        st.code(str(error))
        st.stop()

    except Exception as error:
        st.error(
            "Không thể khởi tạo ứng dụng."
        )
        st.exception(error)
        st.stop()

    with st.sidebar:
        st.markdown("## 🌿 Seller ESG")
        st.caption(
            "Dashboard đánh giá ESG cho nhà bán hàng thời trang"
        )

        page = st.radio(
            "Điều hướng",
            [
                "Tổng quan",
                "Hồ sơ nhà bán hàng",
                "So sánh",
                "Kết quả mô hình",
                "Phương pháp",
            ],
            label_visibility="collapsed",
        )

        st.divider()

        st.caption(
            "Trọng số: E 65,5% · S 29,0% · G 5,5%"
        )

    if page == "Tổng quan":
        show_overview(data)

    elif page == "Hồ sơ nhà bán hàng":
        show_seller_profile(data)

    elif page == "So sánh":
        show_comparison(data)

    elif page == "Kết quả mô hình":
        show_model_results(data)

    elif page == "Phương pháp":
        show_methodology(data)


if __name__ == "__main__":
    main()
