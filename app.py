import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DATA_FILE = Path("SCM_Dataset_Updated_with_Green_Logistics.xlsx")
MODEL_FILE = Path("carbon_emissions_model.joblib")
TARGET_COLUMN = "Carbon Emissions (kg CO2e)"
ROW_ID_COLUMNS = ["Company Name"]
SEED = 42
EXTRA_FEATURES = [
    "Circularity Index",
    "Supplier Energy Load",
    "Fulfillment Efficiency Gap",
]


st.set_page_config(
    page_title="Carbon Emission Estimator",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    :root {
        --forest: #12372f;
        --teal: #0a7f83;
        --mint: #8be0bc;
        --coral: #f26f5e;
        --gold: #e0aa3e;
        --ink: #14231f;
        --muted: #64736e;
        --paper: rgba(255, 255, 255, 0.88);
        --line: rgba(20, 35, 31, 0.13);
    }

    .stApp {
        background:
            linear-gradient(135deg, rgba(139,224,188,0.24), transparent 34%),
            linear-gradient(315deg, rgba(242,111,94,0.16), transparent 38%),
            #f7faf6;
        color: var(--ink);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(18,55,47,0.98), rgba(10,127,131,0.92)),
            repeating-linear-gradient(45deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 13px);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p {
        color: #f7fffb;
    }

    [data-testid="stSidebar"] label { font-weight: 750; }
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label,
    [data-testid="stSidebar"] .stNumberInput label {
        color: #f7fffb;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] div[data-baseweb="select"] span,
    [data-testid="stSidebar"] div[data-baseweb="select"] input {
        color: #14231f !important;
    }

    [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
        color: #f7fffb;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 1180px;
    }

    .hero {
        border: 1px solid var(--line);
        background:
            linear-gradient(135deg, rgba(18,55,47,0.96), rgba(10,127,131,0.88)),
            repeating-linear-gradient(45deg, rgba(255,255,255,0.10) 0 1px, transparent 1px 14px);
        color: white;
        padding: 30px;
        box-shadow: 0 24px 70px rgba(18, 55, 47, 0.16);
        margin-bottom: 18px;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2rem, 4vw, 4.6rem);
        line-height: 0.96;
        letter-spacing: 0;
    }

    .hero p {
        margin: 16px 0 0;
        max-width: 780px;
        color: rgba(255,255,255,0.80);
        font-size: 1.02rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 16px 0;
    }

    .tile, .result-card, .insight-card {
        background: var(--paper);
        border: 1px solid var(--line);
        box-shadow: 0 16px 42px rgba(18,55,47,0.08);
    }

    .tile {
        min-height: 98px;
        padding: 16px;
    }

    .tile span, .eyebrow {
        display: block;
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.1rem;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .tile strong {
        color: var(--ink);
        font-size: 1.32rem;
        letter-spacing: 0;
    }

    .result-card {
        padding: 24px;
        min-height: 290px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        background:
            linear-gradient(145deg, rgba(255,255,255,0.97), rgba(247,250,246,0.88)),
            radial-gradient(circle at 90% 10%, rgba(224,170,62,0.18), transparent 30%);
    }

    .result-value {
        color: var(--teal);
        font-size: clamp(2.4rem, 5.2vw, 5.8rem);
        line-height: 0.92;
        font-weight: 900;
        letter-spacing: 0;
        margin: 10px 0 4px;
    }

    .unit {
        color: var(--muted);
        font-weight: 800;
    }

    .band {
        width: fit-content;
        padding: 8px 11px;
        border-radius: 6px;
        color: white;
        font-weight: 850;
    }

    .insight-card {
        padding: 18px;
        min-height: 136px;
    }

    .insight-card h3 {
        margin: 0 0 8px;
        font-size: 1.04rem;
        letter-spacing: 0;
    }

    .insight-card p {
        color: var(--muted);
        margin: 0;
    }

    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, var(--gold), var(--coral));
        border: 0;
        color: #1d211b;
        min-height: 48px;
        border-radius: 7px;
        font-weight: 900;
        box-shadow: 0 12px 28px rgba(242,111,94,0.24);
    }

    div[data-testid="stButton"] > button:hover {
        border: 0;
        color: #1d211b;
        filter: brightness(1.04);
    }

    div[data-testid="stAlert"] {
        border-radius: 7px;
    }

    @media (max-width: 900px) {
        .metric-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_cost_amount(raw_value):
    """Normalize compact money strings such as '$12.5M' into numeric values."""
    if pd.isna(raw_value):
        return np.nan

    cleaned = str(raw_value).strip().replace(",", "").replace("$", "")
    match = re.fullmatch(r"([-+]?\d*\.?\d+)\s*([KMBT]?)", cleaned, flags=re.IGNORECASE)
    if match is None:
        return pd.to_numeric(cleaned, errors="coerce")

    scale = {"": 1, "K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
    return float(match.group(1)) * scale[match.group(2).upper()]


def add_supply_chain_ratios(frame):
    enriched = frame.copy()
    suppliers = enriched["Supplier Count"].replace(0, np.nan)
    energy = enriched["Energy Consumption (MWh)"]
    renewable_ratio = enriched["Use of Renewable Energy (%)"] / 100
    recycling_ratio = enriched["Recycling Rate (%)"] / 100
    packaging_ratio = enriched["Green Packaging Usage (%)"] / 100

    enriched["Energy per Supplier"] = energy / suppliers
    enriched["Implementation Cost per Supplier"] = enriched["Total Implementation Cost"] / suppliers
    enriched["Renewable Energy MWh"] = energy * renewable_ratio
    enriched["Non Renewable Energy MWh"] = energy * (1 - renewable_ratio)
    enriched["Recycling Packaging Score"] = recycling_ratio * enriched["Green Packaging Usage (%)"]
    enriched["Efficiency Risk Balance"] = (
        enriched["Operational Efficiency Score"] - enriched["Supply Chain Risk (%)"]
    )
    enriched["Log Energy Consumption"] = np.log1p(energy)
    enriched["Circularity Index"] = (recycling_ratio + packaging_ratio + renewable_ratio) / 3
    enriched["Supplier Energy Load"] = np.log1p(enriched["Energy per Supplier"])
    enriched["Fulfillment Efficiency Gap"] = (
        100 - enriched["Order Fulfillment Rate (%)"]
    ) + (100 - enriched["Operational Efficiency Score"])
    return enriched


@st.cache_data(show_spinner=False)
def load_supply_chain_data():
    raw_data = pd.read_excel(DATA_FILE)
    raw_data["Cost of Goods Sold (COGS)"] = raw_data["Cost of Goods Sold (COGS)"].apply(
        parse_cost_amount
    )
    clean_data = raw_data.dropna(subset=[TARGET_COLUMN]).drop_duplicates()
    return add_supply_chain_ratios(clean_data)


def split_feature_types(features):
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = features.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(numeric_features, categorical_features):
    numeric_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_steps = Pipeline(
        steps=[
            ("fill_missing", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_steps, numeric_features),
            ("categorical", categorical_steps, categorical_features),
        ]
    )


def candidate_regressors():
    return {
        "Linear regression": LinearRegression(),
        "Ridge regression": RidgeCV(alphas=np.logspace(-8, 4, 40)),
        "Extra trees": ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Random forest": RandomForestRegressor(
            n_estimators=700,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=SEED,
            n_jobs=-1,
        ),
        "Gradient boosting": GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.9,
            random_state=SEED,
        ),
    }


def score_candidate_models(x_train, x_test, y_train, y_test, preprocessor):
    rows = []
    fitted = {}

    for name, regressor in candidate_regressors().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("regressor", regressor),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        fitted[name] = pipeline
        rows.append(
            {
                "model": name,
                "test_mae": mean_absolute_error(y_test, predictions),
                "test_rmse": root_mean_squared_error(y_test, predictions),
                "test_r2": r2_score(y_test, predictions),
            }
        )

    leaderboard = pd.DataFrame(rows).sort_values("test_rmse")
    best_name = leaderboard.iloc[0]["model"]
    return fitted[best_name], leaderboard


@st.cache_resource(show_spinner=False)
def load_or_create_model():
    dataset = load_supply_chain_data()

    if MODEL_FILE.exists():
        package = joblib.load(MODEL_FILE)
        if set(EXTRA_FEATURES).issubset(package.get("features", [])):
            return package, dataset

    x = dataset.drop(columns=[TARGET_COLUMN, *ROW_ID_COLUMNS])
    y = dataset[TARGET_COLUMN].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=SEED,
    )

    numeric_features, categorical_features = split_feature_types(x_train)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    best_model, leaderboard = score_candidate_models(
        x_train,
        x_test,
        y_train,
        y_test,
        preprocessor,
    )
    best_model.fit(x, y)

    model_package = {
        "model": best_model,
        "target": TARGET_COLUMN,
        "features": x.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "test_metrics": leaderboard.iloc[0].to_dict(),
        "model_results": leaderboard.to_dict(orient="records"),
    }
    joblib.dump(model_package, MODEL_FILE)
    return model_package, dataset


def dataset_default_row(dataset, feature_names):
    defaults = {}
    for feature in feature_names:
        if pd.api.types.is_numeric_dtype(dataset[feature]):
            defaults[feature] = float(dataset[feature].median())
        else:
            mode_values = dataset[feature].mode(dropna=True)
            defaults[feature] = str(mode_values.iloc[0]) if not mode_values.empty else ""
    return defaults


def format_technology_list(selected_values):
    preferred_order = ["ERP", "AI", "Blockchain", "Robotics", "JIT"]
    ordered = [technology for technology in preferred_order if technology in selected_values]
    return ", ".join(ordered) if ordered else "ERP"


def classify_emissions(prediction):
    if prediction < 120000:
        return "Low emissions - Efficient supply chain", "low"
    if prediction < 200000:
        return "Moderate emissions - Optimization possible", "moderate"
    return "High emissions - Action required", "high"


def recommended_actions(values, prediction):
    checks = [
        (
            values["Supplier Count"] > 1000,
            "Consider reducing supplier complexity.",
        ),
        (
            values["Lead Time (days)"] > 15,
            "Reducing lead time can lower emissions.",
        ),
        (
            values["Use of Renewable Energy (%)"] < 50,
            "Increase renewable energy usage.",
        ),
        (
            values["Recycling Rate (%)"] < 60,
            "Improve recycling in packaging and warehouse waste.",
        ),
        (
            values["Green Packaging Usage (%)"] < 50,
            "Use more green packaging materials.",
        ),
        (
            values["Supply Chain Risk (%)"] > 20,
            "Reduce supply chain risk with better supplier planning.",
        ),
        (
            values["Operational Efficiency Score"] < 85,
            "Improve operational efficiency with better routing and inventory control.",
        ),
        (
            prediction >= 200000,
            "Focus first on reducing energy consumption.",
        ),
    ]
    actions = [message for condition, message in checks if condition]
    return actions[:4] or [
        "Your inputs look balanced. Keep monitoring energy use and recycling."
    ]


def make_improved_scenario(values):
    improved = values.copy()
    improved["Use of Renewable Energy (%)"] = min(
        100,
        values["Use of Renewable Energy (%)"] + 20,
    )
    improved["Recycling Rate (%)"] = min(
        100,
        values["Recycling Rate (%)"] + 15,
    )
    improved["Green Packaging Usage (%)"] = min(
        100,
        values["Green Packaging Usage (%)"] + 15,
    )
    improved["Supply Chain Risk (%)"] = max(
        0,
        values["Supply Chain Risk (%)"] - 5,
    )
    improved["Operational Efficiency Score"] = min(
        100,
        values["Operational Efficiency Score"] + 5,
    )
    return improved


def build_prediction_frame(dataset, feature_names, user_values):
    row = dataset_default_row(dataset, feature_names)
    row.update(user_values)
    prediction_frame = pd.DataFrame([row])
    prediction_frame = add_supply_chain_ratios(prediction_frame)
    return prediction_frame.reindex(columns=feature_names)


try:
    model_package, dataset = load_or_create_model()
except Exception as exc:
    st.error(f"Could not load or train the model: {exc}")
    st.stop()


features = model_package["features"]
model = model_package["model"]


with st.sidebar:
    st.header("Input Details")

    st.subheader("Supply Chain")
    scm_practice = st.selectbox(
        "SCM Practice",
        ["Agile SCM", "Lean Manufacturing", "Cross-Docking", "Sustainable SCM"],
    )
    supplier_count = st.number_input("Supplier Count", 0, 50000, 500)
    lead_time = st.slider("Lead Time (days)", 1, 30, 10)

    st.subheader("Operations")
    inventory_turnover = st.slider("Inventory Turnover Ratio", 1.0, 15.0, 5.0)
    order_rate = st.slider("Order Fulfillment Rate (%)", 50, 100, 90)
    customer_sat = st.slider("Customer Satisfaction (%)", 50, 100, 90)
    efficiency = st.slider("Operational Efficiency Score", 50, 100, 85)

    st.subheader("Sustainability")
    recycling = st.slider("Recycling Rate (%)", 0, 100, 50)
    renewable = st.slider("Renewable Energy Usage (%)", 0, 100, 40)
    green_packaging = st.slider("Green Packaging Usage (%)", 0, 100, 30)
    energy_consumption = st.slider("Energy Consumption (MWh)", 50000, 700000, 150000, 500)

    st.subheader("Technology")
    technology = st.multiselect(
        "Technology Used",
        ["AI", "ERP", "Blockchain", "Robotics", "JIT"],
        default=["AI", "ERP"],
    )

    st.subheader("Risk")
    risk = st.slider("Supply Chain Risk (%)", 0, 100, 10)

    predict_clicked = st.button(
        "Predict Emission",
        key="predict_button",
        use_container_width=True,
    )


st.markdown(
    """
    <div class="hero">
        <h1>Carbon Emission Estimator</h1>
        <p>Estimate carbon emissions and get actionable insights to optimize your supply chain.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

user_inputs = {
    "SCM Practices": scm_practice,
    "Supplier Count": supplier_count,
    "Lead Time (days)": lead_time,
    "Inventory Turnover Ratio": inventory_turnover,
    "Order Fulfillment Rate (%)": order_rate,
    "Customer Satisfaction (%)": customer_sat,
    "Technology Utilized": format_technology_list(technology),
    "Recycling Rate (%)": recycling,
    "Use of Renewable Energy (%)": renewable,
    "Green Packaging Usage (%)": green_packaging,
    "Energy Consumption (MWh)": energy_consumption,
    "Supply Chain Risk (%)": risk,
    "Operational Efficiency Score": efficiency,
}

input_row = build_prediction_frame(dataset, features, user_inputs)
prediction = float(model.predict(input_row)[0])
message, band_name = classify_emissions(prediction)
median_emissions = float(dataset[TARGET_COLUMN].median())
delta_from_median = prediction - median_emissions
improved_values = make_improved_scenario(user_inputs)
improved_row = build_prediction_frame(dataset, features, improved_values)
improved_prediction = float(model.predict(improved_row)[0])
estimated_reduction = prediction - improved_prediction
band_colors = {
    "low": "#0f8f68",
    "moderate": "#d6a336",
    "high": "#d95245",
}


st.markdown(
    f"""
    <div class="metric-grid">
        <div class="tile"><span>Renewable Energy</span><strong>{renewable}%</strong></div>
        <div class="tile"><span>Recycling Rate</span><strong>{recycling}%</strong></div>
        <div class="tile"><span>Supply Risk</span><strong>{risk}%</strong></div>
        <div class="tile"><span>Efficiency</span><strong>{efficiency}/100</strong></div>
    </div>
    """,
    unsafe_allow_html=True,
)


if predict_clicked:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="result-card">
                <div>
                    <span class="eyebrow">Predicted Emissions</span>
                    <div class="result-value">{prediction:,.0f}</div>
                    <div class="unit">kg CO2e</div>
                </div>
                <div class="band" style="background:{band_colors[band_name]};">{message}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            f"""
            <div class="metric-grid" style="grid-template-columns: repeat(2, 1fr); margin-top:0;">
                <div class="tile"><span>Impact Level</span><strong>{message.split(" - ")[0]}</strong></div>
                <div class="tile"><span>Vs Average</span><strong>{delta_from_median:,.0f}</strong></div>
                <div class="tile"><span>Improved Estimate</span><strong>{improved_prediction:,.0f}</strong></div>
                <div class="tile"><span>Possible Reduction</span><strong>{estimated_reduction:,.0f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Recommendations")
    suggestion_cols = st.columns(2)
    for index, suggestion in enumerate(recommended_actions(user_inputs, prediction)):
        with suggestion_cols[index % 2]:
            st.markdown(
                f"""
                <div class="insight-card">
                    <h3>Action {index + 1}</h3>
                    <p>{suggestion}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.subheader("Emission Comparison")
    scenario_chart = pd.DataFrame(
        {
            "Scenario": ["Current", "Improved"],
            "Carbon Emissions": [prediction, improved_prediction],
        }
    )
    st.bar_chart(scenario_chart, x="Scenario", y="Carbon Emissions", color="#f26f5e")

    benchmark_chart = pd.DataFrame(
        {
            "Benchmark": ["Low", "Median", "Prediction", "High"],
            "Carbon Emissions": [
                dataset[TARGET_COLUMN].quantile(0.25),
                median_emissions,
                prediction,
                dataset[TARGET_COLUMN].quantile(0.75),
            ],
        }
    )
    st.bar_chart(benchmark_chart, x="Benchmark", y="Carbon Emissions", color="#0a7f83")
else:
    st.markdown(
        """
        <div class="insight-card">
            <h3>Ready to estimate emissions</h3>
            <p>Adjust the sidebar values and click Predict Emission to generate the result, insights, recommendations, and comparison chart.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
