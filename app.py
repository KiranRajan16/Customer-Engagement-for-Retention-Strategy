import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# Page Configuration
st.set_page_config(
    page_icon="🏦",
    layout="wide"
)

# Loading CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title
st.markdown("""
<div class="main-header">
    <p>
        Customer Engagement & Product Utilization Analytics for Retention Strategy
    </p>
</div>
""", unsafe_allow_html=True)

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("data/European_Bank.csv")

    drop_cols = ["CustomerId", "Surname"]

    for col in drop_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

    if "Year" in df.columns:
        df.drop("Year", axis=1, inplace=True)

    return df

df = load_data()

# Feature Engineering
balance_threshold = df["Balance"].quantile(0.75)

df["HighValueCustomer"] = np.where(
    df["Balance"] >= balance_threshold,
    1,
    0
)

df["AtRiskPremium"] = np.where(
    (df["HighValueCustomer"] == 1)
    &
    (df["IsActiveMember"] == 0),
    1,
    0
)

df["RelationshipStrength"] = (
    df["IsActiveMember"] * 40
    +
    df["HasCrCard"] * 20
    +
    df["NumOfProducts"] * 10
)

df["RiskScore"] = (
    (1 - df["IsActiveMember"]) * 40
    +
    (4 - df["NumOfProducts"]) * 15
    +
    (df["Age"] / df["Age"].max()) * 20
)

df["RiskScore"] = df["RiskScore"].clip(0, 100)

def risk_category(score):

    if score < 30:
        return "Low Risk"

    elif score < 60:
        return "Medium Risk"

    else:
        return "High Risk"

df["RiskCategory"] = df["RiskScore"].apply(
    risk_category
)

df["RiskScore"] = (
    (1 - df["IsActiveMember"]) * 40
    +
    (4 - df["NumOfProducts"]) * 15
    +
    (df["Age"] / df["Age"].max()) * 20
)

df["RiskScore"] = df["RiskScore"].clip(0,100)

def segment(row):

    if row["IsActiveMember"] == 1 and row["NumOfProducts"] >= 2:
        return "Active Engaged"

    elif row["IsActiveMember"] == 0 and row["NumOfProducts"] == 1:
        return "Inactive Disengaged"

    elif row["IsActiveMember"] == 1 and row["NumOfProducts"] == 1:
        return "Active Low Product"

    elif row["IsActiveMember"] == 0 and row["Balance"] > df["Balance"].median():
        return "Inactive High Balance"

    else:
        return "Others"

df["EngagementSegment"] = df.apply(segment, axis=1)

st.sidebar.markdown(
    "<h2 class='sidebar-title'>🔍 Filters</h2>",
    unsafe_allow_html=True
)

country = st.sidebar.multiselect(
    "Geography",
    df["Geography"].unique(),
    default=df["Geography"].unique()
)

gender = st.sidebar.multiselect(
    "Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (
        int(df["Age"].min()),
        int(df["Age"].max())
    )
)

balance_range = st.sidebar.slider(
    "Balance Range",
    int(df["Balance"].min()),
    int(df["Balance"].max()),
    (
        int(df["Balance"].min()),
        int(df["Balance"].max())
    )
)

salary_range = st.sidebar.slider(
    "Salary Range",
    int(df["EstimatedSalary"].min()),
    int(df["EstimatedSalary"].max()),
    (
        int(df["EstimatedSalary"].min()),
        int(df["EstimatedSalary"].max())
    )
)

products = st.sidebar.slider(
    "Number of Products",
    int(df["NumOfProducts"].min()),
    int(df["NumOfProducts"].max()),
    (
        int(df["NumOfProducts"].min()),
        int(df["NumOfProducts"].max())
    )
)

filtered_df = df[
    (df["Geography"].isin(country))
    &
    (df["Gender"].isin(gender))
    &
    (df["Age"].between(age_range[0], age_range[1]))
    &
    (df["Balance"].between(balance_range[0], balance_range[1]))
    &
    (df["EstimatedSalary"].between(
        salary_range[0],
        salary_range[1]
    ))
    &
    (df["NumOfProducts"].between(
        products[0],
        products[1]
    ))
]

# KPIs
total_customers = len(filtered_df)

churn_rate = (
    filtered_df["Exited"].mean()
) * 100

retention_rate = 100 - churn_rate

product_depth = (
    filtered_df["NumOfProducts"].mean()
)

relationship_score = (
    filtered_df["RelationshipStrength"].mean()
)

high_value_count = (
    filtered_df["HighValueCustomer"]
    .sum()
)

# KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric(
        "Customers",
        f"{total_customers:,}"
    )
    st.markdown("""
        <div class="kpi-description">
        Engagement Retention Ratio:
        Measures customer retention among active members compared to inactive members.
        </div>
        """, unsafe_allow_html=True)

with kpi2:
    st.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )
    st.markdown("""
        <div class="kpi-description">
        Product Depth Index: 
        Average number of products used by customers.
        </div>
        """, unsafe_allow_html=True)

with kpi3:
    st.metric(
        "Retention Rate",
        f"{retention_rate:.2f}%"
    )
    st.markdown("""
        <div class="kpi-description">
        High Balance Disengagement Rate:
        Measures wealthy inactive customers.
        </div>
        """, unsafe_allow_html=True)

with kpi4:
    st.metric(
        "Product Depth",
        f"{product_depth:.2f}"
    )
    st.markdown("""
        <div class="kpi-description">
        Credit Card Stickiness Score:
        Impact of credit card ownership on retention.
        </div>
        """, unsafe_allow_html=True)
    
with kpi5:
    st.metric(
        "Relationship Score",
        f"{relationship_score:.2f}"
    )
    st.markdown("""
        <div class="kpi-description">
        Relationship Strength Index:
        Combined loyalty score using engagement, product ownership and card usage.
        </div>
        """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Engagement Analysis",
    "Product Utilization",
    "Premium Customers",
    "Retention Strength",
    "Predictive Insights", 
    "Recommendations"
])

# Engagement Analysis
with tab1:

    st.markdown("""
        <div class="tab-title">
        Engagement Segments
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="tab-description">
        This section evaluates customer engagement patterns and their relationship with churn.
        It identifies how customer activity levels influence retention and highlights segments
        requiring engagement-focused interventions.
        </div>
        """, unsafe_allow_html=True)

    fig = px.pie(
        filtered_df,
        names="EngagementSegment",
        hole=0.5
    )

    st.markdown("""
        <div class="graph-description">
        This chart shows the distribution of customers across different engagement segments,
        including Active Engaged, Active Low Product, Inactive Disengaged, and Inactive High 
        Balance customers.
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    active_engaged = len(
        filtered_df[
            filtered_df["EngagementSegment"]=="Active Engaged"
        ]
    )

    st.markdown(f"""
        <div class="insight-box">
        <b>Key Insight:</b><br>
        {active_engaged:,} customers belong to the Active Engaged segment, representing the strongest retention group.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    engagement_churn = (
        filtered_df
        .groupby("IsActiveMember")["Exited"]
        .mean()
        .reset_index()
    )

    fig2 = px.bar(
        engagement_churn,
        x="IsActiveMember",
        y="Exited",
        title="Engagement vs Churn"
    )
   
    st.markdown("""
        <div class="graph-description">
        This chart compares churn rates between active and inactive customers, helping quantify the retention impact of customer engagement.
        </div>
    """, unsafe_allow_html=True)
 
    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Inactive customers generally exhibit significantly higher churn rates than active customers,
        indicating engagement is one of the strongest retention drivers.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
        <div class="tab-title">
        Geography-wise Churn Analysis
        </div>
        """, unsafe_allow_html=True)

    geo_analysis = (
        filtered_df
        .groupby("Geography")["Exited"]
        .mean()
        .reset_index()
    )

    fig_geo = px.bar(
        geo_analysis,
        x="Geography",
        y="Exited",
        color="Exited",
        title="Geography-wise Churn Rate"
    )

    st.markdown("""
        <div class="graph-description">
        This visualization compares churn rates across different geographical regions.
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig_geo,
        use_container_width=True
    )

    highest_geo = geo_analysis.loc[
        geo_analysis["Exited"].idxmax(),
        "Geography"
    ]

    st.markdown(f"""
        <div class="insight-box">
        <b>Insight:</b><br>
        {highest_geo} currently records the highest churn rate, suggesting region-specific retention initiatives may be required.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Product Utilization
with tab2:

    st.markdown("""
        <div class="tab-title">
        Product Utilization
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="tab-description">
        This section examines the relationship between product adoption and customer retention.
        The objective is to determine whether deeper product utilization strengthens loyalty.
        </div>
        """, unsafe_allow_html=True)
    
    product_churn = (
        filtered_df
        .groupby("NumOfProducts")["Exited"]
        .mean()
        .reset_index()
    )

    fig = px.bar(
        product_churn,
        x="NumOfProducts",
        y="Exited",
        color="Exited",
        title="Product Count vs Churn"
    )

    st.markdown("""
        <div class="graph-description">
        Displays churn rates across customers using different numbers of banking products.
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Customers utilizing multiple banking products generally demonstrate higher 
        retention and stronger relationship depth.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    

    fig2 = px.box(
        filtered_df,
        x="NumOfProducts",
        y="Balance",
        title="Balance Distribution by Products"
    )
    
    st.markdown("""
        <div class="graph-description">
        Shows account balance distributions for customers grouped by number of products owned.
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Customers with multiple products typically maintain higher balances, 
        indicating stronger financial commitment to the bank.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Premium Customers
with tab3:

    st.markdown("""
        <div class="tab-title">
        High Value Disengaged Customers
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="tab-description">
        This section focuses on high-value customers and identifies financially valuable
        customers who may be at risk of leaving the bank.
        </div>
        """, unsafe_allow_html=True)

    balance_filter = st.slider(
        "Balance Threshold",
        0,
        int(filtered_df["Balance"].max()),
        int(balance_threshold)
    )

    premium = filtered_df[
        (filtered_df["Balance"] >= balance_filter)
        &
        (filtered_df["IsActiveMember"] == 0)
    ]

    st.metric(
        "Premium Customers At Risk",
        len(premium)
    )

    st.dataframe(
        premium,
        use_container_width=True
    )

    st.markdown(f"""
        <div class="insight-box">
        <b>Insight:</b><br>
        {len(premium):,} premium customers are currently inactive and may require 
        immediate retention interventions.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# Retention Strength
with tab4:

    st.markdown("""
        <div class="tab-title">
        Retention Strength
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="tab-description">
        This section evaluates customer loyalty through relationship strength metrics
        and identifies highly retained 'sticky' customers.
        </div>
        """, unsafe_allow_html=True)
    
    fig = px.histogram(
        filtered_df,
        x="RelationshipStrength",
        color="Exited",
        nbins=20,
        title="Relationship Strength Distribution"
    )

    st.markdown("""
        <div class="graph-description">
        Shows the distribution of customer relationship strength scores and compares retained versus churned customers.
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig,
        use_container_width=True
    )
    
    st.markdown("""
        <div class="insight-box">
        <b>Insight:</b><br>
        Higher relationship strength scores are generally associated with 
        lower churn probabilities and stronger customer loyalty.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    sticky_customers = filtered_df[
        (filtered_df["IsActiveMember"] == 1)
        &
        (filtered_df["NumOfProducts"] >= 2)
        &
        (filtered_df["HasCrCard"] == 1)
    ]

    st.metric(
        "Sticky Customers",
        len(sticky_customers)
    )

    st.dataframe(
        sticky_customers.head(20),
        use_container_width=True
    )
    st.markdown(f"""
        <div class="insight-box">
        <b>Insight:</b><br>
        {len(sticky_customers):,} customers meet the sticky customer criteria, representing the bank's most loyal customer segment.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

with tab5:
    
    st.markdown("""
        <div class="tab-title">
        Customer Churn Risk Prediction
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="tab-description">
        This section provides predictive risk assessment by estimating customer churn risk
        using behavioral indicators and relationship characteristics.
        </div>
        """, unsafe_allow_html=True)

    risk_counts = (
        filtered_df["RiskCategory"]
        .value_counts()
        .reset_index()
    )

    fig_risk = px.pie(
        risk_counts,
        names="RiskCategory",
        values="count",
        hole=0.5
    )

    st.markdown("""
        <div class="graph-description">
        Customers are classified into Low Risk, Medium Risk, and High Risk categories
        based on activity levels, product ownership, and customer profile characteristics.    
        </div>
    """, unsafe_allow_html=True)

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )
    high_risk_count = len(
        filtered_df[
            filtered_df["RiskCategory"]=="High Risk"
        ]
    )

    st.markdown(f"""
        <div class="insight-box">
        <b>Insight:</b><br>
        {high_risk_count:,} customers are currently categorized as High Risk,
        requiring targeted retention strategies.
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    high_risk = filtered_df[
        filtered_df["RiskCategory"]
        == "High Risk"
    ]

    st.metric(
        "High Risk Customers",
        len(high_risk)
    )

    st.dataframe(
        high_risk.head(25),
        use_container_width=True
    )

    st.markdown("""
    <div class="insight-box">
    <b>Recommended Actions:</b>
    <ul>
        <li>Personalized retention campaigns</li>
        <li>Product bundling offers</li>
        <li>Relationship manager outreach</li>
        <li>Loyalty program enrollment</li>
        <li>Engagement reactivation campaigns</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

with tab6:
    st.markdown("""
        <div class="tab-title">
        Strategic Recommendations
        </div>
        """, unsafe_allow_html=True)

    if churn_rate > 20:
        st.warning(
            "High churn detected. Increase engagement campaigns and customer outreach."
        )

    if product_depth < 2:
        st.warning(
            "Low product utilization detected. Promote cross-selling strategies."
        )

    if relationship_score > 70:
        st.success(
            "Strong customer loyalty observed."
        )

    if len(high_risk) > 500:
        st.error(
            "Large number of high-risk customers identified."
        )

# Executive Summary
st.markdown("---")

st.markdown("""
        <div class="tab-description">
        This executive summary consolidates the most important findings from customer engagement,
        product utilization, premium customer analysis, retention strength assessment,
        and predictive risk analytics.
        </div>
        """, unsafe_allow_html=True)

st.markdown(f"""
<div class="summary-box">
    <h3>Executive Summary</h3>

    Total Customers Analysed: {total_customers:,}

    Churn Rate: {churn_rate:.2f}%

    Retention Rate: {retention_rate:.2f}%

    Average Product Depth: {product_depth:.2f}

    Average Relationship Strength: {relationship_score:.2f}

    High Value Customers: {high_value_count:,}
</div>
""", unsafe_allow_html=True)
