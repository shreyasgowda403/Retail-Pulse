import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="RetailPulse", page_icon="📊", layout="wide")

@st.cache_data
def load_data():
    rfm       = pd.read_csv("rfm_with_churn.csv")
    inventory = pd.read_csv("inventory_optimization.csv")
    return rfm, inventory

rfm, inventory = load_data()

st.sidebar.title("📊 RetailPulse")
st.sidebar.caption("AI-Powered Customer Analytics")
st.sidebar.divider()
page = st.sidebar.radio("Navigate", [
    "🏠 Overview", "👥 Customer Segments",
    "⚠️ Churn Analysis", "📦 Inventory"])
st.sidebar.divider()
st.sidebar.caption("Zidio Development · March 2026")

if page == "🏠 Overview":
    st.title("📊 RetailPulse Dashboard")
    st.caption("AI-Powered Customer Analytics & Demand Forecasting")
    st.divider()
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Total Customers",  f"{len(rfm):,}")
    col2.metric("Total Revenue",    f"£{rfm['Monetary'].sum():,.0f}")
    col3.metric("Churn Rate",       f"{rfm['Churned'].mean()*100:.1f}%")
    col4.metric("Avg Order Value",  f"£{rfm['AvgOrderValue'].mean():,.0f}")
    col5.metric("Products Tracked", f"{len(inventory):,}")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        seg = rfm["Segment"].value_counts().reset_index()
        seg.columns = ["Segment","Count"]
        fig = px.bar(seg, x="Segment", y="Count",
                     title="Customer Segment Distribution",
                     color="Segment",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        risk = rfm["Churn_Risk"].value_counts().reset_index()
        risk.columns = ["Risk","Count"]
        fig2 = px.pie(risk, names="Risk", values="Count",
                      title="Churn Risk Distribution",
                      color="Risk",
                      color_discrete_map={"High":"#EF4444",
                                          "Medium":"#F59E0B",
                                          "Low":"#10B981"})
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Segment Performance Summary")
    summary = rfm.groupby("Segment").agg(
        Customers     = ("Customer ID", "count"),
        Avg_Recency   = ("Recency",     "mean"),
        Avg_Frequency = ("Frequency",   "mean"),
        Avg_Monetary  = ("Monetary",    "mean"),
        Churn_Rate    = ("Churned",     "mean")
    ).round(2).reset_index()
    st.dataframe(summary, use_container_width=True)

elif page == "👥 Customer Segments":
    st.title("👥 Customer Segmentation")
    st.divider()
    segments = ["All"] + list(rfm["Segment"].unique())
    selected = st.selectbox("Filter by Segment", segments)
    filtered = rfm if selected == "All" else rfm[rfm["Segment"] == selected]
    st.caption(f"Showing {len(filtered):,} customers")
    col1,col2 = st.columns(2)
    with col1:
        fig = px.scatter(filtered, x="Recency", y="Monetary",
                         color="Segment", size="Frequency",
                         title="RFM Scatter", size_max=20)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cluster = filtered["Cluster_Label"].value_counts().reset_index()
        cluster.columns = ["Cluster","Count"]
        fig2 = px.bar(cluster, x="Cluster", y="Count",
                      title="K-Means Cluster Distribution",
                      color="Cluster")
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Customer Details")
    st.dataframe(filtered[["Customer ID","Recency","Frequency",
                            "Monetary","AvgOrderValue","Segment",
                            "Cluster_Label","Churn_Risk"]].round(2),
                 use_container_width=True)

elif page == "⚠️ Churn Analysis":
    st.title("⚠️ Churn Prediction")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("AUC-ROC",          "0.8331")
    col2.metric("Precision@Top20%", "0.8383")
    col3.metric("High Risk",        f"{(rfm['Churn_Risk']=='High').sum():,}")
    col4.metric("Churn Rate",       f"{rfm['Churned'].mean()*100:.1f}%")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        churn_seg = rfm.groupby("Segment")["Churned"].mean().reset_index()
        churn_seg["Churn_Pct"] = (churn_seg["Churned"]*100).round(1)
        fig = px.bar(churn_seg.sort_values("Churn_Pct", ascending=False),
                     x="Segment", y="Churn_Pct",
                     title="Churn Rate by Segment (%)",
                     color="Churn_Pct", color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.histogram(rfm, x="Churn_Probability",
                            color="Churn_Risk", nbins=50,
                            title="Churn Probability Distribution",
                            color_discrete_map={"High":"#EF4444",
                                                "Medium":"#F59E0B",
                                                "Low":"#10B981"})
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("🚨 High Risk Customers")
    high_risk = rfm[rfm["Churn_Risk"]=="High"].nlargest(20,"Monetary")
    st.dataframe(high_risk[["Customer ID","Recency","Frequency",
                             "Monetary","Segment",
                             "Churn_Probability"]].round(3),
                 use_container_width=True)

elif page == "📦 Inventory":
    st.title("📦 Inventory Optimization")
    st.divider()
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Products",      f"{len(inventory):,}")
    col2.metric("High Velocity", f"{(inventory['StockStatus']=='High Velocity').sum():,}")
    col3.metric("Fast Movers",   f"{(inventory['StockStatus']=='Fast Mover').sum():,}")
    col4.metric("Slow Movers",   f"{(inventory['StockStatus']=='Slow Mover').sum():,}")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        status = inventory["StockStatus"].value_counts().reset_index()
        status.columns = ["Status","Count"]
        fig = px.pie(status, names="Status", values="Count",
                     title="Product Portfolio by Movement",
                     color_discrete_sequence=px.colors.qualitative.Bold)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        rev = inventory.groupby("StockStatus")["TotalRevenue"].sum().reset_index()
        fig2 = px.bar(rev, x="StockStatus", y="TotalRevenue",
                      title="Revenue by Stock Category",
                      color="TotalRevenue", color_continuous_scale="Greens")
        st.plotly_chart(fig2, use_container_width=True)
    st.subheader("Full Inventory Table")
    status_f = st.selectbox("Filter", ["All","High Velocity",
                                        "Fast Mover","Medium Mover","Slow Mover"])
    inv_f = inventory if status_f=="All" else inventory[inventory["StockStatus"]==status_f]
    st.dataframe(inv_f[["StockCode","Description","AvgDailyDemand",
                         "SafetyStock","ReorderPoint","EOQ",
                         "StockStatus","TotalRevenue"]].round(2),
                 use_container_width=True)
