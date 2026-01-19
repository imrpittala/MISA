
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import io

# Page configuration
st.set_page_config(page_title="💰 Income & Spending Analyzer", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("💰 Monthly Income & Spending Analyzer")
st.markdown("Analyze your finances, detect anomalies, and forecast spending with AI")

# ============================================
# SIDEBAR: File Upload & Settings
# ============================================
st.sidebar.header("📁 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])
    df['category'] = df['category'].str.strip().str.title()
    
    st.sidebar.success("✅ File uploaded successfully!")
    
    # ============================================
    # TAB 1: Overview
    # ============================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🚨 Anomalies", "📈 Forecast", "💡 Insights", "📉 Details"])
    
    with tab1:
        st.subheader("Financial Summary")
        
        total_income = df[df['type'] == 'Income']['amount'].sum()
        total_expenses = df[df['type'] == 'Expense']['amount'].sum()
        net_savings = total_income - total_expenses
        savings_pct = (net_savings / total_income * 100) if total_income > 0 else 0
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💵 Total Income", f"₹{total_income:,.0f}")
        with col2:
            st.metric("💸 Total Expenses", f"₹{total_expenses:,.0f}")
        with col3:
            st.metric("💰 Net Savings", f"₹{net_savings:,.0f}")
        with col4:
            st.metric("📊 Savings %", f"{savings_pct:.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Category Distribution")
            expenses = df[df['type'] == 'Expense']
            category_spending = expenses.groupby('category')['amount'].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = sns.color_palette("husl", len(category_spending))
            ax.pie(category_spending, labels=category_spending.index, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.set_title('Spending by Category', fontweight='bold')
            st.pyplot(fig)
        
        with col2:
            st.markdown("### Daily Spending Trend")
            daily_spending = expenses.groupby('date')['amount'].sum()
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(daily_spending.index, daily_spending.values, marker='o', linewidth=2, color='#FF6B6B')
            ax.fill_between(daily_spending.index, daily_spending.values, alpha=0.3, color='#FF6B6B')
            ax.set_title('Daily Spending Trend', fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Amount (₹)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        
        # Category breakdown table
        st.markdown("### Category Breakdown")
        cat_stats = expenses.groupby('category').agg({
            'amount': ['sum', 'count', 'mean']
        }).round(0)
        cat_stats.columns = ['Total', 'Count', 'Average']
        cat_stats['Percentage'] = (cat_stats['Total'] / cat_stats['Total'].sum() * 100).round(1)
        st.dataframe(cat_stats.sort_values('Total', ascending=False))
    
    # ============================================
    # TAB 2: Anomaly Detection
    # ============================================
    with tab2:
        st.subheader("🚨 Unusual Spending Detection")
        
        expenses = df[df['type'] == 'Expense'].copy()
        X = expenses[['amount']].values
        
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        expenses['is_anomaly'] = anomalies == -1
        
        anomalous = expenses[expenses['is_anomaly']]
        
        st.metric("🚨 Anomalies Detected", f"{len(anomalous)} transactions")
        
        if len(anomalous) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Anomaly Statistics")
                normal_avg = expenses[~expenses['is_anomaly']]['amount'].mean()
                anomaly_avg = anomalous['amount'].mean()
                st.write(f"**Normal average:** ₹{normal_avg:,.0f}")
                st.write(f"**Anomaly average:** ₹{anomaly_avg:,.0f}")
                st.write(f"**Difference:** ₹{anomaly_avg - normal_avg:,.0f} ({((anomaly_avg/normal_avg - 1)*100):.1f}% higher)")
            
            with col2:
                st.markdown("### Anomalies by Category")
                anom_by_cat = anomalous.groupby('category').size()
                fig, ax = plt.subplots(figsize=(8, 5))
                anom_by_cat.plot(kind='barh', ax=ax, color='#FF6B6B')
                ax.set_title('Anomalies by Category', fontweight='bold')
                st.pyplot(fig)
            
            st.markdown("### Top Anomalous Transactions")
            top_anom = anomalous.nlargest(10, 'amount')[['date', 'description', 'category', 'amount']]
            st.dataframe(top_anom)
            
            # Visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            normal = expenses[~expenses['is_anomaly']]
            ax.scatter(normal['date'], normal['amount'], alpha=0.6, s=50, color='blue', label='Normal')
            ax.scatter(anomalous['date'], anomalous['amount'], alpha=0.8, s=100, color='red', marker='X', label='Anomaly')
            ax.set_title('Transaction Anomaly Detection', fontweight='bold', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Amount (₹)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.success("✅ No anomalies detected! Your spending is consistent.")
    
    # ============================================
    # TAB 3: Forecast
    # ============================================
    with tab3:
        st.subheader("📈 30-Day Spending Forecast")
        
        expenses = df[df['type'] == 'Expense'].copy()
        daily_expenses = expenses.groupby('date')['amount'].sum().reset_index()
        daily_expenses['day_num'] = (daily_expenses['date'] - daily_expenses['date'].min()).dt.days
        
        X = daily_expenses[['day_num']].values
        y = daily_expenses['amount'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_day = daily_expenses['day_num'].max()
        future_days = np.array([[i] for i in range(last_day + 1, last_day + 31)])
        forecast = model.predict(future_days)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Historical Avg Daily", f"₹{y.mean():,.0f}")
        with col2:
            st.metric("📈 Forecasted Avg Daily", f"₹{forecast.mean():,.0f}")
        with col3:
            st.metric("🎯 30-Day Total Forecast", f"₹{forecast.sum():,.0f}")
        
        # Forecast visualization
        last_date = daily_expenses['date'].max()
        forecast_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(daily_expenses['date'], daily_expenses['amount'], marker='o', label='Historical', linewidth=2, color='blue')
        ax.plot(forecast_dates, forecast, marker='s', label='Forecast', linewidth=2, color='orange', linestyle='--')
        ax.axvline(x=last_date, color='red', linestyle=':', alpha=0.5, label='Today')
        ax.set_title('Spending Forecast - Next 30 Days', fontweight='bold', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Daily Spending (₹)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Forecast table
        st.markdown("### Forecast Details")
        forecast_table = pd.DataFrame({
            'Date': forecast_dates,
            'Predicted Daily': forecast,
        })
        forecast_table['Date'] = forecast_table['Date'].dt.date
        st.dataframe(forecast_table.head(10))
    
    # ============================================
    # TAB 4: Smart Insights
    # ============================================
    with tab4:
        st.subheader("💡 Smart Insights & Recommendations")
        
        expenses = df[df['type'] == 'Expense'].copy()
        total_expense = expenses['amount'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 Top Spending Categories")
            category_totals = expenses.groupby('category')['amount'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
            for cat, row in category_totals.head(5).iterrows():
                pct = (row['sum'] / total_expense) * 100
                st.write(f"**{cat}** | ₹{row['sum']:,.0f} ({pct:.1f}%) | Avg: ₹{row['mean']:,.0f}")
        
        with col2:
            st.markdown("### 💰 Saving Opportunities")
            for cat, row in category_totals.iterrows():
                if row['sum'] > total_expense * 0.15:
                    savings = row['sum'] * 0.15
                    st.write(f"📌 Reduce **{cat}** by 15% → Save ₹{savings:,.0f}")
        
        st.markdown("---")
        st.markdown("### ✅ Recommendations")
        
        # Recommendation 1
        if category_totals.iloc[0]['sum'] > total_expense * 0.35:
            st.warning(f"⚠️ {category_totals.index[0]} is {(category_totals.iloc[0]['sum']/total_expense*100):.0f}% of budget - consider reducing")
        
        # Recommendation 2
        daily_avg = y.mean()
        if forecast.mean() > daily_avg * 1.1:
            st.warning("📈 Spending trend is increasing - monitor closely")
        else:
            st.success("✅ Spending trend is stable or decreasing")
        
        # Recommendation 3
        income_total = df[df['type'] == 'Income']['amount'].sum()
        savings_rate = ((income_total - total_expense) / income_total * 100) if income_total > 0 else 0
        if savings_rate > 20:
            st.success(f"🎯 Great savings rate: {savings_rate:.1f}% - Keep it up!")
        else:
            st.warning(f"💡 Current savings rate: {savings_rate:.1f}% - Try to reach 20%+")
    
    # ============================================
    # TAB 5: Detailed Data
    # ============================================
    with tab5:
        st.subheader("📊 All Transactions")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            transaction_type = st.selectbox("Transaction Type", ["All", "Income", "Expense"])
        with col2:
            selected_category = st.multiselect("Categories", df['category'].unique(), default=df['category'].unique())
        with col3:
            min_amount = st.slider("Minimum Amount (₹)", 0, int(df['amount'].max()), 0)
        
        # Apply filters
        filtered_df = df.copy()
        if transaction_type != "All":
            filtered_df = filtered_df[filtered_df['type'] == transaction_type]
        filtered_df = filtered_df[filtered_df['category'].isin(selected_category)]
        filtered_df = filtered_df[filtered_df['amount'] >= min_amount]
        
        st.dataframe(filtered_df.sort_values('date', ascending=False), use_container_width=True)

else:
    st.info("📁 Please upload a CSV file to get started!")
    st.markdown("""
    ### How to use:
    1. Upload your transaction CSV with columns: date, description, category, type, amount, payment_mode
    2. Explore 5 tabs:
       - 📊 **Overview**: Summary & charts
       - 🚨 **Anomalies**: Detect unusual spending
       - 📈 **Forecast**: Predict next 30 days
       - 💡 **Insights**: Smart recommendations
       - 📉 **Details**: View all transactions
    """)
