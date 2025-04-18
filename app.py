import pandas as pd
import streamlit as st

# Load cleaned data
df = pd.read_csv('cleaned_gujarat_data.csv')

# App Title
st.title('🏠 Gujarat Cost of Living Assistant')

# User Inputs
st.sidebar.header('Your Preferences')
salary = st.sidebar.slider('Monthly Salary (₹)', 15000, 50000, 25000)
aqi_threshold = st.sidebar.slider('Max AQI Tolerance', 50, 150, 100)
priority = st.sidebar.radio('Priority', ['Affordability', 'Quality of Life'])

# Filter Logic
if priority == 'Affordability':
    recommendations = df[
        (df['Total Monthly Cost'] <= salary * 0.6) & 
        (df['AQI'] <= aqi_threshold)
    ].sort_values('Affordability Score', ascending=False)
else:
    recommendations = df[
        (df['Total Monthly Cost'] <= salary * 0.6) & 
        (df['AQI'] <= aqi_threshold)
    ].sort_values('Green Cover (%)', ascending=False)

# Display Results
st.header('Recommended Districts')
st.write(f"Showing districts within ₹{salary*0.6:,.0f} monthly budget:")

if not recommendations.empty:
    cols = st.columns(2)
    
    with cols[0]:
        st.subheader('Top 3 Options')
        for i, row in recommendations.head(3).iterrows():
            st.markdown(f"""
            **{row['District']}**  
            💵 Cost: ₹{row['Total Monthly Cost']:,.0f}  
            🌿 Green Cover: {row['Green Cover (%)']}%  
            😷 AQI: {row['AQI']}  
            🔑 Affordability Score: {row['Affordability Score']:.1f}  
            """)
    
    with cols[1]:
        st.subheader('Detailed Comparison')
        st.dataframe(recommendations[['District', 'Total Monthly Cost', 'AQI', 'Green Cover (%)']])
        
    st.image('cost_comparison.png')
else:
    st.warning("No districts match your criteria. Try adjusting your budget or AQI tolerance.")