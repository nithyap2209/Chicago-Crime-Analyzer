import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
from sklearn.utils.multiclass import unique_labels

# Set the option to suppress the pyplot global use warning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Return a copy of the DataFrame to prevent mutation between runs
    return df.copy()

# Data Cleaning
@st.cache(allow_output_mutation=True)
def clean_data(df):
    df_copy = df.copy()  # Make a copy of the DataFrame
    fill_mode_columns = [
        'Location Description', 'Ward', 'Community Area', 'X Coordinate', 
        'Y Coordinate', 'Latitude', 'Longitude', 'Location'
    ]
    for col in fill_mode_columns:
        df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
    return df_copy

# Temporal Analysis
def temporal_analysis(df):
    crime_trends = df.groupby('Year').size()
    st.write("### Number of Crimes Per Year in Chicago")
    st.bar_chart(crime_trends)

# Peak Crime Hours
def peak_crime_hours(df):
    df['Hour'] = pd.to_datetime(df['Date']).dt.hour
    peak_hours = df.groupby('Hour').size()
    st.write("### Crime Frequency by Hour in Chicago")
    st.line_chart(peak_hours)

# Geospatial Analysis
def geospatial_analysis(df):
    map_hooray = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(map_hooray)
    st.write("### Crime Hotspots in Chicago")
    folium_static(map_hooray)

# Crime Rates by District
def district_crime_rates(df):
    district_crime_counts = df.groupby(['District', 'Primary Type']).size().unstack()
    st.write("### Crime Rates by District")
    st.bar_chart(district_crime_counts)

# Distribution of Crime Types
def crime_type_distribution(df):
    crime_type_counts = df['Primary Type'].value_counts()
    st.write("### Distribution of Crime Types")
    st.bar_chart(crime_type_counts)

# Severity Analysis
def severity_analysis(df):
    severe_crimes = ['HOMICIDE', 'ASSAULT', 'BATTERY', 'ROBBERY', 'CRIM SEXUAL ASSAULT']
    df['Severity'] = df['Primary Type'].apply(lambda x: 'Severe' if x in severe_crimes else 'Less Severe')
    severity_counts = df['Severity'].value_counts()
    # Create a pie chart using matplotlib
    labels = severity_counts.index
    sizes = severity_counts.values
    fig, ax = plt.subplots(figsize=(6, 6))  # Create a Matplotlib figure
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.set_title('Severity Analysis of Crimes')
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.write("### Severity Analysis of Crimes")
    st.pyplot(fig)  # Pass the figure object to st.pyplot()

# Modular Streamlit App
def main():
    st.title("Chicago Crime Analysis")
    file_path = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if file_path is not None:
        df = load_data(file_path)

        st.sidebar.subheader("Data Cleaning")
        cleaned_df = clean_data(df)

        st.sidebar.subheader("Select Analysis")
        analysis_option = st.sidebar.selectbox(
            "Choose an analysis",
            ("Temporal Analysis", "Peak Crime Hours", "Geospatial Analysis", "District Crime Rates",
             "Crime Type Distribution", "Severity Analysis")
        )

        if analysis_option == "Temporal Analysis":
            temporal_analysis(cleaned_df)
        elif analysis_option == "Peak Crime Hours":
            peak_crime_hours(cleaned_df)
        elif analysis_option == "Geospatial Analysis":
            geospatial_analysis(cleaned_df)
        elif analysis_option == "District Crime Rates":
            district_crime_rates(cleaned_df)
        elif analysis_option == "Crime Type Distribution":
            crime_type_distribution(cleaned_df)
        elif analysis_option == "Severity Analysis":
            severity_analysis(cleaned_df)

if __name__ == "__main__":
    main()
