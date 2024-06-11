# Chicago-Crime-Analyzer
1.Objective: To leverage historical and recent crime data to identify patterns, trends, and hotspots within Chicago, supporting strategic decision-making, resource allocation, and crime prevention strategies.

2.Dataset Description: The dataset contains records of reported crimes in Chicago, including various attributes such as ID, case number, date, crime type, location, arrest status, etc.

3.Goals:
* Temporal Analysis: Analyze crime trends over time and identify peak crime hours.
* Geospatial Analysis: Identify crime hotspots and analyze crime rates across different districts and wards.
* Crime Type Analysis: Analyze the distribution and severity of different crime types.
* Arrest and Domestic Incident Analysis: Investigate arrest rates and compare domestic versus non-domestic incidents.
* Location-Specific Analysis: Analyze crime locations and patterns by location description, beat, and community area.
* Seasonal and Weather Impact: Examine seasonal trends and their impact on crime occurrences.
* Repeat Offenders and Recidivism: Identify repeat crime locations and analyze recidivism rates.
* Predictive Modeling and Risk Assessment: Develop predictive models and assess the risk of different areas and times for specific crimes.
* Visualization and Reporting: Create interactive dashboards and detailed reports for stakeholders.

4. Potential Insights: Trends, high-risk areas, effectiveness of law enforcement, patterns of domestic-related crimes, common locations for specific crimes, etc.

5. Deliverables:
* Visualizations (e.g., heatmaps, line charts, bar charts) representing temporal and spatial patterns.
* Statistical summaries and correlation analyses.
* Detailed report with findings and recommendations.
* Interactive dashboards for real-time monitoring of crime data.
## Imports
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import folium
    from folium.plugins import HeatMap
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
    from sklearn.utils.multiclass import unique_labels

## Load the dataset
    df = pd.read_csv(r"C:\Users\Selvam\crime analyzer\Sample Crime Dataset - Sheet1.csv")
    
    df

## Display the first few rows and dataset info
    print(df.head())
    print(df.info())
    print(df.isnull().sum())
    


## Fill missing values with the mode
    fill_mode_columns = [
        'Location Description', 'Ward', 'Community Area', 'X Coordinate', 
        'Y Coordinate', 'Latitude', 'Longitude', 'Location'
    ]
    for col in fill_mode_columns:
        df[col].fillna(df[col].mode()[0], inplace=True)


## Display dataset info after filling missing values
    print(df.info())


## Save the cleaned dataset
    df.to_csv('cleaned_dataset.csv', index=False)

## Temporal Analysis - Crime Trends Over Time
    crime_trends = df.groupby('Year').size()
    plt.figure(figsize=(12, 6))
    crime_trends.plot(kind='bar')
    plt.title('Number of Crimes Per Year in Chicago')
    plt.xlabel('Year')
    plt.ylabel('Number of Crimes')
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/bf676ae4-6f14-451a-a70c-86addcd1c337)

## Peak Crime Hours
    df['Hour'] = pd.to_datetime(df['Date']).dt.hour
    peak_hours = df.groupby('Hour').size()
    plt.figure(figsize=(12, 6))
    peak_hours.plot(kind='line')
    plt.title('Crime Frequency by Hour in Chicago')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Number of Crimes')
    plt.xticks(range(0, 24))
    plt.grid(True)
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/b4b28117-2208-4265-b19e-353f46fb3184)


## Geospatial Analysis - Crime Hotspots
    map_hooray = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=11)
    heat_data = [[row['Latitude'], row['Longitude']] for index, row in df.iterrows()]
    HeatMap(heat_data).add_to(map_hooray)
    map_hooray.save("crime_hotspots.html")  # Save the map as an HTML file


## Crime Rates by District
    district_crime_counts = df.groupby(['District', 'Primary Type']).size().unstack()
    district_crime_counts.plot(kind='bar', stacked=True, figsize=(14, 7))
    plt.title('Crime Rates by District')
    plt.xlabel('District')
    plt.ylabel('Number of Crimes')
    plt.legend(title='Primary Type', bbox_to_anchor=(1.05, 1))
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/c0150f28-5512-4e00-bb8a-7aa7c4f52886)



## Distribution of Crime Types
    crime_type_counts = df['Primary Type'].value_counts()
    plt.figure(figsize=(10, 6))
    crime_type_counts.plot(kind='bar')
    plt.title('Distribution of Crime Types')
    plt.xlabel('Crime Type')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/8424eb86-0db4-4ab5-b52d-5834335960f7)


## Severity Analysis
    severe_crimes = ['HOMICIDE', 'ASSAULT', 'BATTERY', 'ROBBERY', 'CRIM SEXUAL ASSAULT']
    df['Severity'] = df['Primary Type'].apply(lambda x: 'Severe' if x in severe_crimes else 'Less Severe')
    severity_counts = df['Severity'].value_counts()
    plt.figure(figsize=(6, 6))
    severity_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Severity Analysis of Crimes')
    plt.ylabel('')
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/e7d72d45-0119-4f75-a0d7-61732bcc4c2b)

## Arrest and Domestic Incident Analysis
    arrest_rates_by_type = df.groupby('Primary Type')['Arrest'].mean() * 100
    arrest_rates_by_type.sort_values(ascending=False, inplace=True)
    plt.figure(figsize=(10, 6))
    arrest_rates_by_type.plot(kind='bar')
    plt.title('Arrest Rates by Crime Type')
    plt.xlabel('Crime Type')
    plt.ylabel('Arrest Rate (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/4619aec0-b98d-4e57-a8ff-dc8be305c7b4)


## Domestic vs. Non-Domestic Crimes
    domestic_counts = df['Domestic'].value_counts()
    plt.figure(figsize=(6, 6))
    domestic_counts.plot(kind='pie', labels=['Non-Domestic', 'Domestic'], autopct='%1.1f%%', startangle=90)
    plt.title('Domestic vs. Non-Domestic Crimes')
    plt.ylabel('')
    plt.show()
    
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/b596b648-d4b0-426c-aabb-15d1c187af8f)


## Location-Specific Analysis
    location_counts = df['Location Description'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    location_counts.plot(kind='barh')
    plt.title('Top 10 Crime Locations')
    plt.xlabel('Frequency')
    plt.ylabel('Location Description')
    plt.tight_layout()
    plt.show()
    
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/a9ebf76c-40a2-4ae2-8607-aa10f2782b57)


## Crime Types by Location
    location_crime_types = df.groupby(['Location Description', 'Primary Type']).size().unstack()
    top_locations = location_counts.index[:5]
    location_crime_types.loc[top_locations].plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title('Crime Types by Location')
    plt.xlabel('Location Description')
    plt.ylabel('Frequency')
    plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/f5ada1bd-3fee-4532-a992-d8a7dfaa157e)


## Comparison by Beat and Community Area
    beat_crime_counts = df['Beat'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    beat_crime_counts.plot(kind='bar')
    plt.title('Top 10 Beats with Most Crimes')
    plt.xlabel('Beat')
    plt.ylabel('Number of Crimes')
    plt.tight_layout()
    plt.show()
    
    community_crime_counts = df['Community Area'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    community_crime_counts.plot(kind='bar')
    plt.title('Top 10 Community Areas with Most Crimes')
    plt.xlabel('Community Area')
    plt.ylabel('Number of Crimes')
    plt.tight_layout()
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/7f8cd938-d864-48d0-9152-774e82068570)

![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/4df4e3da-ccb5-45d7-8c75-9893d7d861c0)

## Seasonal and Weather Impact
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['Season'] = df['Month'].apply(get_season)
    seasonal_crime_counts = df.groupby(['Season', 'Primary Type']).size().unstack()
    seasonal_crime_counts.plot(kind='bar', figsize=(12, 6), stacked=True)
    plt.title('Seasonal Trends in Crime Types')
    plt.xlabel('Season')
    plt.ylabel('Number of Crimes')
    plt.legend(title='Crime Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/29882c5a-b735-467c-8156-c224b2365267)


## Repeat Offenders and Recidivism
    repeat_locations = df['Block'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=repeat_locations.values, y=repeat_locations.index, palette='viridis')
    plt.title('Top 10 Repeat Crime Locations', fontsize=16)
    plt.xlabel('Number of Incidents', fontsize=14)
    plt.ylabel('Block', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/c6b69a5a-5005-478f-83d7-e824165f9536)

## Create dataset for model
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    
    df1 = df.groupby(['Month', 'Day', 'District', 'Hour'], as_index=False).agg({"Primary Type": "count"})
    df1 = df1.sort_values(by=['District'], ascending=False)
    df1 = df1[['Month', 'Day', 'Hour', 'Primary Type', 'District']]
    
    def crime_rate_assign(x):
        if x <= 7:
            return 0
        elif 7 < x <= 15:
            return 1
        else:
            return 2
    
    df1['Alarm'] = df1['Primary Type'].apply(crime_rate_assign)
    df1 = df1[['Month', 'Day', 'Hour', 'District', 'Primary Type', 'Alarm']]
    print(df1.head())


## Decision Tree Classifier
    X = df1[['Month', 'Day', 'Hour', 'District']]
    y = df1['Alarm']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    d_tree = DecisionTreeClassifier(random_state=10)
    d_tree.fit(X_train, y_train)
    
    plt.figure(figsize=(12, 12))
    plot_tree(d_tree, feature_names=X.columns, filled=True, rounded=True)
    plt.show()
    
    y_pred = d_tree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%\n")
    
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm, index=unique_labels(y_test, y_pred), columns=unique_labels(y_test, y_pred))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()
    
    print("\n---------- Confusion Matrix ----------")
    print(cm)
    
    # Classification Report
    print("\n---------- Classification Report ----------")
    print(classification_report(y_test, y_pred))
    
    # Unweighted Average Recall (UAR)
    recall_per_class = recall_score(y_test, y_pred, average=None)
    uar = recall_per_class.mean()
    print(f"\nUnweighted Average Recall (UAR): {uar:.2f}")
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/630a99ea-e4f2-4fcb-9b10-0eaf5f54fb4f)
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/11f9acb6-768d-4ceb-a7ba-2dfc8013ade4)

---------- Confusion Matrix ----------
[[170]]

---------- Classification Report ----------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       170

    accuracy                           1.00       170
   macro avg       1.00      1.00      1.00       170
weighted avg       1.00      1.00      1.00       170


Unweighted Average Recall (UAR): 1.00

## Support Vector Classifier
    svc_model = SVC(kernel='linear', random_state=1)
    svc_model.fit(X_train, y_train)
    y_pred_svc = svc_model.predict(X_test)
    
    # Model Evaluation for SVC
    accuracy_svc = accuracy_score(y_test, y_pred_svc)
    print(f"Support Vector Classifier Accuracy: {accuracy_svc * 100:.2f}%\n")
    
    # Confusion Matrix for SVC
    cm_svc = confusion_matrix(y_test, y_pred_svc)
    cm_svc_df = pd.DataFrame(cm_svc, index=unique_labels(y_test, y_pred_svc), columns=unique_labels(y_test, y_pred_svc))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_svc_df, annot=True, fmt='d', cmap='Greens')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Support Vector Classifier')
    plt.show()
    
    # Classification Report for SVC
    print("\n---------- Support Vector Classifier Classification Report ----------")
    print(classification_report(y_test, y_pred_svc))
    
    # Unweighted Average Recall (UAR) for SVC
    recall_per_class_svc = recall_score(y_test, y_pred_svc, average=None)
    uar_svc = recall_per_class_svc.mean()
    print(f"\nSupport Vector Classifier Unweighted Average Recall (UAR): {uar_svc:.2f}")
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/f276ebda-b93f-4b1d-9c9b-4a45c8bab1b1)

---------- Support Vector Classifier Classification Report ----------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       170

    accuracy                           1.00       170
   macro avg       1.00      1.00      1.00       170
weighted avg       1.00      1.00      1.00       170


Support Vector Classifier Unweighted Average Recall (UAR): 1.00
## Logistic Regression
    logistic_model = LogisticRegression(random_state=1, max_iter=200)
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    
    # Model Evaluation for Logistic Regression
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    print(f"Logistic Regression Accuracy: {accuracy_logistic * 100:.2f}%\n")
    
    # Confusion Matrix for Logistic Regression
    cm_logistic = confusion_matrix(y_test, y_pred_logistic)
    cm_logistic_df = pd.DataFrame(cm_logistic, index=unique_labels(y_test, y_pred_logistic), columns=unique_labels(y_test, y_pred_logistic))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_logistic_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.show()
    
    # Classification Report for Logistic Regression
    print("\n---------- Logistic Regression Classification Report ----------")
    print(classification_report(y_test, y_pred_logistic))
    
    # Unweighted Average Recall (UAR) for Logistic Regression
    recall_per_class_logistic = recall_score(y_test, y_pred_logistic, average=None)
    uar_logistic = recall_per_class_logistic.mean()
    print(f"\nLogistic Regression Unweighted Average Recall (UAR): {uar_logistic:.2f}")
![image](https://github.com/nithyap2209/Chicago-Crime-Analyzer/assets/92367257/201824b8-9e7a-4131-be5b-efd52b72abe2)
---------- Logistic Regression Classification Report ----------
              precision    recall  f1-score   support

           0       1.00      1.00      1.00       170

    accuracy                           1.00       170
   macro avg       1.00      1.00      1.00       170
weighted avg       1.00      1.00      1.00       170


Logistic Regression Unweighted Average Recall (UAR): 1.00


# STREAMLIT PART 
        import streamlit as st
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import folium
        from streamlit_folium import folium_static
        from folium.plugins import HeatMap
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
        
