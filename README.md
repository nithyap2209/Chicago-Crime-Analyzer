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


## Severity Analysis
    severe_crimes = ['HOMICIDE', 'ASSAULT', 'BATTERY', 'ROBBERY', 'CRIM SEXUAL ASSAULT']
    df['Severity'] = df['Primary Type'].apply(lambda x: 'Severe' if x in severe_crimes else 'Less Severe')
    severity_counts = df['Severity'].value_counts()
    plt.figure(figsize=(6, 6))
    severity_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90)
    plt.title('Severity Analysis of Crimes')
    plt.ylabel('')
    plt.show()

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


## Domestic vs. Non-Domestic Crimes
    domestic_counts = df['Domestic'].value_counts()
    plt.figure(figsize=(6, 6))
    domestic_counts.plot(kind='pie', labels=['Non-Domestic', 'Domestic'], autopct='%1.1f%%', startangle=90)
    plt.title('Domestic vs. Non-Domestic Crimes')
    plt.ylabel('')
    plt.show()
    


## Location-Specific Analysis
    location_counts = df['Location Description'].value_counts().head(10)
    plt.figure(figsize=(10, 6))
    location_counts.plot(kind='barh')
    plt.title('Top 10 Crime Locations')
    plt.xlabel('Frequency')
    plt.ylabel('Location Description')
    plt.tight_layout()
    plt.show()
    


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

