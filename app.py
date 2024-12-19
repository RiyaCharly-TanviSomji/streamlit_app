import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

# Load the dataset
def load_data():
    data = pd.read_csv("datasets/merged_dataset.csv")

    # Fill missing values only in numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

    return data

# Render navigation bar
def render_navbar():
    st.markdown(
        """
        <style>
            .navbar {
                background-color: #f8f9fa;
                padding: 1rem;
                border-bottom: 2px solid #dee2e6;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .navbar a {
                color: #007bff;
                text-decoration: none;
                font-size: 1.2rem;
                margin: 0 1rem;
            }
            .navbar a:hover {
                color: #0056b3;
                text-decoration: underline;
            }
            body {
                background-color: #f0f0f0; /* Light grey background */
                color: #000000; /* Black text color */
                font-family: Arial, sans-serif;
            }
            .css-1v3fvcr {  /* Custom CSS selector for Streamlit elements */
                background-color: #f0f0f0;
            }
            .css-ffhzg2 {  /* Custom CSS selector for Streamlit elements */
                color: #000000;
            }
        </style>
        <div class="navbar">
            <a href="/?page=data_overview">Data Overview</a>
            <a href="/?page=eda">Exploratory Data Analysis</a>
            <a href="/?page=modeling">Modeling and Prediction</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# Main function
def main():
    st.set_page_config(page_title="Beijing Air Quality Analysis", layout="wide")

    # Render the navigation bar
    render_navbar()

    # Get the current page from query parameters
    query_params = st.experimental_get_query_params()
    page = query_params.get("page", ["data_overview"])[0]

    data = load_data()

    if page == "data_overview":
        st.title("Dataset Overview")
        st.write("### Dataset Information")
        st.write(data.info())
        st.write("### First Few Rows of Data")
        st.write(data.head())

        # Show summary statistics
        st.write("### Dataset Summary Statistics")
        st.write(data.describe())

    elif page == "eda":
        st.title("Exploratory Data Analysis")

        # Year range selection (2013 - 2017)
        st.write("### Filter Data by Year Range (2013 - 2017)")
        year_range = st.slider("Select Year Range", 2013, 2017, (2013, 2017))

        # Filter data by the selected year range
        filtered_data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]

        # Show filtered data
        st.write(f"Showing Data from {year_range[0]} to {year_range[1]}")
        st.write(filtered_data)

        # PM2.5 Distribution
        st.write("### Distribution of PM2.5")
        plt.figure(figsize=(10, 6))
        sns.histplot(filtered_data['PM2.5'], kde=True, bins=40, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('PM2.5 Concentration Distribution', fontsize=16)
        plt.xlabel('PM2.5 Levels', fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot()

        # PM2.5 vs Temperature Hexbin Plot
        st.write("### PM2.5 vs Temperature (Hexbin Visualization)")
        plt.figure(figsize=(10, 6))
        plt.hexbin(filtered_data['TEMP'], filtered_data['PM2.5'], gridsize=30, cmap='coolwarm', mincnt=1)
        cb = plt.colorbar()
        cb.set_label('Counts')
        plt.title("PM2.5 vs Temperature (Hexbin Visualization)", fontsize=16)
        plt.xlabel("Temperature (°C)", fontsize=14)
        plt.ylabel("PM2.5 (µg/m³)", fontsize=14)
        plt.grid(alpha=0.3)
        st.pyplot()

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        numeric_data = filtered_data.select_dtypes(include=['number'])
        corr_matrix = numeric_data.corr()

        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5, square=True)
        plt.title('Correlation Heatmap', fontsize=16)
        plt.tight_layout()
        st.pyplot()

    elif page == "modeling":
        st.title("Modeling and Prediction")

        # Features and target
        features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'RAIN', 'WSPM']
        target = 'PM2.5'

        X = data[features]
        y = data[target]
        X = X.select_dtypes(include=['float64', 'int64'])

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Linear Regression
        st.write("### Linear Regression")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_mse = mean_squared_error(y_test, lr_pred)
        lr_r2 = r2_score(y_test, lr_pred)
        st.write(f"Mean Squared Error: {lr_mse:.4f}")
        st.write(f"R-squared: {lr_r2:.4f}")

        # Gradient Boosting Regressor
        st.write("### Gradient Boosting Regressor")
        gb_model = GradientBoostingRegressor(random_state=42)
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_mse = mean_squared_error(y_test, gb_pred)
        gb_r2 = r2_score(y_test, gb_pred)
        st.write(f"Mean Squared Error: {gb_mse:.4f}")
        st.write(f"R-squared: {gb_r2:.4f}")

        # Actual vs Predicted Values (Gradient Boosting)
        st.write("### Gradient Boosting: Actual vs Predicted Values")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, gb_pred, color='blue', alpha=0.6, edgecolors='k', label='Predictions')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal Fit')
        plt.title('Gradient Boosting: Actual vs Predicted Values', fontsize=16)
        plt.xlabel('Actual Values (y_test)', fontsize=14)
        plt.ylabel('Predicted Values (gb_pred)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(alpha=0.3)
        st.pyplot()

if __name__ == "__main__":
    main()
