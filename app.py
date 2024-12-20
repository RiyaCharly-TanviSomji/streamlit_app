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
                background-color: #f0f0f0;
                color: #000000;
                font-family: Arial, sans-serif;
            }
            .css-1v3fvcr {
                background-color: #f0f0f0;
            }
            .css-ffhzg2 {
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

    # Load the data
    data = load_data()

    if page == "data_overview":
        render_data_overview(data)

    elif page == "eda":
        render_eda(data)

    elif page == "modeling":
        render_modeling(data)


# Render Data Overview Page
def render_data_overview(data):
    st.title("Dataset Overview")
    st.write("### Dataset Information")
    st.write(data.info())
    st.write("### First Few Rows of Data")
    st.write(data.head())
    st.write("### Dataset Summary Statistics")
    st.write(data.describe())


# Render Exploratory Data Analysis Page
def render_eda(data):
    st.title("Exploratory Data Analysis")

    # Year range selection
    st.write("### Filter Data by Year Range (2013 - 2017)")
    year_range = st.slider("Select Year Range", 2013, 2017, (2013, 2017))
    filtered_data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]

    # Show filtered data
    st.write(f"Showing Data from {year_range[0]} to {year_range[1]}")
    st.write(filtered_data)

    # PM2.5 Distribution
    st.write("### Distribution of PM2.5")
    visualize_pm25_distribution(data)

    # PM2.5 vs Temperature Hexbin Plot
    st.write("### PM2.5 vs Temperature (Hexbin Visualization)")
    visualize_pm25_vs_temp(data)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    visualize_correlation_heatmap(filtered_data)


# Visualizations for EDA
def visualize_pm25_distribution(data):
    bins = [0, 25, 50, 75, 100, 150, 200, 300]
    labels = ['0-25', '25-50', '50-75', '75-100', '100-150', '150-200', '200-300']
    data['PM2.5_Binned'] = pd.cut(data['PM2.5'], bins=bins, labels=labels)
    bin_counts = data['PM2.5_Binned'].value_counts(sort=False)

    plt.figure(figsize=(8, 6))
    bin_counts.plot(kind='bar', color='steelblue', edgecolor='black', alpha=0.85)
    plt.title('PM2.5 Levels Distribution', fontsize=16)
    plt.xlabel('PM2.5 Bins (µg/m³)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3, linestyle='--', axis='y')
    plt.tight_layout()
    st.pyplot()


def visualize_pm25_vs_temp(data):
    if data[['PM2.5', 'TEMP']].isnull().any().any():
        data['PM2.5'].fillna(data['PM2.5'].mean(), inplace=True)
        data['TEMP'].fillna(data['TEMP'].mean(), inplace=True)

    plt.figure(figsize=(10, 6))
    plt.hexbin(data['TEMP'], data['PM2.5'], gridsize=30, cmap='coolwarm', mincnt=1)
    cb = plt.colorbar()
    cb.set_label('Counts')
    plt.title("PM2.5 vs Temperature (Hexbin Visualization)", fontsize=16)
    plt.xlabel("Temperature (°C)", fontsize=14)
    plt.ylabel("PM2.5 (µg/m³)", fontsize=14)
    plt.grid(alpha=0.3)
    st.pyplot()


def visualize_correlation_heatmap(filtered_data):
    numeric_data = filtered_data.select_dtypes(include=['number'])
    corr_matrix = numeric_data.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="viridis", cbar=True, linewidths=0.5, square=True)
    plt.title('Correlation Heatmap', fontsize=16)
    plt.tight_layout()
    st.pyplot()


# Render Modeling Page
def render_modeling(data):
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

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Linear Regression
    st.write("### Linear Regression")
    train_linear_regression(X_train, X_test, y_train, y_test)

    # Gradient Boosting Regressor
    st.write("### Gradient Boosting Regressor")
    train_gradient_boosting(X_train, X_test, y_train, y_test)


# Train and Visualize Models
def train_linear_regression(X_train, X_test, y_train, y_test):
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    lr_mse = mean_squared_error(y_test, y_pred)
    lr_r2 = r2_score(y_test, y_pred)
    st.write(f"Mean Squared Error: {lr_mse:.4f}")
    st.write(f"R-squared: {lr_r2:.4f}")

    visualize_actual_vs_predicted(y_test, y_pred, "Linear Regression")
        # Residuals Plot for Linear Regression
    residuals = y_test - y_pred
    plt.figure(figsize=(12, 7))
    sns.histplot(residuals, kde=True, bins=30, color='orchid', edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='darkred', linestyle='--', linewidth=2, label='Zero Residual Line')
    plt.title('Residuals Distribution', fontsize=18, fontweight='bold', color='navy')
    plt.xlabel('Residuals (y_test - y_pred)', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    st.pyplot()




def train_gradient_boosting(X_train, X_test, y_train, y_test):
    gb_model = GradientBoostingRegressor(random_state=42)
    gb_model.fit(X_train, y_train)
    gb_pred = gb_model.predict(X_test)
    gb_mse = mean_squared_error(y_test, gb_pred)
    gb_r2 = r2_score(y_test, gb_pred)
    st.write(f"Mean Squared Error: {gb_mse:.4f}")
    st.write(f"R-squared: {gb_r2:.4f}")
    

    visualize_actual_vs_predicted(y_test, gb_pred, "Gradient Boosting")

        # Gradient Boosting: Residuals Plot
    gb_residuals = y_test - gb_pred
    plt.figure(figsize=(12, 7))
    sns.residplot(x=gb_pred, y=gb_residuals, lowess=True, scatter_kws={'alpha': 0.7, 'color': 'blue'}, line_kws={'color': 'red', 'linewidth': 2})
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='Zero Line')
    plt.title('Gradient Boosting: Residuals Plot', fontsize=18, fontweight='bold')
    plt.xlabel('Predicted Values (gb_pred)', fontsize=14)
    plt.ylabel('Residuals (y_test - gb_pred)', fontsize=14)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(alpha=0.4)
    plt.tight_layout()
    st.pyplot()


def visualize_actual_vs_predicted(y_test, y_pred, title):
    plt.figure(figsize=(12, 7))
    sns.regplot(x=y_test, y=y_pred, scatter_kws={'color': 'teal', 'alpha': 0.6, 's': 80},
                line_kws={'color': 'gold', 'linewidth': 2}, ci=None)
    plt.title(f'{title}: Actual vs Predicted Values', fontsize=18, fontweight='bold', color='navy')
    plt.xlabel('Actual Values (y_test)', fontsize=14)
    plt.ylabel('Predicted Values (y_pred)', fontsize=14)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    st.pyplot()


# Entry Point
if __name__ == "__main__":
    main()
