# Software Defect Prediction Visualization App 📊

## What's This All About?

Hey there! 👋 This project is about using machine learning (specifically, Logistic Regression for now) to try and predict which parts of software code (like files or functions, often called 'modules') might have bugs *before* testing finds them.

Finding bugs early saves time and money! This Streamlit application provides an interactive dashboard to:

1.  Load software metric data (like code complexity, lines of code, etc.).
2.  Train a simple defect prediction model.
3.  Visualize how well the model performs.
4.  Explore the data and the model's results in detail.
5.  Understand the steps involved in building such a prediction model.

Think of it as a visual tool to peek into the world of predicting software defects using data!

## 🤔 Why is this Helpful?

Predicting software defects using machine learning offers several key advantages:

* **Save Time & Reduce Costs:** By identifying potentially buggy modules early, development teams can focus inspection and testing efforts where they are most needed, catching defects before they become expensive problems later in the cycle or after release.
* **Prioritize Testing Efforts:** Test teams can use the defect predictions to prioritize their test cases. Running tests covering high-risk modules first can lead to finding critical bugs faster.
* **Improve Code Quality Insights:** Analyzing which code metrics (features) are strong predictors of defects can provide valuable feedback to developers about coding practices or architectural choices that might lead to more bugs.
* **Resource Allocation:** Helps managers allocate quality assurance resources more effectively, focusing effort on the modules most likely to cause issues.
* **Educational Tool:** This interactive app makes it easier to understand the concepts behind machine learning for defect prediction, data preprocessing, model evaluation, and the impact of different parameters (like the classification threshold).

Essentially, this tool helps make the software development and testing process more efficient and data-driven.

## ✨ Key Features

* **Interactive Dashboard:** Built with Streamlit for an easy-to-use web interface.
* **Data Loading:** Load your own software metric dataset (CSV format).
* **Data Exploration:** See sample data, basic statistics, and feature correlations.
* **Automated Preprocessing:** Handles missing values (using median imputation) and scales features (using StandardScaler).
* **Model Training:** Trains a Logistic Regression model with settings helpful for imbalanced data (`class_weight='balanced'`).
* **Performance Visualization:**
    * Confusion Matrix (See True Positives, False Negatives, etc.)
    * Classification Report (Precision, Recall, F1-Score)
    * ROC Curve & AUC Score
    * Precision-Recall Curve & AUC Score
* **Threshold Tuning:** Interactively adjust the classification threshold and see its impact on metrics in real-time.
* **Feature Influence:** Shows which code metrics the model found most important (via coefficients).
* **Detailed Report:** Provides a step-by-step explanation of the entire process, from loading data to evaluation, tailored to your specific run.

## 🚀 Getting Started: Setup & Running

Ready to try it out? Here’s how:

1.  **Prerequisites:**
    * Make sure you have **Python** installed (version 3.7 or higher is recommended). You can get it from [python.org](https://www.python.org/downloads/).
    * You'll need `pip` (Python's package installer), which usually comes with Python.

2.  **Install Libraries:**
    Open your terminal or command prompt and run this command to install all the necessary Python packages:
    ```bash
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Get the Code:**
    * Save the Streamlit application code provided (the Python script) as a file named `streamlit_app.py` (or any `.py` name you like) on your computer.

4.  **Get a Dataset:**
    * You need a dataset of software metrics in **CSV format**.
    * A common example is the `cm1.csv` dataset from the NASA/PROMISE repository. You can search online to download it (e.g., search for "download cm1 dataset nasa promise csv").
    * Place the downloaded CSV file somewhere you can easily find it (like the same folder as `streamlit_app.py`, or your Downloads folder).

5.  **Run the App:**
    * Open your terminal or command prompt again.
    * Navigate (`cd`) to the directory where you saved `streamlit_app.py`.
    * Run the following command:
        ```bash
        streamlit run streamlit_app.py
        ```

6.  **View in Browser:**
    * Streamlit should automatically open the application in your default web browser. If not, it will display a local URL (like `http://localhost:8501`) in the terminal – just copy and paste that into your browser.

## 💻 How to Use the App

1.  **Configure:** Use the sidebar on the left to:
    * Enter the correct **path** to your CSV dataset file.
    * Make sure the **Target Column Name** matches the column in your CSV that indicates defects (e.g., `defects`, `bug`, `problems`). It often needs a value of 1 (or `true`/`yes`) for defective and 0 (or `false`/`no`) for non-defective.
2.  **Explore Data:** Look at the "Data Exploration" section to get a feel for your dataset.
3.  **Train:** Click the **"🚀 Train Model and Evaluate"** button. The app will process the data and train the Logistic Regression model.
4.  **Analyze Results:**
    * Check the **Preprocessing Summary** and **Model Performance** sections.
    * Play with the **Classification Threshold** slider to see how it affects the Confusion Matrix and Classification Report. This helps understand the trade-off between finding defects and avoiding false alarms.
    * Look at the **ROC** and **Precision-Recall curves** for an overall sense of model quality.
    * Examine the **Feature Influence** table to see which metrics mattered most to the model.
5.  **Read the Report:** Scroll down to the **"📝 Detailed Process Report"** for a full explanation of everything the app did during that run.

Enjoy exploring software defect prediction!
