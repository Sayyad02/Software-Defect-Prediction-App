# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# --- Page Configuration ---
st.set_page_config(
    page_title="Enhanced Software Defect Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "An enhanced Streamlit app for visualizing software defect prediction."
    }
)

# --- Initialize Session State ---
# This runs only once at the start or when the session restarts
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'results' not in st.session_state:
    st.session_state.results = None

# --- Configuration ---
st.sidebar.header("⚙️ Configuration")
# Allow user to specify path via sidebar - more flexible
dataset_path_default = 'C:/Users/Shafin ali/Downloads/cm1.csv' # Default path
dataset_path = st.sidebar.text_input("Enter Dataset Path (CSV):", dataset_path_default)

# Allow user to specify target column - crucial for different datasets
target_column_default = 'defects'
target_column = st.sidebar.text_input("Enter Target Column Name:", target_column_default)

# --- Caching Functions for Performance ---

# Cache data loading
@st.cache_data
def load_data(path):
    """Loads data from a CSV file."""
    try:
        df = pd.read_csv(path)
        # Basic check for empty dataframe
        if df.empty:
            st.error("ERROR: Loaded dataset is empty.")
            st.stop()
        # Don't show success message here, show it in main logic
        # st.success(f"Dataset loaded successfully from: {path}")
        return df
    except FileNotFoundError:
        st.error(f"ERROR: Dataset file not found at '{path}'")
        # Reset model trained state if file changes and fails
        st.session_state.model_trained = False
        st.session_state.results = None
        st.stop() # Stop execution if file not found
    except pd.errors.EmptyDataError:
         st.error(f"ERROR: The file at '{path}' is empty.")
         st.session_state.model_trained = False
         st.session_state.results = None
         st.stop()
    except Exception as e:
        st.error(f"ERROR: Failed to load data: {e}")
        st.session_state.model_trained = False
        st.session_state.results = None
        st.stop() # Stop execution on other errors

# Cache model training and evaluation
# We pass necessary parameters to ensure cache updates when they change
@st.cache_data
def train_evaluate_model(_df, target_col_name, test_size=0.3, random_state=42):
    """Preprocesses data, trains Logistic Regression, and returns results."""
    # Create a copy to avoid modifying the cached original df
    df = _df.copy()
    results = {} # Dictionary to store results

    # 1. Separate Features (X) and Target (y)
    if target_col_name not in df.columns:
        # Use st.error within the function for cached errors
        results['error'] = f"Target column '{target_col_name}' not found."
        results['available_columns'] = df.columns.tolist()
        return results # Return error info

    X = df.drop(target_col_name, axis=1)
    y = df[target_col_name]
    results['original_feature_names'] = X.columns.tolist() # Store original feature names
    results['original_shape'] = df.shape

    # 2. Ensure Target Variable is Numeric (0 or 1)
    original_dtype = y.dtype
    results['target_original_dtype'] = str(original_dtype) # Store original dtype as string
    conversion_applied = False
    if y.dtype == 'bool':
        y = y.astype(int)
        conversion_applied = True
    elif y.dtype == 'object':
        unique_vals = y.unique()
        # Standardize case for mapping
        y_str = y.astype(str).str.lower() # Use a temporary variable
        unique_vals_lower = y_str.unique()

        if 'yes' in unique_vals_lower and 'no' in unique_vals_lower:
            y = y_str.map({'yes': 1, 'no': 0})
            conversion_applied = True
        elif 'true' in unique_vals_lower and 'false' in unique_vals_lower:
            y = y_str.map({'true': 1, 'false': 0})
            conversion_applied = True
        else:
            try:
                y_numeric = pd.to_numeric(y) # Try conversion
                # Check if conversion actually changed the type (e.g., from '1' string to 1 int)
                if not pd.api.types.is_numeric_dtype(y_numeric):
                     raise ValueError("Conversion did not result in numeric type")
                y = y_numeric
                conversion_applied = True # Mark as converted if successful
            except ValueError:
                # Only raise error if it wasn't already numeric looking (e.g. strings '0', '1')
                 if not all(val.isdigit() for val in unique_vals if isinstance(val, str)):
                    results['error'] = f"Target column '{target_col_name}' (dtype: {original_dtype}) contains non-numeric values that couldn't be automatically converted: {unique_vals}"
                    return results
                 else: # If they were numeric strings like '0', '1', convert them
                     try:
                         y = y.astype(int)
                         conversion_applied = True
                     except ValueError: # Should not happen if isdigit passed, but safety check
                         results['error'] = f"Failed final conversion attempt for target column: {unique_vals}"
                         return results

        if y.isnull().any():
            results['error'] = f"Conversion of target column '{target_col_name}' resulted in missing values. Check for unexpected values (e.g., empty strings, NaN strings)."
            return results
        y = y.astype(int) # Ensure integer type

    results['target_conversion_applied'] = conversion_applied
    results['target_distribution'] = y.value_counts(normalize=True)

    # 3. Select Only Numeric Features
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    non_numeric_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    results['non_numeric_cols_ignored'] = non_numeric_cols if non_numeric_cols else None

    if len(numeric_cols) == 0:
        results['error'] = "No numeric features found!"
        return results
    elif len(numeric_cols) != X.shape[1]:
        # Store warning info instead of using st.warning inside cache
        results['warning_non_numeric'] = f"Ignoring non-numeric columns: {non_numeric_cols}"
        X = X[numeric_cols]
    results['numeric_features_used'] = numeric_cols

    # 4. Check for Missing Values
    missing_values = X.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    handle_missing = not missing_values.empty
    results['missing_values_info'] = missing_values if handle_missing else "None"
    results['imputation_strategy'] = 'median' if handle_missing else None # Store strategy used

    # 5. Split Data
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        results['split_info'] = {
            'test_size': test_size,
            'random_state': random_state,
            'stratify': True, # Stratification was used
            'train_shape': (X_train.shape, y_train.shape),
            'test_shape': (X_test.shape, y_test.shape),
            'train_defect_rate': y_train.mean(),
            'test_defect_rate': y_test.mean()
        }
    except ValueError as e:
         results['error'] = f"ERROR during train_test_split: {e}. Final unique values in target 'y': {y.unique()}"
         return results

    # 6. Preprocessing Pipeline (Imputation & Scaling)
    # Imputation
    imputer_fitted = None # Store the fitted imputer
    if handle_missing:
        imputer = SimpleImputer(strategy=results['imputation_strategy'])
        # Fit on train, transform train and test
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        imputer_fitted = imputer # Store the fitted imputer
        # Convert back to DataFrame to keep column names
        X_train = pd.DataFrame(X_train_imputed, columns=numeric_cols, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=numeric_cols, index=X_test.index)
    results['imputation_applied'] = handle_missing
    results['imputer'] = imputer_fitted # Store fitted imputer or None

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    results['scaling_applied'] = True
    results['scaler'] = scaler # Store the fitted scaler

    # Store processed test data for evaluation
    results['X_test_scaled'] = X_test_scaled
    results['y_test'] = y_test


    # 7. Train Model
    model = LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000)
    results['model_type'] = 'Logistic Regression'
    results['model_params'] = model.get_params() # Store model parameters
    model.fit(X_train_scaled, y_train)
    results['model'] = model # Store the trained model

    # 8. Evaluate Model (get base predictions and probabilities)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    # y_pred calculation depends on threshold, will be done outside this cached function
    results['y_pred_proba'] = y_pred_proba

    # Calculate Metrics that don't depend on threshold
    results['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred_proba)
    results['pr_curve_data'] = {'precision': precision, 'recall': recall, 'thresholds': thresholds_pr}
    results['pr_auc'] = auc(recall, precision)

    # Store coefficients
    if hasattr(model, 'coef_'):
         # Ensure numeric_features_used matches the columns X_train_scaled was based on
         results['coefficients'] = pd.DataFrame(
             model.coef_[0],
             index=results['numeric_features_used'],
             columns=['Coefficient']
         ).sort_values('Coefficient', ascending=False)
    else:
        results['coefficients'] = None


    # Indicate success if no error key was added
    if 'error' not in results:
        results['success'] = True

    return results

# --- Main App Logic ---
st.title("📊 Enhanced Software Defect Prediction Dashboard")
st.markdown("""
Welcome! This interactive dashboard trains a **Logistic Regression** model to predict software defects
using code metrics. Explore the data, evaluate the model, and see how changing the classification
threshold impacts performance. A detailed report of the process is generated below the results.
""")

# Load data based on user input path
df_loaded = load_data(dataset_path)

if df_loaded is not None:
    st.success(f"Dataset loaded successfully from: {dataset_path}") # Show success message here
    st.header("1. Data Exploration")
    col1, col2 = st.columns([1, 2]) # Adjust column widths if needed
    with col1:
        st.markdown(f"**Dataset Shape:**")
        st.markdown(f"{df_loaded.shape[0]} rows, {df_loaded.shape[1]} columns")
        st.markdown(f"**Target Column:**")
        st.markdown(f"`{target_column}`")
        if target_column in df_loaded:
            if df_loaded[target_column].dtype == 'object' or df_loaded[target_column].dtype == 'bool':
                st.write(f"*Original Data Type:* {df_loaded[target_column].dtype} (will be converted to 0/1)")
            st.write("**Target Distribution (Original):**")
            # Ensure target column exists before value_counts
            if target_column in df_loaded.columns:
                st.dataframe(df_loaded[target_column].value_counts(normalize=True).reset_index().rename(columns={'index':'Value', target_column:'Proportion'}))
            else:
                st.warning(f"Target column '{target_column}' not found in data.")
        else:
             st.warning(f"Target column '{target_column}' not found yet. Check name.")


    with col2:
        st.markdown("**Sample Data:**")
        st.dataframe(df_loaded.head())

    with st.expander("Explore Features Further"):
        st.markdown("#### Descriptive Statistics (Numeric Features)")
        # Ensure target column exists before dropping
        if target_column in df_loaded.columns:
            numeric_features = df_loaded.select_dtypes(include=np.number).drop(columns=[target_column], errors='ignore')
            if not numeric_features.empty:
                 st.dataframe(numeric_features.describe())
            else:
                 st.info("No numeric features found (excluding target).")
        else:
            # Select all numeric if target is not yet valid
            numeric_features = df_loaded.select_dtypes(include=np.number)
            if not numeric_features.empty:
                 st.dataframe(numeric_features.describe())
                 st.caption("(Target column not identified yet, showing all numeric)")
            else:
                 st.info("No numeric features found.")


        st.markdown("#### Feature Correlation Heatmap")
        st.markdown("_Correlation shows linear relationships between numeric features._")
        # Calculate correlation on numeric columns only
        numeric_df_for_corr = df_loaded.select_dtypes(include=np.number)
        if not numeric_df_for_corr.empty and len(numeric_df_for_corr.columns) > 1:
            corr = numeric_df_for_corr.corr()
            fig_corr, ax_corr = plt.subplots(figsize=(10, 8)) # Adjust size as needed
            sns.heatmap(corr, annot=False, cmap='coolwarm', fmt=".2f", ax=ax_corr) # annot=False for cleaner look with many features
            ax_corr.set_title("Correlation Matrix of Numeric Features")
            st.pyplot(fig_corr)
        elif len(numeric_df_for_corr.columns) <= 1:
             st.info("Need more than one numeric feature for correlation analysis.")
        else:
            st.info("No numeric features available for correlation analysis.")


    # --- Train and Evaluate ---
    st.header("2. Model Training & Evaluation")
    st.markdown("Click the button below to preprocess the data and train the Logistic Regression model.")

    # Use a button to trigger potentially long-running training
    if st.button("🚀 Train Model and Evaluate", key="train_button"):
        # Check again if target column is valid before training
        if target_column not in df_loaded.columns:
            st.error(f"Cannot train: Target column '{target_column}' not found in the loaded data.")
            st.stop()

        # Run the cached function
        # Pass df_loaded copy to ensure cache works correctly if df_loaded is mutated elsewhere (though it shouldn't be here)
        with st.spinner("Training model... Please wait."): # Add spinner for feedback
            results_dict = train_evaluate_model(df_loaded.copy(), target_column)

        # Check for errors returned from the cached function
        if 'error' in results_dict:
             st.error(f"Training Failed: {results_dict['error']}")
             if 'available_columns' in results_dict:
                  st.info(f"Available columns: {results_dict['available_columns']}")
             st.session_state.model_trained = False
             st.session_state.results = None
             st.stop()
        else:
            # Store results in session state upon successful training
            st.session_state.results = results_dict
            st.session_state.model_trained = True
            st.success("Model trained successfully!")
             # Display any warnings from training
            if 'warning_non_numeric' in results_dict:
                 st.warning(results_dict['warning_non_numeric'])


    # --- Display Results Area (Only if model has been trained successfully) ---
    if st.session_state.model_trained and st.session_state.results:
        results = st.session_state.results # Get results from session state

        st.subheader("Preprocessing Summary")
        col1, col2 = st.columns(2)
        with col1:
             st.write(f"**Numeric Features Used:** {len(results['numeric_features_used'])}")
             # Display first 5 features used as an example
             st.caption(f"e.g., {results['numeric_features_used'][:5]}...")
             st.write(f"**Imputation Applied:** {'Yes (' + results['imputation_strategy'] + ')' if results['imputation_applied'] else 'No'}")
             if results['imputation_applied'] and isinstance(results['missing_values_info'], pd.Series):
                  st.dataframe(results['missing_values_info'].astype(str), width=300) # Show which cols were imputed
             st.write(f"**Scaling Applied:** {'Yes (StandardScaler)' if results['scaling_applied'] else 'No'}")
        with col2:
             st.write("**Data Split:**")
             st.write(f"- Training: {results['split_info']['train_shape'][0]} samples ({results['split_info']['train_defect_rate']:.2%} defects)")
             st.write(f"- Testing: {results['split_info']['test_shape'][0]} samples ({results['split_info']['test_defect_rate']:.2%} defects)")
             st.write(f"- Test Size: {results['split_info']['test_size']:.0%}")
             st.write(f"- Stratified: {'Yes' if results['split_info']['stratify'] else 'No'}")


        st.subheader("📈 Model Performance on Test Set")

        # --- Threshold Tuning (Now outside the button 'if') ---
        st.markdown("#### Classification Threshold Tuning")
        st.markdown("""
        Adjust the slider below to change the probability threshold used to classify a module as defective (1) or not (0).
        The default is 0.5. Observe how changing the threshold impacts the trade-off between finding defects (Recall)
        and incorrectly flagging non-defects (False Positives reflected in Precision).
        """)
        # Get probabilities from results stored in session state
        y_pred_proba = results['y_pred_proba']
        y_test = results['y_test']

        # Threshold slider
        threshold = st.slider("Select Classification Threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="threshold_slider")

        # Apply threshold to get new predictions
        y_pred_tuned = (y_pred_proba >= threshold).astype(int)

        # --- Display Metrics based on Tuned Threshold ---
        col_met1, col_met2 = st.columns(2)

        with col_met1:
            st.markdown(f"**Metrics at Threshold = {threshold:.2f}**")
            # Calculate and display confusion matrix for the selected threshold
            cm_tuned = confusion_matrix(y_test, y_pred_tuned)
            tn, fp, fn, tp = cm_tuned.ravel()

            fig_cm, ax_cm = plt.subplots(figsize=(5, 4)) # Smaller figure size
            sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Predicted Non-Defective', 'Predicted Defective'],
                        yticklabels=['Actual Non-Defective', 'Actual Defective'])
            ax_cm.set_title(f'Confusion Matrix (Threshold = {threshold:.2f})')
            ax_cm.set_ylabel('Actual Label')
            ax_cm.set_xlabel('Predicted Label')
            st.pyplot(fig_cm)
            st.caption(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        with col_met2:
            st.markdown(f"**Classification Report (Threshold = {threshold:.2f})**")
            # Use zero_division=0 to avoid warnings when a class has no predictions/actuals at extreme thresholds
            report_tuned = classification_report(y_test, y_pred_tuned, target_names=['Non-Defective (0)', 'Defective (1)'], output_dict=True, zero_division=0)
            report_df_tuned = pd.DataFrame(report_tuned).transpose()
            st.dataframe(report_df_tuned)
            st.caption("Note: Precision, Recall, F1 for 'Defective (1)' are often key.")


        # --- Display Threshold-Independent Metrics and Curves ---
        st.markdown("---") # Separator
        st.markdown("**Threshold-Independent Metrics & Curves**")
        col_curve1, col_curve2 = st.columns(2)

        with col_curve1:
            st.metric("AUC-ROC", f"{results['roc_auc']:.4f}")
            st.caption("Area Under the ROC Curve - Model's ability to distinguish classes.")
             # ROC Curve
            model = results['model']
            X_test_scaled = results['X_test_scaled']
            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax_roc, name='Logistic Regression')
            ax_roc.set_title('ROC Curve')
            ax_roc.plot([0, 1], [0, 1], 'k--', label='No Skill (AUC=0.5)') # Add diagonal line
            ax_roc.legend()
            st.pyplot(fig_roc)

        with col_curve2:
            st.metric("AUC-PR", f"{results['pr_auc']:.4f}")
            st.caption("Area Under the Precision-Recall Curve - More informative for imbalance.")
            # Precision-Recall Curve
            fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
            pr_data = results['pr_curve_data']
            # Use PrecisionRecallDisplay or plot manually if needed
            PrecisionRecallDisplay(precision=pr_data['precision'], recall=pr_data['recall']).plot(ax=ax_pr, name='Logistic Regression')
            # Add No Skill line (Precision = Positive Class Proportion)
            no_skill = y_test.mean()
            ax_pr.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill (AUC={no_skill:.2f})')
            ax_pr.set_title('Precision-Recall Curve')
            ax_pr.legend()
            st.pyplot(fig_pr)


        # --- Feature Importance (Coefficients) ---
        if 'coefficients' in results and results['coefficients'] is not None:
            st.subheader("💡 Feature Influence (Model Coefficients)")
            st.markdown("""
            Logistic Regression coefficients indicate the estimated change in the log-odds of the outcome (defect)
            for a one-unit change in the corresponding feature, assuming other features are held constant.
            Features have been scaled (StandardScaler), so coefficients are comparable.
            * **Positive coefficients:** Increase in feature value increases the predicted probability of a defect.
            * **Negative coefficients:** Increase in feature value decreases the predicted probability of a defect.
            """)
            st.dataframe(results['coefficients'])
        else:
            st.info("Coefficient data not available for this model type.")

        # --- DETAILED REPORT SECTION (ENHANCED EXPLANATIONS) ---
        st.header("📝 Detailed Process Report")
        st.markdown("This report summarizes the steps taken during the data processing and model training workflow, explaining the purpose of each stage.")

        with st.expander("1. Data Loading & Initial Setup", expanded=False):
             st.markdown(f"""
             - **Dataset Loaded:** Data was successfully loaded from the specified path (`{dataset_path}`). This is the foundational step, bringing the raw data into the application for analysis.
             - **Initial Shape:** The dataset initially contained **{results['original_shape'][0]}** rows and **{results['original_shape'][1]}** columns. Understanding the dimensions helps gauge the scale of the data.
             - **Target Column Identified:** The column `{target_column}` was designated as the target variable. This is the variable the model aims to predict (in this case, whether a software module is defective or not).
             - **Target Data Type:** The original data type of the target column was `{results['target_original_dtype']}`. Machine learning models typically require numerical targets for classification.
             - **Target Conversion:** {'Conversion to numeric (0/1) was applied.' if results['target_conversion_applied'] else 'No conversion needed, target was already numeric.'}
                 - *Why:* If the target was boolean (`True`/`False`) or object (`'yes'`/`'no'`, `'true'`/`'false'`), it needed conversion to integers (1/0) for the model to process it correctly.
             """)

        with st.expander("2. Feature Preparation", expanded=False):
             st.markdown(f"""
             - **Feature Selection Rationale:** For this baseline model, only features with numeric data types were selected as inputs (predictors). Real-world projects might involve more complex feature engineering, including handling categorical data.
             - **Numeric Features Used ({len(results['numeric_features_used'])}):** `{', '.join(results['numeric_features_used'][:15])}{'...' if len(results['numeric_features_used']) > 15 else ''}` These are the specific metrics (like lines of code, complexity scores) used to predict defects.
             - **Non-Numeric Features Ignored ({len(results['non_numeric_cols_ignored']) if results['non_numeric_cols_ignored'] else 0}):** {'`' + ', '.join(results['non_numeric_cols_ignored']) + '`' if results['non_numeric_cols_ignored'] else 'None'}
                 - *Why:* Non-numeric columns (e.g., text descriptions, IDs) were excluded in this simple approach. Techniques like One-Hot Encoding would be needed to incorporate categorical features meaningfully.
             - **Missing Value Handling:**
                 - Missing values {'were' if results['imputation_applied'] else 'were not'} detected in the numeric features used.
                 - {'Imputation Strategy: **' + results['imputation_strategy'] + '** imputation was applied.' if results['imputation_applied'] else 'No imputation was necessary.'}
                     - *Why:* Most machine learning algorithms cannot process missing data (NaNs). Imputation fills these gaps. Median imputation was chosen because it's less sensitive to outliers compared to mean imputation. The imputer was fitted *only* on the training data to prevent data leakage from the test set.
             - **Feature Scaling:**
                 - Scaling was applied using **StandardScaler**.
                 - *Why:* StandardScaler transforms each feature to have a mean of 0 and a standard deviation of 1. This is crucial for algorithms like Logistic Regression, which use distance calculations or gradient descent. Scaling ensures that features with larger values (e.g., lines of code) don't disproportionately influence the model compared to features with smaller values (e.g., complexity ratios). The scaler was also fitted *only* on the training data.
             """)

        with st.expander("3. Data Splitting", expanded=False):
             split_info = results['split_info']
             st.markdown(f"""
             - **Method:** The dataset was divided into two distinct sets: one for training the model and one for evaluating its performance.
                 - *Why:* This is fundamental in machine learning to assess how well the model generalizes to new, unseen data. Training and evaluating on the same data would lead to an overly optimistic and misleading performance estimate (overfitting).
             - **Test Set Size:** **{split_info['test_size']:.0%}** of the data was allocated to the test set. This is a common split ratio, providing enough data for reliable evaluation while retaining a larger portion for training.
             - **Training Set:** {split_info['train_shape'][0]} samples used *exclusively* to teach the model the patterns between features and the target variable.
             - **Test Set:** {split_info['test_shape'][0]} samples kept separate and used *only* after training to evaluate the model's predictive power on data it has never encountered before.
             - **Stratification:** The split was stratified based on the target variable (`{target_column}`).
                 - *Why:* In datasets where one class is much rarer than the other (like defect prediction, where defects are often rare), simple random splitting might result in a test set with very few or even zero instances of the minority class. Stratification ensures the proportion of defects ({split_info['train_defect_rate']:.2%}) vs. non-defects in the training set mirrors the proportion in the test set ({split_info['test_defect_rate']:.2%}), leading to a more representative evaluation.
             - **Random State:** A fixed random state (`{split_info['random_state']}`) was used.
                 - *Why:* This ensures that the *exact same* split is generated every time the code runs with the same data and parameters. This is essential for reproducibility, allowing others (or yourself later) to obtain identical results.
             """)

        with st.expander("4. Model Training", expanded=False):
             st.markdown(f"""
             - **Model Type:** A **{results['model_type']}** classifier was selected and trained.
                 - *Why Logistic Regression:* It's a standard, relatively simple, and interpretable algorithm for binary classification tasks, making it a good starting point (baseline) for defect prediction.
             - **Training Process:** The model learned the relationship between the preprocessed features (imputed and scaled) from the training set and the corresponding target labels (defective/non-defective).
             - **Key Parameters:**
                 - `class_weight='balanced'`: *Why:* Since defect data is often imbalanced (fewer defects than non-defects), this parameter tells the algorithm to give more importance to misclassifying the minority (defective) class during training. This encourages the model to identify defects better, often improving Recall for the defective class, which is usually desired.
                 - `random_state={results['model_params']['random_state']}`: *Why:* Ensures that the internal random processes within the Logistic Regression algorithm (like solver initialization) are the same each time, contributing to reproducible results.
                 - `max_iter={results['model_params']['max_iter']}`: *Why:* Specifies the maximum number of iterations the optimization algorithm will run. Increasing this can help the model converge if it doesn't with the default setting.
             - **Training Data:** Crucially, the model was trained *only* on the **training set** features and labels. The test set was kept completely separate during this phase.
             """)

        with st.expander("5. Model Evaluation", expanded=False):
             st.markdown(f"""
             - **Evaluation Data:** The performance of the *trained* model was assessed using the **test set**, which the model did not see during training.
                 - *Why:* Evaluating on the test set provides an unbiased estimate of how well the model is likely to perform on new, real-world data.
             - **Evaluation Metrics Explained:**
                 - **Confusion Matrix:** A table summarizing the prediction results.
                     - *True Negatives (TN):* Correctly predicted non-defective.
                     - *False Positives (FP):* Incorrectly predicted as defective (Type I error). *Cost:* Wasted inspection effort.
                     - *False Negatives (FN):* Incorrectly predicted as non-defective (Type II error). *Cost:* Undetected defects escaping to later stages or production (often high cost).
                     - *True Positives (TP):* Correctly predicted as defective.
                 - **Classification Report:** Provides key metrics derived from the confusion matrix:
                     - *Precision (Defective):* `TP / (TP + FP)`. High precision means fewer false alarms when predicting defects.
                     - *Recall (Defective):* `TP / (TP + FN)`. High recall means the model finds most of the actual defects. **Often the most critical metric in defect prediction**, as missing defects (high FN) can be costly.
                     - *F1-Score (Defective):* `2 * (Precision * Recall) / (Precision + Recall)`. A balance between Precision and Recall. Useful if both minimizing false alarms and finding defects are important.
                 - **AUC-ROC ({results['roc_auc']:.4f}):** Area Under the ROC Curve. Represents the model's ability to rank a randomly chosen defective module higher than a randomly chosen non-defective one. An AUC of 0.5 is random guessing, 1.0 is perfect discrimination. It summarizes performance across all thresholds.
                 - **AUC-PR ({results['pr_auc']:.4f}):** Area Under the Precision-Recall Curve. Particularly useful for imbalanced datasets because it focuses on the performance concerning the minority (defective) class. A high AUC-PR indicates good performance even when defects are rare. The 'No Skill' line represents random guessing based on the class distribution.
             - **Threshold Tuning:** The classification threshold determines the cutoff probability for predicting a defect (default is 0.5).
                 - *Why Tune:* Adjusting this threshold allows trading off Precision and Recall. Lowering the threshold typically increases Recall (finds more defects) but decreases Precision (more false alarms). Raising it does the opposite. The optimal threshold depends on the relative costs of missing defects (FN) versus investigating false alarms (FP).
             - **Feature Influence (Coefficients):** For linear models like Logistic Regression, coefficients indicate the relationship between each feature and the predicted outcome (log-odds of defect).
                 - *Why Examine:* Helps understand *which* code metrics are most strongly associated with defects according to the model, providing insights for developers and potentially guiding process improvements. Positive coefficients suggest higher metric values increase defect likelihood; negative suggests the opposite.
             """)


    # Display message if button hasn't been clicked yet
    elif not st.session_state.model_trained:
        st.info("Click the '🚀 Train Model and Evaluate' button to start the analysis and generate the report.")


# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("Adjust settings and click the button to run the analysis.")

