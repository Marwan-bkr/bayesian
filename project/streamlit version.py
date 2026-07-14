import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score,
    precision_score, recall_score, hamming_loss,
    multilabel_confusion_matrix
)

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="Movie Genre Predictor", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
@st.cache_data
def load_and_clean_data():
    # Load dataset. movie_metadata is a dataset from IMDB (https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset) mixed with ChatGPT generated data.
    try:
        df = pd.read_csv("movie_metadata.csv")
    except:
        st.error("Please ensure 'movie_metadata.csv' is in the directory.")
        return None, None, None, None

    df = df.dropna(subset=['genres'])
    
    # 1. Multi-Label Encoding
    df['genres_list'] = df['genres'].str.split('|')
    mlb = MultiLabelBinarizer()
    genres_encoded = mlb.fit_transform(df['genres_list'])
    genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_, index=df.index)
    
    # Frequency filtering
    MIN_GENRE_SAMPLES = 800
    genre_freq = genres_df.sum().sort_values(ascending=False)
    genres_to_keep = genre_freq[genre_freq >= MIN_GENRE_SAMPLES].index.tolist()
    y_multilabel = genres_df[genres_to_keep].values
    
    # 2. Features Selection & Imputation
    features_numeric = [
        'num_critic_for_reviews', 'duration', 'director_facebook_likes',
        'actor_3_facebook_likes', 'actor_1_facebook_likes', 'gross',
        'num_voted_users', 'cast_total_facebook_likes', 'facenumber_in_poster',
        'num_user_for_reviews', 'budget', 'title_year',
        'actor_2_facebook_likes', 'imdb_score', 'movie_facebook_likes'
    ]
    
    X_df = df[features_numeric].copy()
    for col in features_numeric:
        X_df[col].fillna(X_df[col].median(), inplace=True)
    
    # BERT-Simulation: In a real app, this would use a saved file or live inference.
    # We will simulate BERT encoding if bert_output_movie-title.csv is missing.
    try:
        bert_preds = pd.read_csv('bert_output_movie-title.csv')
        df_bert = pd.get_dummies(bert_preds['BERT_pred'], prefix='BERT_pred')
    except:
        # Mock BERT predictions for the demo
        mock_bert = np.random.choice(['Action', 'Drama', 'Comedy'], size=len(X_df))
        df_bert = pd.get_dummies(mock_bert, prefix='BERT_pred')
        
    X_combined = pd.concat([X_df.reset_index(drop=True), df_bert.reset_index(drop=True)], axis=1)
    
    return X_combined, y_multilabel, genres_to_keep, features_numeric

X_combined, y_multilabel, genre_names, feature_names = load_and_clean_data()

# ==========================================
# MODEL TRAINING
# ==========================================
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Models
    gnb = OneVsRestClassifier(GaussianNB())
    gnb.fit(X_train_scaled, y_train)
    
    bnb = OneVsRestClassifier(BernoulliNB())
    bnb.fit(X_train_scaled, y_train)
    
    return gnb, bnb, X_test_scaled, y_test, scaler

# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🎬 Movie Bayesian Predictor")
st.sidebar.info("Analyze and predict multi-label movie genres using Naive Bayes models and BERT features.")
app_mode = st.sidebar.selectbox("Navigate:", ["Dashboard & Stats", "Interactive Prediction"])

if X_combined is not None:
    gnb_model, bnb_model, X_test_scaled, y_test, scaler = train_models(X_combined.values, y_multilabel)

    # ------------------------------------------
    # PAGE 1: DASHBOARD
    # ------------------------------------------
    if app_mode == "Dashboard & Stats":
        st.title("📊 Project Analysis Dashboard")
        
        # Key Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Movies", len(X_combined))
        col2.metric("Target Genres", len(genre_names))
        col3.metric("Numerical Features", len(feature_names))
        col4.metric("Model Status", "✅ Trained")

        # Visualization Section
        tab1, tab2, tab3 = st.tabs(["Genre Distribution", "Model Comparison", "Performance by Genre"])
        
        with tab1:
            st.subheader("Distribution of Top Genres")
            genre_counts = pd.Series(y_multilabel.sum(axis=0), index=genre_names).sort_values()
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=genre_counts.values, y=genre_counts.index, palette="viridis", ax=ax)
            ax.set_title("Genre Frequency in IMDB Dataset")
            st.pyplot(fig)

        with tab2:
            st.subheader("Model Performance Comparison")
            # Predictions
            y_pred_gnb = gnb_model.predict(X_test_scaled)
            y_pred_bnb = bnb_model.predict(X_test_scaled)
            
            comparison_data = {
                'Metric': ['F1 (Micro)', 'Accuracy (Exact Match)', 'Hamming Loss'],
                'Gaussian NB': [
                    f1_score(y_test, y_pred_gnb, average='micro'),
                    accuracy_score(y_test, y_pred_gnb),
                    hamming_loss(y_test, y_pred_gnb)
                ],
                'Bernoulli NB': [
                    f1_score(y_test, y_pred_bnb, average='micro'),
                    accuracy_score(y_test, y_pred_bnb),
                    hamming_loss(y_test, y_pred_bnb)
                ]
            }
            st.table(pd.DataFrame(comparison_data).set_index('Metric'))

        with tab3:
            st.subheader("Detailed Genre Report (Gaussian NB)")
            report = classification_report(y_test, y_pred_gnb, target_names=genre_names, output_dict=True)
            report_df = pd.DataFrame(report).transpose().drop(['micro avg', 'macro avg', 'weighted avg', 'samples avg'])
            st.dataframe(report_df.style.background_gradient(cmap="YlGn"), height=400)

    # ------------------------------------------
    # PAGE 2: INTERACTIVE PREDICTION
    # ------------------------------------------
    elif app_mode == "Interactive Prediction":
        st.title("🔮 Predict Movie Genres")
        st.write("Adjust the features below to see which genres the Naive Bayes model predicts.")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Feature Selection")
            user_input = []
            
            # Interactive Sliders
            for feature in feature_names:
                # Dynamically set slider range based on dataset statistics
                min_val = float(X_combined[feature].min())
                max_val = float(X_combined[feature].max())
                median_val = float(X_combined[feature].median())
                
                val = st.slider(f"{feature.replace('_', ' ').title()}", 
                              min_value=min_val, 
                              max_value=max_val, 
                              value=median_val)
                user_input.append(val)
            
            # Since user can't manually set BERT predictions, we use the average (zeroing them out for neutral effect)
            bert_features = [0] * (X_combined.shape[1] - len(feature_names))
            final_vector = np.array(user_input + bert_features).reshape(1, -1)
            
        with col2:
            st.markdown("### Prediction Result")
            # Scale the input
            final_vector_scaled = scaler.transform(final_vector)
            
            # Predict
            pred_binary = gnb_model.predict(final_vector_scaled)[0]
            pred_proba = gnb_model.predict_proba(final_vector_scaled)
            
            # Display results
            predicted_genres = [genre_names[i] for i, val in enumerate(pred_binary) if val == 1]
            
            if not predicted_genres:
                st.warning("No specific genre predicted for these values. Try adjusting the sliders (e.g., higher budget or score).")
            else:
                st.success(f"**Predicted Genres:** {', '.join(predicted_genres)}")
            
            # Probability Chart
            st.markdown("#### Probability per Genre")

            # Corrected Probability Extraction
            # Some scikit-learn versions return a matrix directly for OneVsRest
            if isinstance(pred_proba, np.ndarray):
                # If it's a 2D array (1, n_genres), flatten it
                probas = pred_proba[0] 
            else:
                # If it's a list of arrays (one per estimator), extract class 1 prob
                probas = [p[0, 1] for p in pred_proba]

            prob_df = pd.DataFrame({
                'Genre': genre_names, 
                'Probability': probas
            }).sort_values('Probability', ascending=False)

            # Render the plot...
            
            fig, ax = plt.subplots(figsize=(8, 10))
            sns.barplot(x='Probability', y='Genre', data=prob_df, palette="crest", ax=ax)
            ax.set_xlim(0, 1)
            st.pyplot(fig)

# FOOTER
st.markdown("---")
st.caption("AI Model Collaboration • Built with Streamlit • login BERT Project 2026")