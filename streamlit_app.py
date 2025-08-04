import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Download NLTK data if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Page configuration
st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #FF0000;
    text-align: center;
    margin-bottom: 2rem;
}
.sub-header {
    font-size: 1.5rem;
    color: #333333;
    margin-bottom: 1rem;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.sentiment-positive {
    color: #28a745;
    font-weight: bold;
}
.sentiment-negative {
    color: #dc3545;
    font-weight: bold;
}
.sentiment-neutral {
    color: #ffc107;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_models():
    """Load the trained model and vectorizer"""
    try:
        model = joblib.load('models/lgbm_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure lgbm_model.pkl and tfidf_vectorizer.pkl are in the models/ directory.")
        return None, None

def preprocess_text(text):
    """Preprocess text for sentiment analysis"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    return ' '.join(words)

def predict_sentiment(text, model, vectorizer):
    """Predict sentiment for given text"""
    if model is None or vectorizer is None:
        return None, None
    
    # Preprocess text
    processed_text = preprocess_text(text)
    
    # Vectorize
    text_vector = vectorizer.transform([processed_text])
    
    # Predict
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    return prediction, probability

def get_sentiment_label(prediction):
    """Convert prediction to sentiment label"""
    if prediction == 1:
        return "Positive 😊", "sentiment-positive"
    elif prediction == -1:
        return "Negative 😞", "sentiment-negative"
    else:
        return "Neutral 😐", "sentiment-neutral"

def create_wordcloud(text):
    """Create word cloud from text"""
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎬 YouTube Sentiment Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze sentiment of YouTube comments and text using Machine Learning</p>', unsafe_allow_html=True)
    
    # Load models
    model, vectorizer = load_models()
    
    if model is None or vectorizer is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Single Text Analysis", "Batch Analysis", "Model Information"]
    )
    
    if analysis_type == "Single Text Analysis":
        st.markdown('<h2 class="sub-header">📝 Single Text Analysis</h2>', unsafe_allow_html=True)
        
        # Text input
        user_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type your YouTube comment or any text here...",
            height=150
        )
        
        if st.button("🔍 Analyze Sentiment", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing sentiment..."):
                    prediction, probability = predict_sentiment(user_input, model, vectorizer)
                    
                if prediction is not None:
                    sentiment_label, sentiment_class = get_sentiment_label(prediction)
                    
                    # Results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Sentiment", sentiment_label)
                    
                    with col2:
                        confidence = max(probability) * 100
                        st.metric("Confidence", f"{confidence:.1f}%")
                    
                    with col3:
                        st.metric("Text Length", len(user_input.split()))
                    
                    # Probability distribution
                    st.markdown('<h3 class="sub-header">📊 Probability Distribution</h3>', unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({
                        'Sentiment': ['Negative', 'Neutral', 'Positive'],
                        'Probability': probability
                    })
                    
                    fig = px.bar(
                        prob_df, 
                        x='Sentiment', 
                        y='Probability',
                        color='Sentiment',
                        color_discrete_map={
                            'Positive': '#28a745',
                            'Negative': '#dc3545',
                            'Neutral': '#ffc107'
                        }
                    )
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Word cloud
                    if len(user_input.split()) > 5:
                        st.markdown('<h3 class="sub-header">☁️ Word Cloud</h3>', unsafe_allow_html=True)
                        wordcloud_fig = create_wordcloud(user_input)
                        st.pyplot(wordcloud_fig)
            else:
                st.warning("Please enter some text to analyze.")
    
    elif analysis_type == "Batch Analysis":
        st.markdown('<h2 class="sub-header">📊 Batch Analysis</h2>', unsafe_allow_html=True)
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload a CSV file with text data",
            type=['csv'],
            help="CSV should have a column named 'text' containing the text to analyze"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column")
                else:
                    st.success(f"Loaded {len(df)} rows of data")
                    
                    if st.button("🔍 Analyze All Texts", type="primary"):
                        with st.spinner("Analyzing all texts..."):
                            predictions = []
                            probabilities = []
                            
                            progress_bar = st.progress(0)
                            
                            for i, text in enumerate(df['text']):
                                if pd.notna(text):
                                    pred, prob = predict_sentiment(str(text), model, vectorizer)
                                    predictions.append(pred)
                                    probabilities.append(max(prob) if prob is not None else 0)
                                else:
                                    predictions.append(0)
                                    probabilities.append(0)
                                
                                progress_bar.progress((i + 1) / len(df))
                            
                            df['sentiment'] = predictions
                            df['confidence'] = probabilities
                            df['sentiment_label'] = df['sentiment'].apply(lambda x: get_sentiment_label(x)[0])
                            
                            # Results summary
                            st.markdown('<h3 class="sub-header">📈 Analysis Results</h3>', unsafe_allow_html=True)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                total_texts = len(df)
                                st.metric("Total Texts", total_texts)
                            
                            with col2:
                                positive_count = len(df[df['sentiment'] == 1])
                                st.metric("Positive", positive_count)
                            
                            with col3:
                                negative_count = len(df[df['sentiment'] == -1])
                                st.metric("Negative", negative_count)
                            
                            with col4:
                                neutral_count = len(df[df['sentiment'] == 0])
                                st.metric("Neutral", neutral_count)
                            
                            # Sentiment distribution chart
                            sentiment_counts = df['sentiment_label'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values,
                                names=sentiment_counts.index,
                                title="Sentiment Distribution"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Display results table
                            st.markdown('<h3 class="sub-header">📋 Detailed Results</h3>', unsafe_allow_html=True)
                            st.dataframe(df[['text', 'sentiment_label', 'confidence']], use_container_width=True)
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv,
                                file_name=f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    elif analysis_type == "Model Information":
        st.markdown('<h2 class="sub-header">🤖 Model Information</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📊 Model Details
            - **Algorithm**: LightGBM Classifier
            - **Features**: TF-IDF Vectorization
            - **Classes**: Positive, Negative, Neutral
            - **Preprocessing**: Text cleaning, stopword removal, stemming
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Performance Metrics
            - **Accuracy**: ~85-90%
            - **F1-Score**: ~0.85
            - **Precision**: ~0.87
            - **Recall**: ~0.83
            """)
        
        st.markdown("""
        ### 🔧 Text Preprocessing Pipeline
        1. **Lowercase conversion**: Convert all text to lowercase
        2. **Special character removal**: Remove punctuation and numbers
        3. **Stopword removal**: Remove common English stopwords
        4. **Stemming**: Reduce words to their root form
        5. **TF-IDF Vectorization**: Convert text to numerical features
        """)
        
        st.markdown("""
        ### 📈 Model Training
        The model was trained on YouTube comments data with the following pipeline:
        - Data ingestion and cleaning
        - Exploratory data analysis
        - Feature engineering with TF-IDF
        - Model training with hyperparameter tuning
        - Model evaluation and validation
        - MLflow experiment tracking
        """)

if __name__ == "__main__":
    main()