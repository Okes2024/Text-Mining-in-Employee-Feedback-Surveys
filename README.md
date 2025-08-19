Text-Mining-in-Employee-Feedback-Surveys
📌 Project Overview

This project performs text mining and sentiment analysis on synthetic employee feedback data. It demonstrates how Natural Language Processing (NLP) techniques can be applied to understand workplace feedback, extract insights, and identify patterns in employee sentiments.

The workflow includes:

Generating synthetic employee feedback data (>100 records)

Preprocessing (tokenization, stopword removal, lemmatization)

Sentiment analysis using VADER

TF-IDF keyword extraction

Clustering with KMeans

Topic modeling using LDA (gensim)

Visualizations: wordclouds, sentiment distribution, clusters, and topic summaries

📂 Project Structure
Text-Mining-in-Employee-Feedback-Surveys/
│
├── text_mining_employee_feedback.py    # Main script
├── synthetic_employee_feedback.xlsx    # Example synthetic dataset
├── /text_mining_feedback/              # Output directory
│   ├── plots/                          # Generated visualizations
│   ├── synthetic_employee_feedback_clean.csv
│   ├── synthetic_employee_feedback_with_keywords.csv
│   ├── synthetic_employee_feedback_with_topics.csv
│   ├── synthetic_employee_feedback_results.xlsx
│   ├── sentiment_summary.csv
│   ├── cluster_summary.csv
│   ├── lda_topics.txt
│   └── analysis_summary.txt

⚙️ Installation

Clone this repository and install dependencies:

pip install numpy pandas matplotlib seaborn scikit-learn nltk gensim wordcloud openpyxl


Ensure NLTK resources are downloaded in your script:

import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("vader_lexicon")

🚀 Usage

Run the main script to generate synthetic data, process feedback, and perform full text-mining analysis:

python text_mining_employee_feedback.py


The results (datasets, plots, and reports) will be saved in:

/mnt/data/text_mining_feedback/

📊 Key Outputs

Sentiment Distribution: Pie/bar charts showing positive, neutral, and negative feedback.

WordCloud: Common keywords from employee feedback.

KMeans Clusters: Feedback grouped by textual similarity.

t-SNE Visualization: 2D scatter plot of clustered feedback.

LDA Topics: Automatically extracted themes from feedback.

Excel Report: Cleaned data with keywords, clusters, and topic labels.

👨‍💻 Author

Okes Imoni

GitHub: Okes2024
