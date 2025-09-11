# file: text_mining_employee_feedback.py
# Purpose: Text-Mining-in-Employee-Feedback-Surveys (synthetic data)

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# NLP
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Topic modeling
import gensim
from gensim import corpora
from wordcloud import WordCloud

# Ensure necessary NLTK downloads
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("vader_lexicon")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

OUT_DIR = "/mnt/data/text_mining_feedback"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "plots"), exist_ok=True)

# -------- 1) Synthetic data generation --------
def generate_synthetic_feedback(n_samples=500):
    positives = [
        "I appreciate the supportive leadership and clear communication.",
        "Great teamwork and collaborative environment, I feel valued.",
        "Opportunities for growth and training are excellent.",
        "Flexible working hours and good work-life balance.",
        "Management recognises effort and rewards performance."
    ]
    negatives = [
        "Too much bureaucracy and slow decision-making.",
        "Workload is high and deadlines are unrealistic.",
        "Poor communication from leadership, unclear priorities.",
        "Lack of career advancement and inadequate training.",
        "Compensation is not competitive and benefits are weak."
    ]
    neutrals = [
        "I work on project tasks and attend regular meetings.",
        "Daily routine involves standard responsibilities and reports.",
        "Office resources are available but sometimes require updates.",
        "We have scheduled check-ins and team updates.",
        "Tasks are assigned based on project needs."
    ]

    departments = ["engineering","sales","hr","product","operations","it","finance","marketing"]
    themes = ["communication","work-life balance","leadership","training","compensation","tools","processes"]

    records = []
    for i in range(n_samples):
        r = random.random()
        if r < 0.4:
            template = random.choice(positives); sentiment = "positive"
        elif r < 0.8:
            template = random.choice(neutrals); sentiment = "neutral"
        else:
            template = random.choice(negatives); sentiment = "negative"
        dept = random.choice(departments)
        theme = random.choice(themes)
        mod = random.choice(["Overall, ","Frankly, ","In my experience, ","To be honest, ",""])
        example = random.choice([""," e.g., last quarter we had multiple last-minute requests.",
                                 " for instance during peak season."," especially during product launches.",
                                 " as seen in recent sprints."])
        feedback = f"{mod}{template} This is common in the {dept} team, often about {theme}.{example}"
        if random.random() < 0.25:
            extra = random.choice(positives + neutrals + negatives)
            feedback = f"{feedback} {extra}"
        records.append({"employee_id": f"E{1000+i}","department": dept,
                        "feedback": feedback,"sentiment_label": sentiment})
    return pd.DataFrame.from_records(records)

# -------- 2) Preprocessing --------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    text = str(text).lower()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t.isalpha() and t not in stop_words and len(t)>2]
    return tokens

# -------- 3) Analysis pipeline --------
def run_analysis(df):
    df["feedback_clean_tokens"] = df["feedback"].apply(preprocess_text)
    df["feedback_clean"] = df["feedback_clean_tokens"].apply(lambda toks: " ".join(toks))
    df.to_csv(os.path.join(OUT_DIR,"synthetic_employee_feedback_clean.csv"), index=False)

    sia = SentimentIntensityAnalyzer()
    df["vader_compound"] = df["feedback"].apply(lambda t: sia.polarity_scores(str(t))["compound"])
    df["vader_sentiment"] = df["vader_compound"].apply(lambda c: "positive" if c>=0.05 else ("negative" if c<=-0.05 else "neutral"))
    df["vader_sentiment"].value_counts().to_frame("count").to_csv(os.path.join(OUT_DIR,"sentiment_summary.csv"))

    plt.figure(figsize=(6,4))
    sns.countplot(x="vader_sentiment", data=df, order=["positive","neutral","negative"])
    plt.title("Sentiment Distribution (VADER)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,"plots","sentiment_distribution.png"), dpi=150)
    plt.close()

    vectorizer = TfidfVectorizer(max_df=0.8, min_df=2, ngram_range=(1,2), max_features=2000)
    X_tfidf = vectorizer.fit_transform(df["feedback_clean"].values)
    feature_names = np.array(vectorizer.get_feature_names_out())
    df["top_keywords"] = [feature_names[vec.toarray().ravel().argsort()[::-1][:8]].tolist() if vec.nnz>0 else [] for vec in X_tfidf]
    df.to_csv(os.path.join(OUT_DIR,"synthetic_employee_feedback_with_keywords.csv"), index=False)

    wc = WordCloud(width=800,height=400,background_color="white").generate(" ".join(df["feedback_clean"].tolist()))
    plt.figure(figsize=(10,5)); plt.imshow(wc, interpolation="bilinear"); plt.axis("off")
    plt.title("WordCloud - Employee Feedback (cleaned)")
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR,"plots","wordcloud_feedback.png"), dpi=150); plt.close()

    best_k,best_score,best_kmodel = None,-1,None
    for k in range(2,7):
        kmodel = KMeans(n_clusters=k,random_state=RANDOM_SEED,n_init=10)
        labels = kmodel.fit_predict(X_tfidf)
        score = silhouette_score(X_tfidf,labels) if len(set(labels))>1 else -1
        if score>best_score: best_score, best_k, best_kmodel = score, k, kmodel
    df["kcluster"] = best_kmodel.predict(X_tfidf)
    df.groupby("kcluster").feedback.count().to_frame("count").to_csv(os.path.join(OUT_DIR,"cluster_summary.csv"))

    svd = TruncatedSVD(n_components=50, random_state=RANDOM_SEED)
    X_svd = svd.fit_transform(X_tfidf)
    tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, n_iter=800, learning_rate=200)
    X_tsne = tsne.fit_transform(X_svd)
    df["tsne_1"], df["tsne_2"] = X_tsne[:,0], X_tsne[:,1]

    dictionary = corpora.Dictionary(df["feedback_clean_tokens"].tolist())
    dictionary.filter_extremes(no_below=3, no_above=0.6, keep_n=2000)
    corpus = [dictionary.doc2bow(text) for text in df["feedback_clean_tokens"].tolist()]
    lda = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=6, random_state=RANDOM_SEED, passes=10, iterations=100)
    with open(os.path.join(OUT_DIR,"lda_topics.txt"),"w") as f:
        for tid, topic in lda.print_topics(num_words=8): f.write(f"Topic {tid}: {topic}\n")

    df["lda_topic"] = [sorted(lda.get_document_topics(bow), key=lambda x:-x[1])[0][0] if lda.get_document_topics(bow) else -1 for bow in corpus]
    df.to_excel(os.path.join(OUT_DIR,"synthetic_employee_feedback_results.xlsx"), index=False)

    return df, lda

# -------- 4) Run --------
def main():
    df = generate_synthetic_feedback(500)
    run_analysis(df)

if __name__ == "__main__":
    main()
