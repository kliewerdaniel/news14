from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import logging

from src.core.models import Article
from src.core.config import CONFIG

class ArticleClusterer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def cluster_articles_tfidf(self, articles: List[Article]):
        """Fast TF-IDF based clustering"""
        if len(articles) < 2:
            return

        texts = [f"{article.title} {article.summary}" for article in articles]

        try:
            vectorizer = TfidfVectorizer(max_features=50, stop_words='english')
            embeddings = vectorizer.fit_transform(texts).toarray()

            n_clusters = min(CONFIG["processing"]["max_clusters"], len(articles))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(embeddings)

            for article, label in zip(articles, cluster_labels):
                article.cluster_id = int(label)

        except Exception as e:
            self.logger.error(f"Clustering error: {e}")
            for i, article in enumerate(articles):
                article.cluster_id = i % 3
