import pandas as pd
import nltk
from utils.preprocess_text import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


# Get keywords (bigrams)
def get_keywords(content):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.BigramCollocationFinder.from_words(tokenizer(content))

    keywords = []
    for bigram in finder.nbest(bigram_measures.likelihood_ratio, 10):
        keywords.append(' '.join([bigram[0], bigram[1]]))

    return keywords

# Put keywords into a dataframe
def keywords_into_df(keywords):
    df = pd.DataFrame()
    df['ID'] = ['I']
    df['Title'] = ['I']
    df['Description'] = ['I']
    df['All'] = ' '.join(keywords)
    return df

# Get recommendations based on similarity scores
def get_recommendation(top, df_all, scores):
    recommendation = pd.DataFrame(columns = ['ID',  'Title', 'Description','score'])
    count = 0
    for i in top:
        recommendation.at[count, 'ID'] = df_all.index[i]
        recommendation.at[count, 'Title'] = df_all['Title'][i]
        recommendation.at[count, 'Description'] = df_all['Description'][i]
        recommendation.at[count, 'score'] =  scores[count]
        count += 1
    return recommendation

# Get similarity from TFIDF
def cosine_tfidf(descricoes_vaga, keywords_user):
    tfidf_vectorizer = TfidfVectorizer()

    # TF-IDF Scraped data
    tfidf_jobid = tfidf_vectorizer.fit_transform(descricoes_vaga)

    # TF-IDF CV
    user_tfidf = tfidf_vectorizer.transform(keywords_user)

    # Using cosine_similarity on (Scraped data) & (CV)
    cos_similarity_tfidf = map(lambda x: cosine_similarity(user_tfidf,x),tfidf_jobid)

    tfidf_cos_similarity = list(cos_similarity_tfidf)
    return tfidf_cos_similarity

# Get similarity from Count Vectorizer
def cosine_count_vectorizer(descricoes_vaga, keywords_user):
    # CountV the scraped data
    count_vectorizer = CountVectorizer()
    count_jobid = count_vectorizer.fit_transform(descricoes_vaga) # fitting and transforming the vector

    # CountV the cv
    user_count = count_vectorizer.transform(keywords_user)
    cos_similarity_countv = map(lambda x: cosine_similarity(user_count, x),count_jobid)
    cos_similarity_vectorizer = list(cos_similarity_countv)
    return cos_similarity_vectorizer

# Get similarity from KNN
def get_KNN(scraped_data, cv, n_neighbors=20):
    tfidf_vectorizer = TfidfVectorizer()    
    KNN = NearestNeighbors(n_neighbors=n_neighbors)
    KNN.fit(tfidf_vectorizer.fit_transform(scraped_data))
    NNs = KNN.kneighbors(tfidf_vectorizer.transform(cv))

    return NNs

# Combine TFIDF, Count Vectorizer and KNN into dataframe
def combine_similarities(tfidf_scored, count_vec_scored, knn_scored):
    merge1 = knn_scored[['ID','Title', 'Description', 'score']].merge(tfidf_scored[['ID','score']], on= "ID")
    final = merge1.merge(count_vec_scored[['ID','score']], on = "ID")
    final = final.rename(columns={"score_x": "KNN", "score_y": "TF-IDF","score": "CV"})
    return final

# Scale and assign weights
def scale_final(final):
    slr = MinMaxScaler()
    final[["KNN", "TF-IDF", 'CV']] = slr.fit_transform(final[["KNN", "TF-IDF", 'CV']])
    final['KNN'] = (1-final['KNN'])/3
    final['TF-IDF'] = final['TF-IDF']/3
    final['CV'] = final['CV']/3
    final['Final'] = final['KNN']+final['TF-IDF']+final['CV']
    return final.sort_values(by="Final", ascending=False)