from utils import *
import argparse

# Inputs
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-pdf_path", "--pdf_path", type=str)
parser.add_argument("-jobs_path", "--jobs_path", type=str)
args = parser.parse_args()

pdf_path = args.pdf_path
jobs_path = args.jobs_path

# Get text from curriculum 
text_from_cv = text_extractor(pdf_path)

# Preprocess text from curriculum
text_preprocessed = preprocess_txt(text_from_cv)

# Get keywords from preprocessed text
keywords = get_keywords(text_preprocessed)

# Jobs dataframe
jobs = pd.read_csv(jobs_path)

# Job Title + Description
jobs['All'] = jobs['Title'] + ' ' + jobs['Description']
jobs['All'] = jobs['All'].apply(lambda x: preprocess_txt(x))

# Keywords from cv into dataframe
df = keywords_into_df(keywords)

# TFIDF
cos_tfidf = cosine_tfidf(jobs['All'], df['All'])
top = sorted(range(len(cos_tfidf)), key=lambda i: cos_tfidf[i], reverse=True)
list_scores = [cos_tfidf[i][0][0] for i in top]
tfidf_scored = get_recommendation(top, jobs, list_scores)

# Count Vectorizer
cos_cvectorizer = cosine_count_vectorizer(jobs['All'], df['All'])
top = sorted(range(len(cos_cvectorizer)), key=lambda i: cos_cvectorizer[i], reverse=True)
list_scores = [cos_cvectorizer[i][0][0] for i in top]
cv_scored = get_recommendation(top, jobs, list_scores)

# KNN
nearest_n = get_KNN(jobs['All'], df['All'], n_neighbors=20)
top = nearest_n[1][0][1:]
index_score = nearest_n[0][0][1:]
knn_scored = get_recommendation(top, jobs, index_score)

# Combine scores
final = combine_similarities(tfidf_scored, cv_scored, knn_scored)

# Scale and assign weights to final
final_scaled = scale_final(final)

final_scaled.to_json("vagas_recomendadas.json")