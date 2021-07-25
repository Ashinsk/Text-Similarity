from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sent_1 = "I love horror movies."
sent_2 = "Lights out is a horror movie."

sentences = [sent_1,sent_2]

vectorizer = HashingVectorizer(stop_words='english',n_features=10)
vector = vectorizer.fit(sentences)

#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)

vector = vectorizer.transform(sentences)
# print(vector.shape)
print(vector.toarray())
print(cosine_similarity(vector[0].toarray(),vector[1].toarray()))

