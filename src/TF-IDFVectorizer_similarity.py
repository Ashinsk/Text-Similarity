from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sent_1 = "I love horror movies."
sent_2 = "Lights out is a horror movie."

sentences = [sent_1,sent_2]

vectorizer = TfidfVectorizer(stop_words='english')
vectorizer.fit(sentences)

#print(vectorizer.vocabulary_)
#print(vectorizer.idf_)

vector = vectorizer.transform(sentences)
# print(vector.shape)
print(vector.toarray())
print(cosine_similarity(vector[0].toarray(),vector[1].toarray()))

