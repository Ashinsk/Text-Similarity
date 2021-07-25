from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sent_1 = "I love horror movies."
sent_2 = "Lights out is a horror movie."

sentences = [sent_1,sent_2]

vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(sentences)
#print(cv.vocabulary_)

feature_names = vectorizer.get_feature_names()
print(feature_names)

vector = vectorizer.transform(sentences)
# print(vector.shape)
print(vector.toarray())
print(cosine_similarity(vector[0].toarray(),vector[1].toarray()))

