from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

sent_1 = "I love horror movies."
sent_2 = "Lights out is a horror movie."

sent_1_tokens = word_tokenize(sent_1)
sent_2_tokens = word_tokenize(sent_2)

stop_words = stopwords.words('english')
sent_1_vector = []
sent_2_vector = []

cleaned_sent_1_tokens = {token for token in sent_1_tokens if not token in stop_words}  
cleaned_sent_2_tokens = {token for token in sent_2_tokens if not token in stop_words} 

tokens = cleaned_sent_1_tokens.union(cleaned_sent_2_tokens)
tokens = sorted(tokens)
print(tokens)
for token in tokens:
	if token in cleaned_sent_1_tokens: 
		sent_1_vector.append(1)
	else:
		sent_1_vector.append(0)
		
	if token in cleaned_sent_2_tokens:
		sent_2_vector.append(1)
	else:
		sent_2_vector.append(0)

print('Doc 1 vector',sent_1_vector)
print('Doc 2 vector',sent_2_vector)

dot_product = 0

for i in range(len(tokens)):
	dot_product += sent_1_vector[i] * sent_2_vector[i]


cosine_similarity = dot_product / float(pow(sum(sent_1_vector) * sum(sent_2_vector), 1/2))
print(cosine_similarity)

