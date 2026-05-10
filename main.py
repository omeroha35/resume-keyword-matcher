from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

resume = input("Paste your resume:\n")
job = input("\nPaste job description:\n")

vectorizer = CountVectorizer().fit_transform([resume, job])
vectors = vectorizer.toarray()

score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

print(f"\nMatch Score: {round(score * 100, 2)}%")
