import nltk
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download all required NLTK data first
nltk.download(['stopwords', 'punkt', 'punkt_tab'], quiet=True)

# FAQ dataset
faq = {
  "Which planet rotates clockwise, opposite to most planets?": "Venus rotates clockwise (retrograde rotation).",
  "If you weigh 70 kg on Earth, do you weigh the same in space?": "No, in space you are weightless but your mass stays the same.",
  "What expands as it gets colder?": "Water expands when it freezes, unlike most substances.",
  "Which planet has a day longer than its year?": "Venus — one rotation takes more time than one orbit.",
  "Can sound travel in space?": "No, because space is a vacuum with no medium for sound.",
  "Which element makes up most of the Sun?": "Hydrogen, which fuels nuclear fusion.",
  "Is Pluto always the farthest from the Sun?": "No, sometimes Neptune is farther due to Pluto’s elliptical orbit.",
  "Does the Moon have a dark side?": "No, all parts of the Moon receive sunlight; one side just always faces Earth.",
  "What color is the Sun actually?": "It's white, but Earth's atmosphere makes it look yellow.",
  "Can two black holes merge?": "Yes, and it releases massive gravitational waves.",
  "What is the hottest planet in our solar system?": "Venus is the hottest due to its thick CO₂ atmosphere.",
  "How long does light take to reach Earth from the Sun?": "About 8 minutes and 20 seconds.",
  "What is a black hole?": "A region in space where gravity is so strong not even light escapes.",
  "Who was the first person to walk on the Moon?": "Neil Armstrong in 1969.",
  "What is the Milky Way?": "It's the galaxy that contains our solar system.",
  "What is dark matter?": "Invisible matter that doesn't emit light but has gravitational effects.",
  "How many planets are in our solar system?": "There are 8 recognized planets.",
  "What is the International Space Station (ISS)?": "A habitable space lab orbiting Earth.",
  "Which planet has the most moons?": "Saturn has the most confirmed moons.",
  "What causes a solar eclipse?": "The Moon passes between the Sun and Earth.",
  "Which planet rotates clockwise, opposite to most planets?": "Venus rotates clockwise (retrograde rotation).",
  "If you weigh 70 kg on Earth, do you weigh the same in space?": "No, in space you are weightless but your mass stays the same.",
  "What expands as it gets colder?": "Water expands when it freezes, unlike most substances.",
  "Which planet has a day longer than its year?": "Venus — one rotation takes more time than one orbit.",
  "Can sound travel in space?": "No, because space is a vacuum with no medium for sound.",
  "Which element makes up most of the Sun?": "Hydrogen, which fuels nuclear fusion.",
  "Is Pluto always the farthest from the Sun?": "No, sometimes Neptune is farther due to Pluto’s elliptical orbit.",
  "Does the Moon have a dark side?": "No, all parts of the Moon receive sunlight; one side just always faces Earth.",
  "What color is the Sun actually?": "It's white, but Earth's atmosphere makes it look yellow.",
  "Can two black holes merge?": "Yes, and it releases massive gravitational waves."

}

stop_words = nltk.corpus.stopwords.words('english')
punctuation = string.punctuation

def preprocess(text):
    try:
        tokens = nltk.word_tokenize(text.lower())
    except LookupError:
        nltk.download('punkt_tab')
        tokens = nltk.word_tokenize(text.lower())
    return ' '.join([word for word in tokens 
                    if word not in stop_words and word not in punctuation])


questions = list(faq.keys())
processed_questions = [preprocess(q) for q in questions]
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(processed_questions)

def chatbot_response(query):
    processed_query = preprocess(query)
    query_vec = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vec, tfidf_matrix)
    max_score = similarities.max()
    
    if max_score > 0.5:
        return faq[questions[similarities.argmax()]]
    else:
        return "Could you please rephrase your question?"

print("Chatbot: Ask me about space and science")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit']:
        print("Chatbot: bye! have a nice day")
        break
    print("Chatbot:", chatbot_response(user_input))