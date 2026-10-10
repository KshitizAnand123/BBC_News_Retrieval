import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

class NLPProcessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        if not isinstance(text, str): return ""
        text = text.lower().translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        processed = [self.stemmer.stem(w) for w in tokens if w not in self.stop_words]
        return " ".join(processed)

    def expand_query(self, query):
        tokens = word_tokenize(query.lower())
        expanded_terms = set(tokens)
        for word in tokens:
            if word not in self.stop_words:
                for syn in wordnet.synsets(word):
                    for l in syn.lemmas():
                        expanded_terms.add(l.name().replace('_', ' '))
        return " ".join(list(expanded_terms)[:15])

    def query_focused_summary(self, text, query, top_sentences=2):
        sentences = sent_tokenize(text)
        if len(sentences) <= top_sentences: return text
        processed_query = set(self.preprocess(query).split())
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            processed_sent = set(self.preprocess(sentence).split())
            overlap = len(processed_query.intersection(processed_sent))
            scored_sentences.append((overlap, i, sentence))
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        best_sentences = sorted(scored_sentences[:top_sentences], key=lambda x: x[1])
        return " ... ".join([s[2] for s in best_sentences])
