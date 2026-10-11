import string
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize

for _pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

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

    def get_expansion_details(self, query):
        if not isinstance(query, str) or not query.strip():
            return {"original": query, "original_terms": [], "added_synonyms": [], "expanded_query": query}
        
        tokens = [t.lower() for t in word_tokenize(query) if t.isalpha()]
        content_tokens = [t for t in tokens if t not in self.stop_words]
        
        added_synonyms = []
        for word in content_tokens:
            count = 0
            for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    syn_term = l.name().replace('_', ' ').lower()
                    # Only add genuine single-word or closely related synonyms not identical to word
                    if syn_term != word and syn_term not in content_tokens and syn_term not in added_synonyms:
                        added_synonyms.append(syn_term)
                        count += 1
                        if count >= 2:  # Max 2 clean synonyms per content term
                            break
                if count >= 2:
                    break
        
        # Keep original query tokens FIRST to maintain primary intent and prevent topic drift
        expanded_list = tokens + added_synonyms
        expanded_query = " ".join(expanded_list[:12])
        return {
            "original": query,
            "original_terms": content_tokens,
            "added_synonyms": added_synonyms[:8],
            "expanded_query": expanded_query
        }

    def expand_query(self, query):
        details = self.get_expansion_details(query)
        return details["expanded_query"]

    def highlight_keywords(self, text, query):
        """
        Wraps matched query keywords in standard HTML <mark> tags for visual highlighting.
        """
        import re
        import html
        if not text or not query:
            return text
        
        # Extract meaningful non-stopword query terms (at least 3 characters)
        raw_tokens = [t.lower() for t in word_tokenize(query) if t.isalpha() and len(t) > 2]
        query_words = [t for t in raw_tokens if t not in self.stop_words]
        if not query_words:
            return text
        
        # Escape raw text to prevent HTML injection
        escaped_text = html.escape(text)
        
        # Highlight words using regex with word boundary
        pattern = re.compile(r'\b(' + '|'.join(map(re.escape, query_words)) + r')\b', re.IGNORECASE)
        highlighted = pattern.sub(r'<mark style="background-color: #fef08a; color: #0f172a; padding: 1px 4px; border-radius: 3px; font-weight: 600;">\1</mark>', escaped_text)
        return highlighted

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
