# built-in
import string
import re
from typing import List

# installed
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')


class Preprocess:
    """Preprocessing routines for Clickbait dataset"""

    def to_lowercase(self, tokens: List[str]) -> List[str]:

        for idx, token in enumerate(tokens):
            a_lower_token = token.lower()
            tokens[idx] = a_lower_token

        return tokens

    def remove_punctuation(self, tokens: List[str]):

        for idx, token in enumerate(tokens):
            a_token_with_no_punct = token.translate(str.maketrans('', '', string.punctuation))
            tokens[idx] = a_token_with_no_punct

        return tokens

    def remove_numbers(self, tokens: List[str]):

        for idx, token in enumerate(tokens):
            a_token_with_no_numbers = re.sub(r'\d+', '', token).strip()
            tokens[idx] = a_token_with_no_numbers

        return tokens

    def tokenize_to_word(self, tokens: List[str]) -> List[List[str]]:

        all_word_tokens = []
        for sentence in tokens:
            word_tokens_from_current_sentence = nltk.tokenize.word_tokenize(sentence, language='english')
            all_word_tokens.append(word_tokens_from_current_sentence)

        return all_word_tokens

    def remove_stop_words(self, sentences_as_word_tokens: List[List[str]]) -> List[List[str]]:

        stop_words = set(stopwords.words('english'))

        word_tokens_with_no_stop_words = [[word for word in sent if not word in stop_words] for sent in
                                          sentences_as_word_tokens]

        return word_tokens_with_no_stop_words

    def remove_empty_sentences(self, sentences_as_word_tokens: List[List[str]]) -> List[List[str]]:

        for idx, sentence in enumerate(sentences_as_word_tokens):
            if len(sentence) == 0:
                sentences_as_word_tokens.pop(idx)

        return sentences_as_word_tokens

    def __call__(self, sentence_list: List[str]) -> List[List[str]]:

        sentence_list = self.to_lowercase(sentence_list)
        sentence_list = self.remove_punctuation(sentence_list)
        sentence_list = self.remove_numbers(sentence_list)

        sentence_list = self.tokenize_to_word(sentence_list)
        sentence_list = self.remove_stop_words(sentence_list)
        sentence_list = self.remove_empty_sentences(sentence_list)

        return sentence_list
