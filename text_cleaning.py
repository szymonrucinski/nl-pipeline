from pyexpat import features
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import nltk
from nltk.corpus import stopwords
import spacy
from spacy.lang.de.examples import sentences
import coloredlogs, logging

coloredlogs.install()
logger = logging.getLogger("text_cleaning.py")


class TransformLettersSize(BaseEstimator, TransformerMixin):
    """Converts all letters to upper case or lower case."""

    def __init__(self, case_transform):
        self.case_transform = case_transform
        logger.info("Transforming letters to upper or lower case")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        preproc_data = X.copy()
        if self.case_transform == "lower":
            for column in preproc_data:
                column_str = column
                if preproc_data[column_str].dtype in ["object", "str"]:
                    preproc_data[column_str] = data[column_str].str.lower()

        elif self.case_transform == "upper":
            for column in preproc_data:
                column_str = column
                if preproc_data[column_str].dtype in ["object", "str"]:
                    preproc_data[column_str] = data[column_str].str.upper()

        return preproc_data


class RemoveStopWords(BaseEstimator, TransformerMixin):
    """Removes stop words from the text. By default english stop words are removed."""

    def __init__(self, stop_words=set(stopwords.words("english"))):
        self.stop_words = stop_words

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Removing stop words.")
        if self.remove_stop_words == True:
            preproc_data = X.copy()
            for column in preproc_data:
                if preproc_data[column].dtype in ["object", "str"]:
                    preproc_data[column] = preproc_data[column].apply(
                        lambda words: " ".join(
                            word.lower()
                            for word in words.split()
                            if word.lower() not in stopwords
                        )
                    )
            return preproc_data


class RemoveHtmlTags(BaseEstimator, TransformerMixin):
    """Removes html tags from the text using regex."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info("Removing html tags.")
        preproc_data = X.copy()
        for column in preproc_data:
            if preproc_data[column].dtype in ["object", "str"]:
                preproc_data[column] = preproc_data[column].str.replace("<[^<]+?>", "")
        return preproc_data


class RemovePunctuation(BaseEstimator, TransformerMixin):
    """Remove punctuation from the text using regex."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preproc_data = X.copy()
        for column in preproc_data:
            if preproc_data[column].dtype in ["object", "str"]:
                preproc_data[column].str.replace(r"[^A-Za-z ]+", "", regex=True)
        return preproc_data


class RemoveDigits(BaseEstimator, TransformerMixin):
    """Remove digits from the text using regex."""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        preproc_data = X.copy()
        preproc_data.MarketingDescription_DE = (
            preproc_data.MarketingDescription_DE.str.replace("\d+", "")
        )
        return preproc_data


class Lemmatize(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize):
        self.lemmatize = lemmatize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.lemmatize == "yes":
            nlp = spacy.load("de_core_news_sm")
            preproc_data = X.copy()
            lemmatized_sentences = []
            for sentence in preproc_data.MarketingDescription_DE.values:
                x = nlp(sentence)
                list_of_strings = [i.text for i in x]
                x = " ".join(list_of_strings)
                lemmatized_sentences.append(x)

            preproc_data.MarketingDescription_DE = lemmatized_sentences
            return preproc_data
        else:
            return X
