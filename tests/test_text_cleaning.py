import pandas as pd

from pippi.text_cleaning import (
    Lemmatize,
    RemoveAccentedChars,
    RemoveHTMLTags,
    RemoveMultipleSpaces,
    RemovePunctuation,
    RemoveStopWords,
    TransformLettersSize,
)


def test_transform_letters_size():
    """Test TransformLettersSize class."""
    df = pd.DataFrame(
        {
            "review": ["This is a test review", "This is another test review"],
            "sentiment": ["positive", "negative"],
        }
    )
    transform_letters_size = TransformLettersSize(columns=["review"], case_transform="lower")
    output = transform_letters_size.fit_transform(df)
    assert output["review"][0] == "this is a test review"
    assert output["review"][1] == "this is another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"

    transform_letters_size = TransformLettersSize(columns=["review"], case_transform="upper")
    output = transform_letters_size.fit_transform(df)
    assert output["review"][0] == "THIS IS A TEST REVIEW"
    assert output["review"][1] == "THIS IS ANOTHER TEST REVIEW"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"


def test_remove_stop_words():
    """Test RemoveStopWords class."""
    df = pd.DataFrame(
        {
            "review": ["This is a test review", "This is another test review"],
            "sentiment": ["positive", "negative"],
        }
    )
    remove_stop_words = RemoveStopWords(columns=["review"])
    output = remove_stop_words.fit_transform(df)
    assert output["review"][0] == "test review"
    assert output["review"][1] == "another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"


def test_lemmatize():
    """Test Lemmatize class."""
    df = pd.DataFrame(
        {
            "review": ["This is a test review", "This is another test review"],
            "sentiment": ["positive", "negative"],
        }
    )
    lemmatize = Lemmatize(columns=["review"])
    output = lemmatize.fit_transform(df)
    assert output["review"][0] == "This be a test review"
    assert output["review"][1] == "This be another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"


def test_remove_punctuation():
    """Test RemovePunctuation class."""
    df = pd.DataFrame(
        {
            "review": ["This is a test review", "This is another test review"],
            "sentiment": ["positive", "negative"],
        }
    )
    remove_punctuation = RemovePunctuation(columns=["review"])
    output = remove_punctuation.fit_transform(df)
    assert output["review"][0] == "This is a test review"
    assert output["review"][1] == "This is another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"


def test_remove_html_tag():
    """Test RemoveHTMLTags class."""
    df = pd.DataFrame(
        {
            "review": ["This is a test review", "This is another test review"],
            "sentiment": ["positive", "negative"],
        }
    )
    remove_html_tags = RemoveHTMLTags(columns=["review"])
    output = remove_html_tags.fit_transform(df)
    assert output["review"][0] == "This is a test review"
    assert output["review"][1] == "This is another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"
    assert output["sentiment"][1] == "negative"


def test_remove_multiple_spaces():
    """Test RemoveMultipleSpaces class."""
    df = pd.DataFrame(
        {
            "review": ["This        is     a        test review", "This  is      another test   review"],
            "sentiment": ["  positive", " negative"],
        }
    )
    remove_multiple_spaces = RemoveMultipleSpaces(columns=["review"])
    output = remove_multiple_spaces.fit_transform(df)
    assert output["review"][0] == "This is a test review"
    assert output["review"][1] == "This is another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"
    assert output["sentiment"][1] == "negative"


def test_remove_accented_chars():
    """Test RemoveAccentedChars class."""
    df = pd.DataFrame(
        {
            "review": ["This is a test review", "This is another test review"],
            "sentiment": ["positive", "negative"],
        }
    )
    remove_accented_chars = RemoveAccentedChars(columns=["review"])
    output = remove_accented_chars.fit_transform(df)
    assert output["review"][0] == "This is a test review"
    assert output["review"][1] == "This is another test review"
    assert output["sentiment"][0] == "positive"
    assert output["sentiment"][1] == "negative"
    assert output["sentiment"][1] == "negative"
