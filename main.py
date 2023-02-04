import pandas as pd
import nltk
from sklearn.compose import ColumnTransformer
from utils import download_dataset
from sklearn.pipeline import Pipeline
from text_cleaning import (
    TransformLettersSize,
    RemoveStopWords,
    Lemmatize,
    RemovePunctuation,
    RemoveHTMLTags,
)
import coloredlogs, logging

nltk.download("stopwords")
coloredlogs.install()
logger = logging.getLogger("main.py")


def main() -> None:
    logger.info("Downloading dataset")
    df = download_dataset()
    logger.info("Creating pipeline")

    pipeline = Pipeline(
        steps=[
            ("remove_stop_words", RemoveStopWords(columns=["review", "sentiment"])),
            ("remove_html_tags", RemoveHTMLTags(columns=df.columns.to_list())),
            (
                "uppercase_letters",
                TransformLettersSize(columns=["sentiment"], case_transform="upper"),
            ),
            ("remove_punctuation", RemovePunctuation(columns=["review"])),
        ]
    )
    output = pipeline.fit_transform(df)
    logger.info("Saving dataset")
    df = pd.DataFrame(output, columns=["review", "sentiment"])
    logging.info("dataframe head - {}".format(df.head()))
    df.to_csv("preprocessed_dataset.csv", index=False)


main()
