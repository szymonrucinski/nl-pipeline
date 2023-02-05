import pandas as pd
from pippi.utils import download_dataset
from sklearn.pipeline import Pipeline

from pippi.text_cleaning import (
    TransformLettersSize,
    RemoveStopWords,
    RemovePunctuation,
    RemoveHTMLTags,
)
import coloredlogs
import logging

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
