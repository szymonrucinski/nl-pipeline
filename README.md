# Text cleaning Pipeline 

[![Build package](https://github.com/szymonrucinski/pippi-lang/actions/workflows/build-pkg.yml/badge.svg)](https://github.com/szymonrucinski/pippi-lang/actions/workflows/build-pkg.yml) [![Check style](https://github.com/szymonrucinski/pippi-lang/actions/workflows/check-style.yml/badge.svg)](https://github.com/szymonrucinski/pippi-lang/actions/workflows/check-style.yml)[![Run Tests](https://github.com/szymonrucinski/pippi-lang/actions/workflows/run-tests.yml/badge.svg)](https://github.com/szymonrucinski/pippi-lang/actions/workflows/run-tests.yml)
___
## Description
This code contains a pipeline for pre-processing text data for sentiment analysis. It includes steps for removing stop words, HTML tags, changing letter size, and removing punctuation.
*Future code will include text-transformations like word-embedding and word-vectorization.*

### Example
Elegant data pipelines are a key component of any data science project. They allow you to automate the process of cleaning, transforming, and analyzing data. This code is a simple example of how to create a pipeline for text data using cutom transformers and the sklearn Pipeline class.

``` python

from pippi import (
    TransformLettersSize,
    RemoveStopWords,
    Lemmatize,
    RemovePunctuation,
    RemoveHTMLTags,
)
from sklearn.pipeline import Pipeline
import pandas as pd

    pipeline = Pipeline(
        steps=[
            ("remove_stop_words", RemoveStopWords(columns=["review","sentiment"])),
            ("remove_html_tags", RemoveHTMLTags(columns=df.columns.to_list())),
            ("uppercase_letters", TransformLettersSize(columns=["sentiment"], case_transform="upper")),
            ("remove_punctuation", RemovePunctuation(columns=["review"])),
        ]
    )
    output = pipeline.fit_transform(df)
    df = pd.DataFrame(output, columns=["review", "sentiment"])

```
Pipeline Visualization:

``` markdown
[RemoveStopWords] -> [RemoveHTMLTags] -> [TransformLettersSize] ->   [RemovePunctuation]
```

