"""AG News, loaded from the copies committed alongside the app.

The dataset ships as gzipped CSV rather than being pulled from Hugging Face at
runtime. Two reasons: the deployed app has no business making network calls on a
page load, and the `datasets` package would be a heavy dependency on a 512 MB
tier for something that is two files.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TOPICS = ["World", "Sports", "Business", "Sci/Tech"]


@dataclass
class AGNews:
    train: pd.DataFrame
    test: pd.DataFrame

    @property
    def topics(self):
        return TOPICS


@lru_cache(maxsize=1)
def load() -> AGNews:
    train = pd.read_csv(HERE / "data" / "train.csv.gz")
    test = pd.read_csv(HERE / "data" / "test.csv.gz")
    return AGNews(train=train, test=test)
