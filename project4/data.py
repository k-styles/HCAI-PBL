"""The movie pool and the feature vector x that the utility model runs on.

IMDB 5000 has 28 columns and none of them were collected for preference
elicitation, so choosing what becomes a feature is the whole of Task 1.
"""

import functools
import math
from pathlib import Path

import numpy as np
import pandas as pd
from django.conf import settings

CSV = Path(settings.BASE_DIR) / "project4" / "data" / "movies.csv"

# A participant who has never heard of a film is not expressing a preference,
# they are guessing from the metadata card -- which adds response noise that no
# amount of modelling can undo. Cutting the pool at 25k IMDB votes and dropping
# duplicate titles leaves 2,734 films spanning 1927-2016, which is wide enough
# that random pairs are not all the same decade.
MIN_VOTES = 25_000

MATURE = {"R", "NC-17", "X", "Unrated", "TV-MA", "M", "GP"}

# ---------------------------------------------------------------------------
# OWN WORK REQUIRED -- Project 4, Task 1
#
#   "Choose a feature representation that you believe is appropriate for a
#    movie recommender system. Justify your choice of features and write a
#    method to extract them from the dataset."
#
# Two constraints decide this, and neither is about which columns describe a
# film best.
#
# (1) The elicitation budget. w is estimated from something like 25 responses.
#     A one-hot over director_name is 2,398 columns and two randomly drawn
#     films share a director with probability under 0.1%, so essentially every
#     one of those weights is multiplied by zero in every observation we will
#     ever collect. The same goes for the three actor columns. They are not
#     weak features, they are unidentifiable ones at this sample size, and
#     including them would only let the prior invent preferences.
#
# (2) The participant can only react to what is on the card. Budget, gross,
#     facebook likes and aspect ratio are not things anyone consults when
#     picking a film to watch tonight; if they are in x but not on the card,
#     the responses cannot explain the variance they carry, and the fit will
#     attribute choices to them by accident.
#
# So a genre earns a dimension only if random pairs actually differ on it. For
# prevalence p that happens with probability 2p(1-p); requiring at least one in
# ten informative pairs gives p >= 0.053, which admits 14 genres and drops War,
# History, Sport, Music, Musical, Western, Documentary, Film-Noir and News.
#
# The rest are the axes people actually talk about: how old it is, how long it
# is, how well reviewed, how widely seen (deliberately separate from reviewed
# -- "acclaimed but obscure" is a real taste), whether it is for adults, and
# whether it is in colour. 20 dimensions.
#
# Continuous features are standardised so that w_j reads as utility per
# standard deviation and one isotropic Gaussian prior is sensible for all of
# them. Genre indicators are left as raw 0/1 so that w_g reads as the utility
# of the tag itself and the same prior shrinks every genre equally; scaling
# them to unit variance would hand rare genres larger utility swings for the
# same weight, which is backwards.
# ---------------------------------------------------------------------------

PAIR_INFORMATIVENESS = 0.10          # at least 1 informative pair in 10

CONTINUOUS = [
    ("era", "How recent it is", "title_year"),
    ("length", "How long it runs", "duration"),
    ("acclaim", "How well reviewed", "imdb_score"),
    ("reach", "How widely seen", None),   # log10 of the vote count
]

BINARY = [
    ("mature", "Rated for adults"),
    ("monochrome", "Black and white"),
]


def _genre_threshold():
    """Smallest prevalence at which 2p(1-p) clears PAIR_INFORMATIVENESS."""
    # p^2 - p + PAIR_INFORMATIVENESS/2 = 0, lower root.
    return (1 - math.sqrt(1 - 2 * PAIR_INFORMATIVENESS)) / 2


@functools.lru_cache(maxsize=1)
def pool():
    """The films, their feature matrix, and the bookkeeping to explain both."""
    raw = pd.read_csv(CSV)

    # Titles in this dump carry a trailing non-breaking space, and the colour
    # column a leading ordinary one.
    raw["movie_title"] = raw["movie_title"].str.replace("\xa0", "", regex=False).str.strip()
    raw["color"] = raw["color"].fillna("").str.strip()

    keep = (raw["num_voted_users"] >= MIN_VOTES)
    for column in ("movie_title", "genres", "title_year", "duration", "imdb_score"):
        keep &= raw[column].notna()
    films = raw[keep].drop_duplicates(subset="movie_title").reset_index(drop=True)

    tags = films["genres"].str.split("|")
    prevalence = (tags.explode().value_counts() / len(films)).sort_values(ascending=False)
    cut = _genre_threshold()
    genres = [g for g, p in prevalence.items() if p >= cut]

    columns = []
    for genre in genres:
        columns.append(tags.apply(lambda t, g=genre: float(g in t)).to_numpy())

    year = films["title_year"].to_numpy(float)
    duration = films["duration"].to_numpy(float)
    score = films["imdb_score"].to_numpy(float)
    reach = np.log10(films["num_voted_users"].to_numpy(float))

    scaling = {}
    for name, values in (("era", year), ("length", duration),
                         ("acclaim", score), ("reach", reach)):
        mean, sd = values.mean(), values.std()
        scaling[name] = (mean, sd)
        columns.append((values - mean) / sd)

    columns.append(films["content_rating"].isin(MATURE).to_numpy(float))
    columns.append((films["color"] == "Black and White").to_numpy(float))

    names = (list(genres)
             + [key for key, _, _ in CONTINUOUS]
             + [key for key, _ in BINARY])
    labels = ({g: g for g in genres}
              | {key: label for key, label, _ in CONTINUOUS}
              | {key: label for key, label in BINARY})

    X = np.column_stack(columns)
    return {
        "films": films,
        "X": X,
        "names": names,
        "labels": labels,
        "genres": genres,
        "prevalence": prevalence,
        "threshold": cut,
        "scaling": scaling,
        "dropped": [g for g, p in prevalence.items() if p < cut],
    }


def cards(indices):
    """What the participant is shown -- and nothing that is not a feature."""
    data = pool()
    films = data["films"]
    out = []
    for i in indices:
        row = films.iloc[int(i)]
        tags = [g for g in row["genres"].split("|")]
        out.append({
            "index": int(i),
            "title": row["movie_title"],
            "year": int(row["title_year"]),
            "duration": int(row["duration"]),
            "score": float(row["imdb_score"]),
            "votes": int(row["num_voted_users"]),
            "rating": row["content_rating"] if isinstance(row["content_rating"], str) else "Unrated",
            "genres": tags,
            "shown": [g for g in tags if g in data["genres"]],
            "monochrome": row["color"] == "Black and White",
            "link": row["movie_imdb_link"],
        })
    return out


def sample(rng, count, avoid=()):
    """Uniform draw without replacement, which is what the brief permits."""
    banned = set(int(i) for i in avoid)
    total = len(pool()["films"])
    picked = []
    while len(picked) < count:
        i = int(rng.integers(total))
        if i not in banned:
            banned.add(i)
            picked.append(i)
    return picked
