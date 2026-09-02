"""Plain-language explanations for every technical word the interface uses.

Lecture 1: "most users are no machine learning experts: they ignore the
characteristics of the algorithms and models, and they have no understanding nor
intuition on the parameters." An interface that offers a box labelled C and no
way to find out what C is has handed the user a choice they cannot make. Every
term on every page links here.

Written for someone who has never trained a model. No formula appears without
being said in words first, and each page ends with what to actually do.
"""

from dataclasses import dataclass, field


@dataclass
class Topic:
    slug: str
    title: str
    short: str                       # the one-sentence answer
    sections: list                   # (heading, [paragraphs]) pairs
    advice: str = ""                 # what to do about it
    related: list = field(default_factory=list)
    group: str = "General"


TOPICS = {}


def _add(**kw):
    topic = Topic(**kw)
    TOPICS[topic.slug] = topic


# --------------------------------------------------------------- the basics --
_add(
    slug="feature", title="Feature", group="The basics",
    short="One column of your table -- a single piece of information the model is allowed to use.",
    sections=[
        ("An example", [
            "If your table is about flowers, the features might be petal length, petal width, "
            "sepal length and sepal width. Four numbers measured for each flower.",
            "Features are the clues. The model looks only at these when making a guess.",
        ]),
        ("What counts as a feature here", [
            "Every column except the last one. This app follows the course convention: the last "
            "column is the thing being predicted, everything before it is a feature.",
            "Columns that only number the rows (1, 2, 3, 4...) are dropped automatically, because "
            "a row number tells you nothing about the flower.",
        ]),
    ],
    advice="Nothing to do. Just check the preview table shows the columns you expected.",
    related=["target", "classification-regression"],
)

_add(
    slug="target", title="Target", group="The basics",
    short="The last column -- the thing you want the model to predict.",
    sections=[
        ("An example", [
            "For the flower table, the target is the species. You know the species for the "
            "flowers you already have; you want the model to work it out for a new flower from "
            "the measurements alone.",
        ]),
        ("Why the last column", [
            "It is just a convention so the app knows which column is the answer. If your target "
            "is not the last column, move it there in a spreadsheet before uploading.",
        ]),
    ],
    advice="Make sure the answer you care about is the rightmost column of your CSV.",
    related=["feature", "classification-regression"],
)

_add(
    slug="classification-regression", title="Classification and regression", group="The basics",
    short="Classification predicts which category something is. Regression predicts a number.",
    sections=[
        ("The difference in one line", [
            "\"Which species is this flower?\" is classification -- the answer is one of a fixed "
            "list of options.",
            "\"How far will this illness progress?\" is regression -- the answer is a number that "
            "could be anything on a scale.",
        ]),
        ("How the app decides", [
            "It looks at your last column. Words mean classification. Numbers with a lot of "
            "different values mean regression. Numbers with only a handful of repeated values "
            "(like 0, 1, 2) usually mean categories written as numbers, so it calls that "
            "classification.",
            "It always tells you which it chose, and you can override it when you upload the "
            "file. Some overrides are impossible -- you cannot do regression on words, because "
            "there is no sensible answer halfway between two species names.",
        ]),
    ],
    advice="Check the label on the visualise page. If it guessed wrong, upload again and set it "
           "yourself.",
    related=["feature", "target"],
)

_add(
    slug="model", title="Model", group="The basics",
    short="The recipe the computer uses to turn your features into a prediction.",
    sections=[
        ("What it means", [
            "A model is a way of making a guess. Different models guess in genuinely different "
            "ways -- one looks for similar past examples, another asks a series of yes/no "
            "questions, another draws a line through the data.",
            "None of them is best in general. Which one suits your data is something you find out "
            "by trying, which is what this page does.",
        ]),
        ("Training", [
            "Training means showing the model the rows where you already know the answer, so it "
            "can adjust itself to match them. After training it can be shown a new row and asked "
            "to guess.",
        ]),
    ],
    advice="Try two or three and compare. The run log keeps the results so you can look back.",
    related=["knn", "tree", "hyperparameter", "overfitting"],
)

# ------------------------------------------------------------------- models --
_add(
    slug="knn", title="k-nearest neighbours", group="Models",
    short="To guess a new case, find the most similar past cases and copy their answer.",
    sections=[
        ("How it works", [
            "Imagine plotting every flower you already know as a dot. A new flower arrives. Find "
            "the k dots closest to it, and go with whatever most of them are.",
            "That is the whole idea. There is no clever formula -- it just remembers everything "
            "and looks things up.",
        ]),
        ("What k does", [
            "k is how many neighbours get a vote. With k = 1 the new flower simply copies the "
            "single closest one, which makes it very sensitive to one odd example. With k = 25 "
            "it asks a wide circle, which smooths out oddities but may blur a real boundary.",
        ]),
        ("A catch worth knowing", [
            "\"Closest\" depends on units. If one column is measured in centimetres and another "
            "in kilometres, the kilometre column would dominate the distance for no good reason. "
            "This app rescales all the columns to comparable ranges first, automatically.",
        ]),
    ],
    advice="Good first thing to try. It is easy to explain and hard to get badly wrong.",
    related=["hyperparameter-k", "model", "scaling"],
)

_add(
    slug="tree", title="Decision tree", group="Models",
    short="A flowchart of yes/no questions that ends in an answer.",
    sections=[
        ("How it works", [
            "\"Is the petal shorter than 2.5 cm? If yes, it's a Setosa. If no, is the petal "
            "narrower than 1.8 cm? If yes...\" and so on until it reaches a verdict.",
            "The model works out the questions itself, picking at each step the one that best "
            "separates the remaining examples.",
        ]),
        ("Why people like it", [
            "You can read it. Unlike most models, you can follow exactly why it said what it "
            "said, which matters a great deal if anyone has to justify the decision.",
        ]),
        ("Why it needs watching", [
            "Left alone, a tree will keep asking questions until it has a rule for every single "
            "training example -- including the flukes. That is why you cap how deep it goes.",
        ]),
    ],
    advice="Pick this when someone will ask you to explain the model's reasoning.",
    related=["hyperparameter-depth", "overfitting", "model"],
)

_add(
    slug="logistic-regression", title="Logistic regression", group="Models",
    short="Draws a boundary line between the categories and reports how confident it is.",
    sections=[
        ("How it works", [
            "It gives each feature a weight -- how much that feature pushes towards one answer or "
            "the other -- adds them up, and turns the total into a probability.",
            "Despite the name it is used for classification, not regression. The name is a "
            "historical accident that has confused people for decades.",
        ]),
        ("What you get from it", [
            "Because each feature has a weight, you can see which features mattered and in which "
            "direction. And because the output is a probability, you can tell a confident guess "
            "from a coin-flip.",
        ]),
    ],
    advice="A sensible default when the categories are roughly separable by a straight boundary.",
    related=["hyperparameter-c", "model", "scaling"],
)

_add(
    slug="svm", title="Support vector machine (RBF)", group="Models",
    short="Finds the dividing line that leaves the widest possible gap between the categories.",
    sections=[
        ("The idea", [
            "Of all the lines that separate two groups, it looks for the one with the most "
            "clearance on both sides -- the boundary that is least likely to be wrong about a "
            "new point that lands near it.",
        ]),
        ("What RBF adds", [
            "A straight line cannot always separate the groups. RBF lets the boundary curve and "
            "wrap around clusters, so it can handle shapes a straight line cannot.",
            "The cost is that you can no longer easily explain the boundary in words.",
        ]),
    ],
    advice="Often accurate, rarely explainable. Worth trying, but check what you give up.",
    related=["hyperparameter-c", "model"],
)

_add(
    slug="ridge", title="Ridge regression", group="Models",
    short="Fits a straight-line relationship, while deliberately keeping the numbers modest.",
    sections=[
        ("How it works", [
            "Ordinary linear regression finds the weights that fit your training data as closely "
            "as possible. Ridge does the same but adds a penalty for large weights, so it prefers "
            "a slightly worse fit made of smaller, steadier numbers.",
        ]),
        ("Why hold it back on purpose", [
            "When two features say almost the same thing, a plain fit can produce wild weights "
            "that cancel each other out -- huge positive on one, huge negative on the other. It "
            "matches the training data and falls apart on anything new. The penalty prevents that.",
        ]),
    ],
    advice="The standard first choice for predicting a number. This is the model from lecture 1.",
    related=["hyperparameter-alpha", "correlation", "overfitting"],
)

# ----------------------------------------------------------- hyperparameters --
_add(
    slug="hyperparameter", title="Hyperparameter (\"values to try\")", group="Settings",
    short="A setting you choose before training, rather than something the model learns from data.",
    sections=[
        ("The cooking analogy", [
            "The model learns the recipe from your data. The hyperparameter is the oven "
            "temperature -- you set it beforehand, and it changes how the whole thing turns out.",
            "Nobody can tell you the right value by looking at it. You find it by trying several "
            "and seeing which works, which is exactly what the \"values to try\" box is for.",
        ]),
        ("Why there is a list, not one number", [
            "The app trains a separate model for every value in the list and compares them. "
            "Leave the box empty and it uses a sensible spread; type your own values, separated "
            "by commas, to look somewhere specific.",
            "Each model here has one hyperparameter that matters most, and that is the one swept.",
        ]),
        ("Almost all of them are the same dial", [
            "Whatever it is called, it usually controls one thing: how hard the model is allowed "
            "to chase your training data. Turn it one way and the model stays simple and may miss "
            "real patterns. Turn it the other and it fits every wobble, including the meaningless "
            "ones.",
        ]),
    ],
    advice="Leave it empty the first time. Look at the curve, then narrow in if you want to.",
    related=["overfitting", "cross-validation", "hyperparameter-k", "hyperparameter-c"],
)

_add(
    slug="hyperparameter-k", title="k -- the number of neighbours", group="Settings",
    short="How many similar past cases get a vote when guessing a new one.",
    sections=[
        ("What changes as you turn it up", [
            "k = 1 means the new case copies the single most similar old case. Very responsive, "
            "and completely at the mercy of one strange example.",
            "k = 25 means twenty-five cases vote. One odd example cannot swing it, but a small "
            "genuine group can get outvoted by the crowd around it.",
        ]),
        ("Practical notes", [
            "For two categories, an odd number avoids tied votes.",
            "k cannot be larger than the number of examples available to learn from. If you ask "
            "for values that are too big, the app skips them and says so.",
        ]),
    ],
    advice="Somewhere between 3 and 15 suits most datasets. Let the sweep pick.",
    related=["knn", "hyperparameter", "cross-validation"],
)

_add(
    slug="hyperparameter-depth", title="Max depth -- how deep the tree goes", group="Settings",
    short="The most yes/no questions the tree may ask before it has to commit to an answer.",
    sections=[
        ("What changes as you turn it up", [
            "Depth 1 is a single question. Fast, readable, usually too crude.",
            "Depth 3 is up to three questions in a row -- eight possible endings. Often plenty.",
            "Depth 15 can carve out a private rule for nearly every training row. It will look "
            "perfect on data it has seen and disappoint on anything new.",
        ]),
        ("The classic warning sign", [
            "On the sweep table, watch the \"on training rows\" column climb towards 1.000 while "
            "the cross-validation column stops improving. Everything past that point is "
            "memorising, not learning.",
        ]),
    ],
    advice="Shallow is usually enough, and you can read a shallow tree out loud.",
    related=["tree", "overfitting", "hyperparameter"],
)

_add(
    slug="hyperparameter-c", title="C -- how hard the model tries", group="Settings",
    short="How much the model is allowed to contort itself to get every training example right.",
    sections=[
        ("Careful -- it runs backwards", [
            "Small C means a simpler, more cautious model that accepts getting some training "
            "examples wrong.",
            "Large C means the model strains to get every training example right, which usually "
            "means it has started fitting noise.",
            "It feels inverted because C is technically the inverse of a penalty. Most people have "
            "to look this up more than once.",
        ]),
        ("Why the values jump by tens", [
            "The default list goes 0.001, 0.01, 0.1, 1, 10, 100 rather than 1, 2, 3. What matters "
            "is the order of magnitude, not small steps -- the difference between 1 and 2 is "
            "negligible, the difference between 1 and 100 is not.",
        ]),
    ],
    advice="Start with the default spread. C = 1 is a reasonable answer surprisingly often.",
    related=["logistic-regression", "svm", "hyperparameter", "overfitting"],
)

_add(
    slug="hyperparameter-alpha", title="Alpha -- the penalty on large weights", group="Settings",
    short="How firmly the model is pushed towards small, cautious numbers.",
    sections=[
        ("What changes as you turn it up", [
            "Alpha near zero means no restraint: fit the training data as closely as possible.",
            "Large alpha means heavy restraint: keep every weight small, even at the cost of a "
            "worse fit. Push it far enough and the model barely reacts to the features at all.",
            "This is the opposite direction to C. Larger alpha is a simpler model; larger C is a "
            "more complicated one.",
        ]),
        ("Where you have seen it", [
            "This is the lambda from lecture 1 -- the same dial, a different letter.",
        ]),
    ],
    advice="The default spread covers everything from no restraint to heavy. Let the sweep choose.",
    related=["ridge", "hyperparameter", "overfitting"],
)

_add(
    slug="seed", title="Random seed", group="Settings",
    short="A number that makes the random shuffling repeatable.",
    sections=[
        ("Why there is randomness at all", [
            "The rows have to be shuffled before being split into training and test groups, "
            "otherwise any ordering in your file would bias the result.",
            "The seed fixes that shuffle. Run it twice with the same seed and you get identical "
            "results -- which is what makes a result checkable by someone else.",
        ]),
        ("A genuinely useful trick", [
            "Run the same settings with three or four different seeds. If the answer barely "
            "moves, it is real. If it swings around, your dataset is too small to support the "
            "conclusion you were about to draw from it.",
        ]),
    ],
    advice="Leave it at 0. Change it when you want to check a result is not a fluke.",
    related=["test-set", "cross-validation", "standard-error"],
)

# ------------------------------------------------------- splitting & scoring --
_add(
    slug="test-set", title="Test set", group="Measuring",
    short="Rows locked away and never shown to the model, kept to check it honestly at the end.",
    sections=[
        ("Why bother", [
            "Any model can recite the examples it was trained on. That proves nothing. The only "
            "meaningful question is how it does on cases it has never seen.",
            "So some rows are set aside at the start -- 25% by default -- and the model is not "
            "allowed near them until everything else is decided.",
        ]),
        ("The trap this avoids", [
            "If you use the test rows to help choose your settings, the final score is flattering "
            "rather than honest: you have quietly tuned towards those exact rows.",
            "This app chooses the hyperparameter using cross-validation inside the training rows "
            "only. The test rows are touched once, at the very end.",
        ]),
        ("How big", [
            "Too small and the score is noisy. Too large and the model has little left to learn "
            "from. Between 20% and 30% is the usual compromise.",
        ]),
    ],
    advice="25% is a good default. Lower it if your dataset is small.",
    related=["cross-validation", "overfitting", "seed"],
)

_add(
    slug="cross-validation", title="Cross-validation and folds", group="Measuring",
    short="Testing several times on different slices instead of once, then averaging.",
    sections=[
        ("How it works", [
            "Split the training rows into 5 equal groups -- the folds. Train on 4 of them, check "
            "against the 5th. Repeat five times so every group gets a turn being the one held "
            "back. Average the five scores.",
        ]),
        ("Why not just test once", [
            "One test can be lucky. Maybe the held-back rows happened to be easy. Five tests are "
            "much harder to fluke, and the spread between them tells you how much to trust the "
            "average -- that is the standard error.",
        ]),
        ("Keeping the folds fair", [
            "Each fold is built to contain roughly the same mix of categories as the whole "
            "dataset. Without that, one fold could end up with hardly any of a rare category and "
            "the score would be measuring the shuffle rather than the model.",
        ]),
    ],
    advice="5 folds is standard. Use fewer if your dataset is small and it complains.",
    related=["test-set", "standard-error", "one-se-rule"],
)

_add(
    slug="accuracy", title="Accuracy", group="Measuring",
    short="Of all the predictions made, the fraction that were right.",
    sections=[
        ("The arithmetic", [
            "Count the correct predictions, divide by the total. 33 right out of 36 is 33/36 = "
            "0.9167, usually read as 91.67%.",
            "That is genuinely all there is to it, which is why it is the default.",
        ]),
        ("When it lies to you", [
            "Suppose 99 out of 100 emails are not spam. A model that says \"not spam\" every "
            "single time, without looking, scores 99% accuracy. It is useless -- it never catches "
            "any spam at all -- but the number looks superb.",
            "Whenever one category is much larger than the others, accuracy flatters a model that "
            "simply ignores the small ones.",
        ]),
    ],
    advice="Fine when your categories are roughly equal in size. Check the class balance table; "
           "if it is lopsided, look at macro F1 instead.",
    related=["macro-f1", "confusion-matrix", "class-balance"],
)

_add(
    slug="macro-f1", title="Macro F1", group="Measuring",
    short="Scores each category separately and averages them, so small categories count as much "
          "as large ones.",
    sections=[
        ("Two questions per category", [
            "For each category the model is asked two things.",
            "When it said \"cat\", how often was it actually a cat? That is precision -- how much "
            "you can trust it when it makes that call.",
            "Of all the real cats, how many did it find? That is recall -- how much it misses.",
            "You can cheat either one alone. Say \"cat\" once, on the most obvious cat, and "
            "precision is perfect while recall is dreadful. Say \"cat\" about everything and "
            "recall is perfect while precision collapses.",
        ]),
        ("F1 combines them", [
            "F1 is a single number that is only high when both precision and recall are high. If "
            "either one is poor, F1 is poor. It cannot be gamed by going all-in on one.",
        ]),
        ("What \"macro\" adds", [
            "It means work out F1 for each category separately, then take a plain average -- "
            "every category counted equally, no matter how many rows it has.",
            "That is the whole point. In the spam example, the ignored category drags the average "
            "right down, and the useless model finally scores as badly as it deserves.",
        ]),
    ],
    advice="Use this when some categories are much rarer than others, or when getting the rare "
           "ones right is what you actually care about.",
    related=["accuracy", "confusion-matrix", "class-balance"],
)

_add(
    slug="r-squared", title="R squared", group="Measuring",
    short="How much of the variation in the answer the model manages to explain. 1.0 is perfect.",
    sections=[
        ("What the number means", [
            "Compare your model against the laziest possible one: always guess the average, "
            "ignoring the features entirely.",
            "R squared of 0 means your model is no better than that lazy guess. 1.0 means it "
            "predicts every case exactly. 0.49 means it explains about half of the variation.",
            "It can go negative, which means the model is doing actively worse than always "
            "guessing the average. That is a signal something is wrong.",
        ]),
        ("Why it is convenient", [
            "It is on the same 0-to-1 scale whatever you are predicting, so you can compare "
            "across different problems. Mean squared error cannot do that.",
        ]),
    ],
    advice="Good for a quick sense of whether the model is working at all. Use MAE when you need "
           "to know the size of the error in real units.",
    related=["mse", "mae"],
)

_add(
    slug="mse", title="Mean squared error", group="Measuring",
    short="The average of the squared misses. Lower is better.",
    sections=[
        ("The arithmetic", [
            "For each case, take the difference between the prediction and the truth, square it, "
            "and average those across all cases.",
            "Squaring does two things: it stops overestimates and underestimates cancelling out, "
            "and it punishes one big mistake far more than several small ones.",
        ]),
        ("Why it is hard to read", [
            "The squaring leaves the number in squared units. If you are predicting a score out "
            "of 300, an MSE of 2889 is not 2889 points off -- the square root, about 54, is "
            "closer to the typical miss.",
            "Use it to compare two models on the same data. Do not try to interpret it on its own.",
        ]),
    ],
    advice="Choose this when a single large error would be much worse than several small ones.",
    related=["mae", "r-squared"],
)

_add(
    slug="mae", title="Mean absolute error", group="Measuring",
    short="The average size of the miss, in the original units. Lower is better.",
    sections=[
        ("The arithmetic", [
            "For each case, how far off was the prediction, ignoring whether it was too high or "
            "too low. Average those.",
            "An MAE of 44 means the model is off by about 44 on average -- in whatever your "
            "target is measured in. That is the plainest error figure available.",
        ]),
        ("How it differs from MSE", [
            "MAE treats a miss of 10 as exactly twice as bad as a miss of 5. MSE treats it as "
            "four times as bad.",
            "So MAE is calmer about outliers. If a handful of cases are wildly wrong, MAE will "
            "shrug where MSE will be dominated by them.",
        ]),
    ],
    advice="The easiest score to explain to someone else. Start here if you want a number you can "
           "quote in a sentence.",
    related=["mse", "r-squared"],
)

_add(
    slug="confusion-matrix", title="Confusion matrix", group="Measuring",
    short="A table of what was true against what the model said, so you can see which mistakes it "
          "makes.",
    sections=[
        ("How to read it", [
            "Each row is a true category, each column is what the model predicted. The diagonal "
            "is where they agree -- the correct answers.",
            "Everything off the diagonal is a mistake, and its position tells you what kind. A "
            "number in the Versicolor row under the Virginica column means: two flowers were "
            "really Versicolor and the model called them Virginica.",
        ]),
        ("Why it beats a single score", [
            "\"91% accurate\" does not tell you whether the 9% is spread evenly or all falling on "
            "one category.",
            "Lecture 1 makes this point with aircraft inspection: a false alarm costs an "
            "unnecessary check, a miss endangers everyone on board. Same accuracy, wildly "
            "different consequences. Only this table shows you which you are getting.",
        ]),
    ],
    advice="Always glance at it. It takes five seconds and it is where the surprises live.",
    related=["accuracy", "macro-f1"],
)

# ------------------------------------------------------------ the sweep bits --
_add(
    slug="overfitting", title="Overfitting", group="Ideas worth knowing",
    short="When a model memorises the examples it was shown instead of learning the general pattern.",
    sections=[
        ("The exam analogy", [
            "A student who memorises the answers to last year's paper will ace last year's paper "
            "and fail this year's. They learned the answers, not the subject.",
            "Models do exactly this, given the chance. The more freedom you give them -- a deeper "
            "tree, a larger C, a smaller penalty -- the more they can memorise.",
        ]),
        ("How to spot it here", [
            "The sweep table has two score columns. One is measured on rows the model was trained "
            "on; the other on rows held back from it.",
            "When the training score keeps climbing while the held-back score stops improving, "
            "you are watching overfitting happen. On the validation curve it is the moment the "
            "dashed line pulls away from the solid one.",
        ]),
    ],
    advice="This is the whole reason the app sweeps a range instead of just maximising the fit.",
    related=["hyperparameter", "test-set", "one-se-rule", "validation-curve"],
)

_add(
    slug="standard-error", title="Standard error", group="Ideas worth knowing",
    short="How much a score wobbles depending on which rows happened to land where.",
    sections=[
        ("Where it comes from", [
            "Cross-validation gives five scores, not one. They will not be identical -- some "
            "slices are a bit easier than others.",
            "The standard error summarises how much they disagree. A small one means the five "
            "runs broadly concurred; a large one means the result depends heavily on the shuffle.",
        ]),
        ("Why it matters more than it sounds", [
            "If setting A scores 0.9655 and setting B scores 0.9640, B looks worse. But if the "
            "standard error is 0.0158, that gap is a tenth of the ordinary wobble -- there is no "
            "real difference. Picking A over B is reading meaning into noise.",
        ]),
    ],
    advice="Look at the ± column before believing any difference between two rows.",
    related=["cross-validation", "one-se-rule", "seed"],
)

_add(
    slug="one-se-rule", title="How the winner is picked", group="Ideas worth knowing",
    short="Two options: take the top score, or take the simplest setting that is statistically "
          "tied with the top score.",
    sections=[
        ("Best mean", [
            "The obvious approach: whichever value scored highest, wins.",
            "The catch is that the highest of five noisy numbers is partly just the luckiest of "
            "five noisy numbers.",
        ]),
        ("One standard error (the default)", [
            "Take the best score, and gather every setting within one standard error of it. "
            "Statistically, none of those is distinguishable from the winner.",
            "From that group, keep the simplest model -- the shallower tree, the wider "
            "neighbourhood, the heavier penalty.",
        ]),
        ("A real example from the iris data", [
            "Tree depths 3, 4, 5, 6, 8, 10 and 14 all score exactly 0.9655. Identical. But the "
            "training score climbs from 0.9847 to a perfect 1.000 across that range -- the deeper "
            "trees are memorising, and gaining nothing for it.",
            "\"Best mean\" picks arbitrarily among seven tied values. The one standard error rule "
            "takes depth 3: same measured performance, a fraction of the complexity, and a tree "
            "you can actually read.",
        ]),
    ],
    advice="Leave it on one standard error unless you have a specific reason to chase the last "
           "fraction of a percent.",
    related=["standard-error", "overfitting", "cross-validation"],
)

_add(
    slug="validation-curve", title="The sweep curve", group="Ideas worth knowing",
    short="One line per score, drawn across every value tried, so you can see the trade-off "
          "instead of just its answer.",
    sections=[
        ("The two lines", [
            "The solid line is how the model did on data held back from it -- the one that "
            "matters. The shaded band around it is the standard error.",
            "The dashed line is how it did on the data it was trained on. That one is always "
            "flattering; it is drawn for comparison, not for judging.",
        ]),
        ("What to look for", [
            "A peak, or a plateau, in the solid line. That is roughly where the setting should be.",
            "The two lines separating. That is where memorising begins.",
            "A completely flat solid line means the setting barely matters for your data -- also "
            "useful to know, and invisible if you only ever saw the winner.",
        ]),
    ],
    advice="This picture is the reason to sweep a range rather than trust a single number.",
    related=["overfitting", "standard-error", "one-se-rule"],
)

_add(
    slug="automl", title="\"Decide for me\"", group="Ideas worth knowing",
    short="Runs every model with default settings and keeps the winner, without asking you "
          "anything.",
    sections=[
        ("What it does", [
            "Tries each available model, sweeps each one's default range, and keeps whichever "
            "scored best. Test size, fold count, score and tie-breaking are all chosen for you.",
        ]),
        ("What it does not do", [
            "Every one of those decisions still gets made -- just not by you, and without telling "
            "you they were decisions at all.",
            "It might pick a model that scores half a percent higher and is impossible to "
            "explain, over one that is nearly as good and readable in three lines. It has no way "
            "of knowing that you would have preferred the second.",
        ]),
        ("Why the button exists", [
            "Lecture 1 walks through an automated pipeline, gets a good answer, and concludes "
            "\"no human needed\" -- then spends the next slide listing every human judgement "
            "hidden inside it. This button is that argument, next to the form that makes those "
            "judgements visible. Press it, then compare.",
        ]),
    ],
    advice="Useful as a baseline. Read the table of what it discarded before accepting its answer.",
    related=["model", "one-se-rule", "hyperparameter"],
)

# ---------------------------------------------------- visualisation concepts --
_add(
    slug="correlation", title="Correlation", group="Reading your data",
    short="How strongly two columns move together, from -1 to +1.",
    sections=[
        ("Reading the number", [
            "+1 means when one goes up the other always goes up, in perfect lock-step. -1 means "
            "one always goes up as the other goes down. 0 means no straight-line relationship.",
            "0.87 between petal length and petal width means longer petals are reliably wider.",
        ]),
        ("Why the heatmap is worth a look", [
            "Two features correlated at 0.96 are telling you nearly the same thing twice. That is "
            "not fatal, but it can make some models produce unstable, hard-to-read weights -- "
            "which is exactly what ridge regression's penalty is there to control.",
        ]),
        ("The usual caution", [
            "Correlation only detects straight-line relationships, and it never establishes that "
            "one thing causes the other.",
        ]),
    ],
    advice="Skim it for values near +1 or -1 between features. Those are your duplicates.",
    related=["ridge", "feature-ranking"],
)

_add(
    slug="feature-ranking", title="Which features carry the answer", group="Reading your data",
    short="A rough ranking of how much each column tells you about the target, on its own.",
    sections=[
        ("How it is worked out", [
            "For a number target, it is the strength of the correlation with the target.",
            "For categories, it compares how far apart the category averages are against how "
            "spread out each category is internally. A feature scores highly when the groups sit "
            "well apart and each group is tight.",
        ]),
        ("The important limitation", [
            "Every feature is judged alone. Two features that are useless separately but decisive "
            "together will both rank near the bottom, and the ranking will be wrong about them.",
            "So this is a guide to what is worth plotting first. It is not a list of columns to "
            "delete.",
        ]),
    ],
    advice="Use it to choose which two features to put on the scatter plot.",
    related=["correlation", "projection", "feature"],
)

_add(
    slug="projection", title="The two-direction view", group="Reading your data",
    short="Squashes all your features down to two axes so the whole dataset fits in one picture.",
    sections=[
        ("The shadow analogy", [
            "A 3D object casts a 2D shadow. Rotate it and you get different shadows -- some "
            "uninformative, one that shows the shape best.",
            "This does the same with all your features at once, finding the angle that keeps as "
            "much of the variation as possible.",
        ]),
        ("What the percentages mean", [
            "\"73.0% of the variance\" means the horizontal axis alone captures nearly three "
            "quarters of everything that varies in your data. The higher the two percentages add "
            "up to, the more faithful the picture is.",
            "If they add up to something low, treat the plot as a rough sketch -- a lot is being "
            "flattened out of view.",
        ]),
        ("What to look for", [
            "If the groups separate here but in no single pair of features, the information is "
            "spread across several columns rather than sitting in one. That is a reason to prefer "
            "a model that combines features.",
        ]),
    ],
    advice="A quick sanity check on whether the categories are distinguishable at all.",
    related=["feature-ranking", "scaling"],
)

_add(
    slug="class-balance", title="Class balance", group="Reading your data",
    short="How many rows each category has. Lopsided counts distort several things at once.",
    sections=[
        ("Why it matters", [
            "If 95% of your rows are one category, a model can score 95% accuracy by always "
            "guessing that one and never learning anything.",
            "A category with very few rows also gives the model almost nothing to learn from, and "
            "makes every score involving it unreliable.",
        ]),
        ("What the app does about it", [
            "Both the test split and every cross-validation fold are built to preserve these "
            "proportions, and every category is guaranteed at least one row in the test set. "
            "Otherwise a rare category could vanish from the evaluation entirely.",
        ]),
    ],
    advice="If the shares are badly uneven, use macro F1 rather than accuracy.",
    related=["macro-f1", "accuracy", "cross-validation"],
)

_add(
    slug="scaling", title="Putting features on the same scale", group="Reading your data",
    short="Rescaling columns so a feature measured in big units does not drown out the others.",
    sections=[
        ("The problem", [
            "Say one column is age in years (roughly 0-100) and another is income in euros "
            "(roughly 0-100,000).",
            "Any model that measures distance would be almost entirely driven by income, simply "
            "because its numbers are bigger. Not because it matters more -- because of the units "
            "someone happened to pick.",
        ]),
        ("What is done here", [
            "Before training, every column is shifted and stretched to a comparable range. This "
            "is applied to nearest neighbours, logistic regression, the SVM and ridge, all of "
            "which are sensitive to it.",
            "Decision trees are left alone, because they only ever ask whether a value is above "
            "or below a threshold, and that answer does not change if you rescale the column.",
        ]),
    ],
    advice="Handled automatically. Worth knowing about because it explains why a model can be "
           "ruined by units alone.",
    related=["knn", "projection", "ridge"],
)


def groups():
    """Topics arranged by section, in the order they were defined."""
    ordered = {}
    for topic in TOPICS.values():
        ordered.setdefault(topic.group, []).append(topic)
    return ordered
