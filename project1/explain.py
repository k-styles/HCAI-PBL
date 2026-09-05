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


class Math(str):
    """A display formula inside a section's paragraph list.

    Subclassing str keeps every other paragraph a plain string, so nothing else
    changes. The template asks for `.html`, which a plain str does not have --
    Django resolves that to the empty string, so the two render differently
    without any type checking in the template.

    Rendering is done by home.mathfmt: real subscripts, italic variables and
    upright function names, built as HTML so the pages need no network and no
    JavaScript. See that module for the notation.
    """
    is_math = True

    @property
    def html(self):
        from home.mathfmt import render
        return render(self)


class Aside(str):
    """A smaller note under a paragraph -- notation reminders, caveats."""
    is_aside = True


TOPICS = {}


def _add(**kw):
    topic = Topic(**kw)
    TOPICS[topic.slug] = topic


# --------------------------------------------------------------- the basics --

_add(slug="feature", title="Feature", group="The basics",
     short="One column of your table — a single measured quantity the model may use.",
     sections=[("What it is", [
         "In a table of flowers, petal length is a feature and so is sepal width. Each row "
         "is one example; each column other than the target is one feature.",
         "The standard notation writes a single example as a vector x ∈ ℝᵈ, where d is the "
         "number of features, and the whole dataset as a matrix X ∈ ℝⁿˣᵈ with n rows."]),
       ("Why the choice matters", [
         "The model sees only these numbers. Anything not in a column is invisible to it — "
         "which is the point lecture 1 makes about the human hiding in every step: somebody "
         "decided what to measure, and that decision bounds everything the model can ever "
         "learn."])],
     advice="Features must be numeric here. Text columns can only be a target, not an input.",
     related=["target", "scaling", "correlation"])

_add(slug="target", title="Target", group="The basics",
     short="The column you want predicted — by convention here, the last one.",
     sections=[("What it is", [
         "Written y. For one example the pair is (x, y): the features and the answer. The "
         "dataset is D = {(x₁, y₁), …, (xₙ, yₙ)}, which is exactly lecture 1's notation.",
         "This app follows the brief's convention that the last column of the CSV is the "
         "target and everything before it is a feature."]),
       ("What its type decides", [
         "If y takes a few unordered values it is a classification problem; if it is a "
         "number on a continuous scale it is regression. That single distinction changes "
         "which models are available and which scores make sense."])],
     advice="Put the column you want predicted last, or the app will predict the wrong thing perfectly happily.",
     related=["classification-regression", "feature"])

_add(slug="classification-regression", title="Classification or regression", group="The basics",
     short="Predicting which category, or predicting how much.",
     sections=[("The difference", [
         "Classification predicts a label from a finite unordered set: which species, spam "
         "or not. Regression predicts a number where ordering and distance are meaningful: "
         "disease progression, price.",
         "Formally the difference is the codomain: 𝒴 = {1,…,K} for classification, 𝒴 = ℝ "
         "for regression."]),
       ("Why it cannot be read off the datatype", [
         "Iris species stored as 1, 2, 3 is still classification — species 3 is not three "
         "times species 1, and the average of species 1 and 3 is not species 2. A "
         "temperature stored as text is still regression.",
         "So any automatic detection is a heuristic. This app uses two signals: whether the "
         "column is non-numeric, and how many distinct values it holds relative to the "
         "number of rows. The cutoff grows with √n rather than being fixed, so a "
         "twelve-class problem in a large table is not mistaken for a small regression."]),
       ("Why you are told what it guessed", [
         "Because it can be wrong, and silently. The detected type is always displayed with "
         "an override available — an automatic choice the user cannot see or change is "
         "exactly the AutoML failure lecture 1 warns about."])],
     advice="A text target cannot be forced into regression: there is no arithmetic on strings, and the app will refuse rather than crash.",
     related=["target", "accuracy", "mse"])

_add(slug="model", title="Model", group="The basics",
     short="The family of functions the algorithm is allowed to choose from.",
     sections=[("The formal picture", [
         "Lecture 1 puts supervised learning in four steps, and the first is choosing a "
         "hypothesis class",
         Math("ℋ ⊆ { h : 𝒳 → 𝒴 }"),
         "the set of functions you are willing to consider. Picking “decision tree” means "
         "picking ℋ = all trees; picking ridge regression means ℋ = all linear functions."]),
       ("The remaining three steps", [
         "Choose a loss ℓ measuring how bad a prediction is; choose a penalty R discouraging "
         "complicated members of ℋ; then minimise",
         Math("h* ∈ argmin_{h∈ℋ}  (1/N) Σₙ ℓ(h(xₙ), yₙ)  +  λ R(h)"),
         "Steps 1 to 3 are human choices. Only step 4 is arithmetic — which is the whole "
         "argument of the lecture, and of this app."])],
     advice="Different model families encode different assumptions; trying two and comparing is usually more informative than reasoning about which should win.",
     related=["knn", "tree", "ridge", "automl"])

# ------------------------------------------------------------------- models --

_add(slug="knn", title="k-nearest neighbours", group="Models",
     short="Predicts by looking up the k most similar rows and taking a vote.",
     sections=[("How it works", [
         "To classify a new point, find the $k$ training rows closest to it and return the "
         "majority label among them. Closeness means Euclidean distance by default:",
         Math("d(x, x′) = √( Σ_j (x_j − x′_j)² )"),
         "The sum runs over features, so every column contributes its squared difference, "
         "and the square root converts the total back into the original units. It is the "
         "straight-line distance you would measure with a ruler, generalised past three "
         "dimensions.",
         "That formula is also why the method is so sensitive to scale. A feature measured "
         "in grams contributes differences in the thousands; one in centimetres contributes "
         "differences of a few. Squaring widens the gap further, so without standardising, "
         "the distance is decided by whichever column happens to carry the largest units — "
         "whether or not it carries any signal.",
         "For regression the same neighbours are found and their target values averaged "
         "rather than voted on."]),
       ("What is unusual about it", [
         "There is no training step. The model is the dataset — all the work happens at "
         "prediction time, which makes fitting instant and prediction slow. Methods like "
         "this are called lazy or instance-based learners."]),
       ("Where it struggles", [
         "In high dimensions the distances between all pairs of points become nearly equal, "
         "so “nearest” stops meaning much. This is one face of the curse of dimensionality, "
         "and it is why k-NN works well on a handful of features and poorly on hundreds."])],
     advice="Always standardise before using it; otherwise the distance measures units rather than similarity.",
     related=["hyperparameter-k", "scaling", "model"])

_add(slug="tree", title="Decision tree", group="Models",
     short="A sequence of yes/no threshold tests ending in an answer.",
     sections=[("How it works", [
         "Each node asks whether one feature is below a threshold. Follow the answers down "
         "to a leaf, which carries the prediction. The algorithm chooses the questions "
         "itself by searching, at each node, over every feature and threshold for the split "
         "that best separates the classes.",
         "The usual measure of separation is Gini impurity,",
         Math("G = 1 − Σₖ pₖ²"),
         Aside("pₖ is the proportion of the node's samples in class k. G = 0 when the node is pure."),
         "and the chosen split maximises the weighted drop in G from parent to children."]),
       ("Why it is worth knowing", [
         "It is directly readable — the rules are the model, not a summary of it. It also "
         "needs no scaling, since a threshold test gives the same partition under any "
         "monotone rescaling."]),
       ("Where it fails", [
         "Left unconstrained it grows until every leaf is pure, memorising individual rows. "
         "Trees are also unstable: small changes in the data can produce a completely "
         "different tree with similar accuracy."])],
     advice="Constrain the depth. An unconstrained tree scores 1.0 on the training set and tells you nothing.",
     related=["hyperparameter-depth", "overfitting", "model"])

_add(slug="logistic-regression", title="Logistic regression", group="Models",
     short="A weighted sum of the features, squashed into a probability.",
     sections=[("How it works", [
         "Compute a score wᵀx + b, then map it into (0,1) with the logistic sigmoid,",
         Math("σ(z) = 1 ⁄ (1 + e^{−z})"),
         "giving P(y = 1 | x) = σ(wᵀx + b). For more than two classes the softmax "
         "generalises it, producing one probability per class that together sum to one."]),
       ("Despite the name", [
         "It is a classification method. The “regression” refers to regressing the "
         "log-odds — log(p/(1−p)) — on the features, which is linear in x even though the "
         "probability is not."]),
       ("Why it is a good default", [
         "It is fast, it yields calibrated probabilities rather than bare labels, and the "
         "coefficients are inspectable. It fits a linear decision boundary, so it "
         "underfits genuinely curved problems — which is often a useful thing to discover "
         "early."])],
     advice="Its probabilities are meaningful, so it is a good choice when you need confidence rather than just a label.",
     related=["hyperparameter-c", "scaling", "model"])

_add(slug="svm", title="Support vector machine", group="Models",
     short="Finds the boundary sitting as far as possible from the nearest points of both classes.",
     sections=[("The idea", [
         "Many boundaries separate two classes; the SVM picks the one maximising the margin "
         "— the distance to the closest training point on either side. Those closest points "
         "are the support vectors, and they alone determine the boundary; the rest could be "
         "deleted without changing it."]),
       ("Soft margins", [
         "Real data overlaps, so the strict version has no solution. The soft-margin "
         "formulation allows violations at a price, minimising",
         Math("(1/2)‖w‖²  +  C Σᵢ ξᵢ"),
         Aside("ξᵢ measures how far example i intrudes into the margin, and C sets the price of intrusion."),
         "which is once again lecture 1's shape: a fit term plus a penalty."]),
       ("The kernel trick", [
         "The optimisation depends on the data only through inner products between pairs of "
         "points, so replacing that inner product with a kernel function fits a non-linear "
         "boundary without ever computing coordinates in the higher-dimensional space."])],
     advice="Scale your features first; like k-NN, the SVM is defined through distances.",
     related=["hyperparameter-c", "scaling", "model"])

_add(slug="ridge", title="Ridge regression", group="Models",
     short="Linear regression with a penalty on the size of the coefficients.",
     sections=[("The model", [
         "Predict y ≈ wᵀx, and fit by minimising squared error plus a penalty on ‖w‖² — "
         "which is the worked example running through lecture 1:",
         Math("w* ∈ argmin_w  Σₙ (yₙ − wᵀxₙ)²  +  λ‖w‖²"),
         "The first term is the loss ℓ(y, y′) = (y − y′)², the second is the penalisation "
         "$R(h_w) = ‖w‖²$, and λ is the hyperparameter weighting them."]),
       ("Why penalise at all", [
         "Ordinary least squares is unstable when features are correlated: the fitted "
         "coefficients become large and opposite, cancelling each other, and swing wildly "
         "with small changes in the data. The penalty shrinks them towards zero and "
         "stabilises the fit.",
         "Unlike an ℓ₁ penalty, ‖w‖² shrinks coefficients smoothly without ever setting any "
         "exactly to zero."]),
       ("Why it is the lecture's example", [
         "Ridge is the smallest model that still contains every part of the recipe: a "
         "hypothesis class, a loss, a penalty, and a hyperparameter that must be chosen "
         "from outside the minimisation.",
         "That last piece is what the lecture builds on. Sweeping $λ$ and keeping whichever "
         "value scores best looks like a fully automatic procedure, and the lecture uses it "
         "to show that a human was in fact present at every step of it."])],
     advice="λ is on a logarithmic scale; sweep it by factors of ten rather than in equal steps.",
     related=["hyperparameter-alpha", "mse", "model"])

# ----------------------------------------------------------- hyperparameters --

_add(slug="hyperparameter", title="Hyperparameter", group="Settings",
     short="A setting fixed before training, which the training procedure cannot learn.",
     sections=[("The distinction", [
         "In ridge regression, w is a parameter — found by the minimisation. λ is a "
         "hyperparameter — fixed beforehand and not learned. Lecture 1 is precise about "
         "this: “The coefficient λ is a hyperparameter of the algorithm, which is fixed and "
         "not learned.”"]),
       ("Why it cannot simply be learned too", [
         "Because λ controls the penalty for complexity, and the training loss always "
         "improves as the penalty weakens. Minimising training loss over λ would drive it "
         "to zero every time.",
         "The quantity that identifies a good λ cannot be the quantity already being "
         "minimised — which is exactly why hyperparameters need held-out data and "
         "parameters do not."])],
     advice="Every hyperparameter is a human decision the app is making visible rather than hiding.",
     related=["cross-validation", "validation-curve", "one-se-rule"])

_add(slug="hyperparameter-k", title="k, the number of neighbours", group="Settings",
     short="How many nearby rows vote on each prediction.",
     sections=[("What it controls", [
         "k = 1 asks the single closest row, which follows the training data exactly and "
         "reproduces its noise. Large k averages over a wide region and smooths real "
         "structure away.",
         "It is a direct capacity control: small k gives a jagged, high-variance boundary; "
         "large k gives a smooth, high-bias one."]),
       ("Practical points", [
         "Odd values avoid ties in binary problems. And k cannot exceed the number of rows "
         "in a training fold — which is why the app caps the grid at the smallest fold size "
         "rather than letting the fit fail."])],
     advice="Sweep k and look for the plateau rather than the single best point; the plateau is the honest region.",
     related=["knn", "validation-curve", "overfitting"])

_add(slug="hyperparameter-depth", title="Maximum tree depth", group="Settings",
     short="How many questions the tree may ask before it must answer.",
     sections=[("What it controls", [
         "Depth caps the length of any root-to-leaf path. A tree of depth d has at most 2^d "
         "leaves, so depth constrains complexity only loosely — a depth-5 tree may have "
         "anywhere from 6 to 32 leaves."]),
       ("What happens without it", [
         "The tree splits until every leaf is pure. Training accuracy reaches 1.0 and the "
         "model has effectively memorised the table.",
         "On iris, cross-validated accuracy is statistically flat from depth 3 to 14 while "
         "training accuracy climbs to 1.0 — a textbook picture of capacity buying training "
         "score and nothing else."])],
     advice="Depth 3 to 5 is usually enough for small tabular problems; deeper is rarely better and always less readable.",
     related=["tree", "overfitting", "one-se-rule"])

_add(slug="hyperparameter-c", title="C, the regularisation strength", group="Settings",
     short="How much the model is allowed to contort itself to fit the training data.",
     sections=[("What it means", [
         "C is an inverse penalty: large C means little regularisation and a model that "
         "tries hard to classify every training point correctly; small C means a heavier "
         "penalty and a simpler, smoother boundary.",
         "Both logistic regression and the SVM use it. Note the inversion relative to λ — "
         "roughly C ≈ 1/λ — which is a common source of confusion when moving between "
         "formulations."]),
       ("The trade-off", [
         "Large C tends to overfit; small C tends to underfit. It is usually swept "
         "logarithmically, over values like 0.01, 0.1, 1, 10, 100."])],
     advice="Because C is inverse, a larger value means less regularisation — the opposite of what the name suggests.",
     related=["logistic-regression", "svm", "overfitting"])

_add(slug="hyperparameter-alpha", title="α, the ridge penalty", group="Settings",
     short="How strongly the coefficients are pulled towards zero.",
     sections=[("What it does", [
         "It is the λ of lecture 1's ridge example — scikit-learn calls it alpha. At α = 0 "
         "you get ordinary least squares; as α grows, coefficients shrink towards zero and "
         "the fitted function flattens.",
         "In the limit of very large α every coefficient is zero and the model predicts the "
         "mean of y for every input."]),
       ("Choosing it", [
         "There is no natural scale for $α$, so sweep it logarithmically — 0.001, 0.01, "
         "0.1, 1, 10 — rather than in equal steps. A linear grid spends most of its points "
         "in a region where the penalty barely changes the fit at all."])],
     advice="Ridge is not scale-invariant, so standardise before sweeping α or the penalty falls unevenly across features.",
     related=["ridge", "scaling", "cross-validation"])

_add(slug="seed", title="Random seed", group="Settings",
     short="Fixes the randomness, so the same run gives the same answer.",
     sections=[("Where randomness enters", [
         "Splitting rows into training and test, shuffling before folding, and the internals "
         "of some solvers. All of it derives from a pseudo-random generator, and fixing its "
         "starting state makes the whole pipeline reproducible."]),
       ("Why it matters more than it sounds", [
         "Two runs differing only in seed can give visibly different accuracies on a small "
         "dataset. That variation is not a bug — it is the sampling noise in your estimate, "
         "and seeing it is a useful reminder that a single number is not a measurement.",
         "For reporting, fixing the seed makes results checkable; for judging stability, "
         "varying it is more informative."])],
     advice="Change the seed a few times and watch the accuracy move; that spread is the honest error bar on a single split.",
     related=["cross-validation", "standard-error"])

# ------------------------------------------------------- splitting & scoring --

_add(slug="test-set", title="Training and test sets", group="Measuring",
     short="Rows held back from fitting, used to estimate performance on unseen data.",
     sections=[("Why the split exists", [
         "Error measured on the data used for fitting is systematically optimistic, because "
         "part of what the model learned was the noise particular to those rows. Only data "
         "that played no part in fitting gives an unbiased estimate."]),
       ("The three roles", [
         "Training data fits the parameters. Validation data chooses the hyperparameters. "
         "Test data estimates final performance and is looked at once.",
         "The moment you choose something using the test set, it has participated in fitting "
         "and is no longer a test set. Lecture 1's worked example does exactly this — it "
         "picks λ by lowest test MSE — which makes the reported 2856 optimistic by an "
         "unknown amount. This app selects on cross-validation folds instead."])],
     advice="If you tune, look, adjust and look again, you no longer have a test set — you have a second validation set.",
     related=["cross-validation", "overfitting", "accuracy"])

_add(slug="cross-validation", title="k-fold cross-validation", group="Measuring",
     short="Rotating which slice is held out, so every row is used for both fitting and evaluation.",
     sections=[("The procedure", [
         "Cut the training data into k equal parts. For each fold i: train on the other "
         "k−1 parts and score on part i. Average the k scores.",
         "Every row trains k−1 times and validates exactly once."]),
       ("What it buys", [
         "Two things. The estimate uses all the data rather than wasting a fixed slice; and "
         "because it is an average of k measurements it is less noisy than a single split, "
         "roughly by a factor of √k.",
         "More usefully, the k individual scores give you a spread — so you can compute a "
         "standard error and know how uncertain the estimate is. A single split gives one "
         "number and no way to judge it."]),
       ("Stratification", [
         "For classification, each fold should preserve the class proportions of the whole "
         "dataset. Iris has exactly 50 of each species; a careless split can give one fold "
         "20 setosa and another 5, so the folds measure different problems and averaging "
         "them means less. This app deals each class's shuffled indices round-robin into the "
         "folds, which keeps the proportions right by construction."])],
     advice="k = 5 or 10 is standard. Larger k costs more compute and gives diminishing returns.",
     related=["standard-error", "one-se-rule", "test-set"])

_add(slug="accuracy", title="Accuracy", group="Measuring",
     short="The fraction of predictions that are correct.",
     sections=[("Definition", [
         Math("acc = (1⁄n) Σ_i 1[ ŷ_i = y_i ]"),
         Aside("1[·] is the indicator function: 1 when the condition inside holds, 0 otherwise."),
         "The indicator turns each row into a 1 or a 0, so the sum counts correct "
         "predictions and dividing by $n$ turns that count into a proportion between 0 and "
         "1.",
         "Notice what does not appear: nothing about the prediction survives except whether "
         "it was right. A model that was 51% sure and a model that was 99% sure score "
         "identically. That is why accuracy is easy to read, and also why it discards "
         "information that other measures use."]),
       ("When it misleads", [
         "With imbalanced classes it is dominated by the majority. If 95% of rows belong to "
         "one class, always predicting that class scores 0.95 while being worthless — the "
         "accuracy paradox.",
         "Accuracy also treats every error alike, which is rarely true in practice. In "
         "medical screening a missed case and a false alarm carry very different costs, and "
         "no algorithm can know the ratio: it is a fact about the world, not about the "
         "data."])],
     advice="Always read accuracy beside the class balance, and prefer macro-F1 when the classes are uneven.",
     related=["class-balance", "macro-f1", "confusion-matrix"])

_add(slug="macro-f1", title="Macro-averaged F1", group="Measuring",
     short="Averages per-class performance, so small classes count as much as large ones.",
     sections=[("Building it up", [
         "For one class, precision is the share of predictions for that class that were "
         "right; recall is the share of that class's true members that were found. F1 is "
         "their harmonic mean,",
         Math("F₁ = 2 · (precision · recall) ⁄ (precision + recall)"),
         "The harmonic mean is used because it punishes imbalance: a classifier with "
         "precision 1.0 and recall 0.1 has F1 ≈ 0.18, not 0.55."]),
       ("The macro average", [
         "Compute F1 separately for each class and average with equal weight. A rare class "
         "then contributes as much as a common one, which is exactly what accuracy fails to "
         "do. Micro-averaging, by contrast, pools all predictions first and behaves like "
         "accuracy."])],
     advice="Use it as the default score whenever any class is much rarer than the others.",
     related=["accuracy", "class-balance", "confusion-matrix"])

_add(slug="r-squared", title="R², coefficient of determination", group="Measuring",
     short="The share of variance in the target the model accounts for.",
     sections=[("Definition", [
         Math("R² = 1 − Σᵢ (yᵢ − ŷᵢ)² ⁄ Σᵢ (yᵢ − ȳ)²"),
         "The numerator is the model's squared error; the denominator is the error of "
         "always predicting the mean. So R² = 1 is perfect, R² = 0 is no better than the "
         "mean, and negative values mean worse than the mean — which is possible on held-out "
         "data and is a genuine signal, not a bug."]),
       ("How to read it", [
         "It is unitless, so it compares across datasets in a way MSE cannot. But it says "
         "nothing about whether the errors are large in practical terms — only whether they "
         "are small relative to the spread of y."])],
     advice="Read R² together with MAE, which tells you the size of a typical error in real units.",
     related=["mse", "mae", "classification-regression"])

_add(slug="mse", title="Mean squared error", group="Measuring",
     short="The average squared gap between prediction and truth.",
     sections=[("Definition", [
         Math("MSE = (1⁄n) Σ_i (y_i − ŷ_i)²"),
         "Here $y_i$ is the true value for row $i$ and $ŷ_i$ the prediction, so each term is "
         "one error, squared. Averaging over $n$ rows makes the result independent of "
         "dataset size.",
         "It is the loss $ℓ(y, y′) = (y − y′)²$ from lecture 1’s ridge example averaged "
         "over the data, so minimising MSE and minimising the empirical loss in step 4 of "
         "the recipe are the same operation."]),
       ("Why squared", [
         "Squaring makes every error positive, so errors in opposite directions cannot "
         "cancel. It also penalises large errors disproportionately: being out by 10 counts "
         "a hundred times as much as being out by 1, so the fit works hardest on its worst "
         "cases.",
         "And it is smooth and differentiable everywhere, which is what makes least squares "
         "solvable in closed form rather than by search. The cost of all this is sensitivity "
         "to outliers — a single wild point can dominate the total."]),
       ("Units", [
         "MSE is in the square of the target’s units, which is why its square root, the "
         "RMSE, is often reported instead.",
         "That difference matters when reading a number. An MSE of 2500 sounds enormous "
         "until you take the root: it means a typical error of about 50, and whether 50 is "
         "large depends entirely on what $y$ measures."])],
     advice="MSE compares models on one dataset; it cannot be compared across datasets with different scales.",
     related=["mae", "r-squared", "ridge"])

_add(slug="mae", title="Mean absolute error", group="Measuring",
     short="The average size of the error, in the target's own units.",
     sections=[("Definition", [
         Math("MAE = (1⁄n) Σ_i |y_i − ŷ_i|"),
         "The absolute value replaces the square, so an error of 10 counts ten times an "
         "error of 1 rather than a hundred times. Every error contributes in proportion to "
         "its size and nothing more.",
         "Because there is no squaring, the result stays in the same units as $y$. An MAE "
         "of 3 means the typical prediction is out by about 3 units of whatever $y$ "
         "measures — whereas an MSE of 9 is in squared units and has to be square-rooted "
         "before it means anything."]),
       ("Versus MSE", [
         "MAE is far less sensitive to outliers, since one extreme error is not amplified. "
         "The historical trade-off was that $|·|$ is not differentiable at zero, which made "
         "it harder to optimise.",
         "There is also a statistical difference worth knowing: minimising squared error "
         "predicts the conditional mean of $y$, while minimising absolute error predicts the "
         "conditional median. On a skewed target those are different numbers, and which one "
         "you want is a modelling decision."])],
     advice="Report MAE alongside R²; one gives the scale of the error, the other its significance.",
     related=["mse", "r-squared"])

_add(slug="confusion-matrix", title="Confusion matrix", group="Measuring",
     short="A table of what was predicted against what was true.",
     sections=[("How to read it", [
         "Rows are true classes, columns are predicted ones. The diagonal holds correct "
         "predictions; everything off it is a mistake, and its position says which mistake.",
         "For binary problems the four cells have names — true positive, false positive, "
         "false negative, true negative — and every scalar metric here is some ratio of "
         "them."]),
       ("Why it is worth looking at", [
         "A single accuracy figure cannot tell you whether the model confuses two similar "
         "classes systematically or errs uniformly. Those call for completely different "
         "fixes, and only the matrix distinguishes them."])],
     advice="Look at the matrix before deciding a model is bad; often only one pair of classes is the problem.",
     related=["accuracy", "macro-f1", "class-balance"])

_add(slug="overfitting", title="Overfitting", group="Ideas worth knowing",
     short="Learning the noise in the training data rather than the pattern.",
     sections=[("What happens", [
         "As capacity grows, training error falls monotonically toward zero while test error "
         "falls, bottoms out, then rises. The rise is the model memorising particulars that "
         "will not recur."]),
       ("The decomposition", [
         "Expected squared error splits into three parts,",
         Math("E[(y − ŷ)²] = bias² + variance + irreducible noise"),
         "More capacity lowers bias and raises variance. The best model sits where the sum "
         "is smallest, which is generally not where either term alone is smallest."]),
       ("Seeing it here", [
         "The validation curve plots both training and validation score. The gap between "
         "them is the overfitting, and it widens as capacity grows."])],
     advice="A training score of exactly 1.0 is a warning rather than an achievement.",
     related=["validation-curve", "cross-validation", "hyperparameter-depth"])

# ------------------------------------------------------------ the sweep bits --

_add(slug="standard-error", title="Standard error", group="Ideas worth knowing",
     short="How much a cross-validation score would move if you reshuffled the folds.",
     sections=[("Definition", [
         "From the $k$ fold scores $s_1 … s_k$,",
         Math("SE = s ⁄ √k"),
         Aside("s is the sample standard deviation of the k fold scores, and k the number of folds."),
         "It estimates the standard deviation of the mean itself rather than of the "
         "individual scores — which is why it shrinks as k grows."]),
       ("Why it changes what you conclude", [
         "Two models scoring 0.94 and 0.95 with a standard error of 0.02 are not "
         "distinguishable on this data. Reading the second as better is reading noise.",
         "Reporting a mean without a spread invites exactly that mistake, which is why the "
         "sweep table here shows both."])],
     advice="Treat differences smaller than one standard error as ties, not as rankings.",
     related=["cross-validation", "one-se-rule", "validation-curve"])

_add(slug="one-se-rule", title="The one-standard-error rule", group="Ideas worth knowing",
     short="Among models that are statistically tied, choose the simplest.",
     sections=[("The rule", [
         "Find the best mean cross-validated score and its standard error. Then keep every "
         "candidate whose mean lies within one standard error of that best score, and among "
         "the survivors take the simplest:",
         Math("keep f if  score(f) ≥ max_g score(g) − SE"),
         "The inequality defines a band rather than a point. Everything scoring above "
         "$max_g score(g) − SE$ is inside it, and the rule treats every member of that band "
         "as equally supported by the evidence — which is what a standard error means. Only "
         "after the band is drawn does simplicity break the tie.",
         "“Simplest” has to be defined per model family, and it is a human judgement rather "
         "than something the data supplies: fewer leaves for a tree, larger $k$ for "
         "k-nearest neighbours since more neighbours means more smoothing, smaller $C$ or "
         "larger $α$ for the penalised models. In each case it is the direction of less "
         "capacity."]),
       ("Why not just take the maximum", [
         "Because with many candidates the best mean is partly the luckiest mean. Sweeping "
         "thirty values and taking the argmax selects on genuine quality and on noise "
         "together, and afterwards the two cannot be separated.",
         "If two models are statistically indistinguishable there is no evidence favouring "
         "the complicated one, and the simpler generalises at least as well while being "
         "easier to explain."]),
       ("Why it is in this app", [
         "GridSearchCV takes the argmax silently. This rule encodes a human judgement — "
         "prefer simplicity when the evidence does not distinguish — as an explicit, "
         "auditable step, and the interface shows which candidates tied and which was "
         "chosen."])],
     advice="On iris every tree depth from 3 to 14 ties; the rule takes 3, which is also the only one you can read.",
     related=["standard-error", "cross-validation", "automl"])

_add(slug="validation-curve", title="Validation curve", group="Ideas worth knowing",
     short="Score against hyperparameter value, with training and validation both plotted.",
     sections=[("What it shows", [
         "Two curves. The training score generally rises with capacity. The validation score "
         "rises, plateaus and eventually falls. Where they separate is overfitting; where "
         "the validation curve peaks is the useful setting."]),
       ("Reading it well", [
         "The peak is less informative than the plateau. A flat region means many settings "
         "are equivalent, and the one-standard-error rule takes the simplest of them.",
         "A validation curve that is still rising at the edge of the grid means your grid is "
         "too small, not that the largest value is best."])],
     advice="If both curves are low, the model is underfitting and no hyperparameter will rescue it — change the model family.",
     related=["one-se-rule", "overfitting", "hyperparameter"])

_add(slug="automl", title="Automated machine learning", group="Ideas worth knowing",
     short="Automating model choice, hyperparameters and preprocessing — and what that costs.",
     sections=[("What it automates", [
         "Lecture 1 lists data preparation, model selection, hyperparameter optimisation and "
         "algorithm selection. The lecture's own worked example runs the whole pipeline and "
         "concludes, on a slide of its own: “No human needed!”"]),
       ("The pros and cons the lecture gives", [
         "In favour: it automates time-consuming work where no expertise is needed, and it "
         "searches more thoroughly than a person would.",
         "Against: the user is not in control, and it does not exploit the user's domain "
         "expertise. The lecture quotes Barbudo et al.: most AutoML approaches operate as "
         "black-box methods, so the human must simply rely on the generated models."]),
       ("Where the human actually was", [
         "The lecture answers its own slide. Humans collected the data — with what goal, and "
         "what selection bias? Humans chose the model and the algorithm. Humans chose the "
         "evaluation criterion, and MSE may mean nothing to whoever uses the result. Humans "
         "will use the model, for a purpose the pipeline never saw.",
         "That is why this app has a “decide for me” button and then shows you what it "
         "decided: the contrast is the point."])],
     advice="Use the automatic path first, then look at what it chose. Disagreeing with it is the most useful thing you can do here.",
     related=["one-se-rule", "model", "hyperparameter"])

# ---------------------------------------------------- visualisation concepts --

_add(slug="correlation", title="Correlation", group="Reading your data",
     short="How strongly two features move together.",
     sections=[("The measure", [
         Math("ρ(u,v) = cov(u,v) ⁄ (σ_u σ_v)"),
         "The covariance on top is large and positive when the two features are above their "
         "means together, and negative when one is high while the other is low. By itself it "
         "is in the product of the two features’ units, so it cannot be compared across "
         "different pairs.",
         "Dividing by both standard deviations removes those units and pins the result "
         "inside $[−1, 1]$: $+1$ is a perfect increasing linear relationship, $−1$ a perfect "
         "decreasing one, and $0$ no linear relationship at all."]),
       ("Two warnings", [
         "It captures only linear dependence. Two features can be perfectly related — one "
         "the square of the other — and still show $ρ$ near zero, so a low correlation is "
         "not evidence of independence.",
         "And it says nothing about causation. Correlated features also make linear models "
         "unstable, because weight can shift between them with little change in predictions, "
         "so coefficients should not be read as importances."])],
     advice="Check the correlation matrix before trusting any per-feature importance.",
     related=["feature-ranking", "projection", "feature"])

_add(slug="feature-ranking", title="Feature ranking", group="Reading your data",
     short="Ordering features by how well each separates the classes on its own.",
     sections=[("How it is computed here", [
         "By a one-way ANOVA F-ratio. For a single feature, compare how far apart the class "
         "means are against how spread out the values are inside each class:",
         Math("F = (between-class variance) ⁄ (within-class variance)"),
         "The numerator asks whether the classes sit in different places along this feature. "
         "The denominator asks how noisy each class is around its own mean. A large ratio "
         "means the groups are far apart relative to their internal scatter, which is "
         "exactly the condition for the feature to be useful by itself.",
         "Dividing by the within-class spread also removes the units, so a feature measured "
         "in grams and one measured in millimetres can be ranked against each other."]),
       ("The catch", [
         "It is univariate — each feature is judged alone. Two features that separate the "
         "classes only in combination will both score badly, and two features carrying "
         "identical information will both score well even though one is redundant.",
         "So read the ranking as a guide to what is worth plotting, not as a decision about "
         "what to keep."])],
     advice="Use it to choose which pair of features to scatter, then look at the plot before drawing conclusions.",
     related=["correlation", "projection", "feature"])

_add(slug="projection", title="Projection to two dimensions", group="Reading your data",
     short="Compressing many features into two axes so the data can be drawn.",
     sections=[("Principal component analysis", [
         "PCA finds the directions along which the data varies most and uses the first two "
         "as axes. Each is a weighted combination of the original features, chosen so the "
         "first captures as much variance as possible, the second as much of the remainder "
         "as possible while being orthogonal to the first.",
         "It is computed from the singular value decomposition of the centred data matrix, "
         "and the proportion of variance retained tells you how much the picture leaves "
         "out."]),
       ("How to read the picture", [
         "Well-separated blobs mean the classes are separable in the full space too. "
         "Overlapping blobs are weaker evidence — the overlap may be an artefact of "
         "flattening, and the classes may still be separable in dimensions the plot "
         "discarded.",
         "The axes have no units and no meaning of their own; distances are meaningful, "
         "positions are not."])],
     advice="Standardise before projecting, or the components will simply follow whichever feature has the largest units.",
     related=["scaling", "correlation", "class-balance"])

_add(slug="class-balance", title="Class balance", group="Reading your data",
     short="How many examples each class has, and why a skew changes everything.",
     sections=[("Why it matters first", [
         "It sets the baseline. If 90% of rows are one class, a model predicting only that "
         "class scores 0.90, and any accuracy near that number is meaningless. You cannot "
         "interpret a score without knowing this."]),
       ("What to do about a skew", [
         "Score with macro-F1 rather than accuracy, so the rare class counts. Stratify the "
         "splits, so folds keep the proportions. And consider class weighting during "
         "fitting, which makes errors on the rare class cost more."])],
     advice="Read the class counts before reading any score; it is the first plot worth looking at.",
     related=["accuracy", "macro-f1", "cross-validation"])

_add(slug="scaling", title="Feature scaling", group="Reading your data",
     short="Rewriting features onto a common scale so no one of them dominates by unit alone.",
     sections=[("Standardisation", [
         Math("z = (x − μ) ⁄ σ"),
         "After it every feature has mean 0 and standard deviation 1, so a change of one "
         "means the same amount everywhere. Min–max scaling to [0,1] is the common "
         "alternative and is more sensitive to outliers."]),
       ("Which models care", [
         "k-NN and SVMs are defined through distances, so an unscaled feature in thousands "
         "swamps one in units. Penalised linear models care too, since the penalty treats "
         "all coefficients alike and so depends on the units they are in.",
         "Decision trees do not care at all: a threshold test produces the same partition "
         "under any monotone rescaling."]),
       ("The leakage trap", [
         "μ and σ must be computed on the training fold and applied unchanged to the "
         "validation fold. Computing them over the whole dataset lets information about the "
         "held-out rows influence the fit, which inflates the score."])],
     advice="When in doubt, scale — it never hurts a tree and often rescues everything else.",
     related=["knn", "svm", "projection"])


def groups():
    """Topics arranged by section, in the order they were defined."""
    ordered = {}
    for topic in TOPICS.values():
        ordered.setdefault(topic.group, []).append(topic)
    return ordered
