"""Plain-language explanations for project 2, in the same shape as projects 1 and 3."""

from dataclasses import dataclass, field


@dataclass
class Topic:
    slug: str
    title: str
    short: str
    sections: list
    advice: str = ""
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
    TOPICS[kw["slug"]] = Topic(**kw)


_add(slug="penguins", title="The Palmer Penguins", group="The data",
     short="344 penguins from three species, measured on four body dimensions.",
     sections=[("What is in it", [
         "Body measurements collected at Palmer Station, Antarctica, between 2007 and "
         "2009 by Kristen Gorman: bill length and depth, flipper length and body mass, "
         "plus the island, the sex and the year. The task is to recover the species.",
         "Eleven of the 344 rows are missing at least one measurement and are dropped, "
         "leaving 333. Formally the design matrix is X ∈ ℝ³³³ˣ⁴ for the numeric features, "
         "with the target y taking one of three values."]),
       ("Why it replaced iris", [
         "It is the same shape of problem as Fisher's iris data — three classes, four "
         "continuous measurements — without iris's baggage: Fisher published it in the "
         "Annals of Eugenics, and the dataset has been used to death.",
         "It is also slightly harder in a useful way. Gentoo separates almost trivially "
         "on flipper length and body mass, while Adelie and Chinstrap overlap heavily on "
         "everything except bill length. A model therefore has to do real work on one "
         "boundary while the other is free."])],
     advice="Every number on these pages comes from this one table.",
     related=["accuracy", "test-set", "correlation"])

_add(slug="tree", title="Decision tree", group="Models",
     short="A sequence of threshold tests on single features, ending in a class.",
     sections=[("How it works", [
         "The model asks questions of the form \"is feature j below threshold t?\" and "
         "follows the answer down a branch until it reaches a leaf, which carries a "
         "prediction. \"Is the flipper shorter than 207 mm? If yes, is the bill shorter "
         "than 43 mm? Then Adelie.\"",
         "Nothing about the questions is supplied by you. At each node the algorithm "
         "searches over every feature and every candidate threshold and keeps the split "
         "that most reduces impurity in the two children. scikit-learn's default measure "
         "is the Gini impurity of a node,"]),
       ("The splitting criterion", [
         Math("G = 1 − Σₖ pₖ²"),
         Aside("pₖ is the proportion of the node's samples belonging to class k."),
         "G is 0 when a node holds a single class and rises as the node becomes mixed — "
         "for three equally represented classes it reaches 1 − 3·(1/3)² = 2/3. The split "
         "chosen is the one maximising the weighted drop in G from parent to children.",
         "Entropy, −Σₖ pₖ log pₖ, is the usual alternative and behaves almost identically; "
         "Gini is cheaper because it avoids the logarithm."]),
       ("Why this project uses one", [
         "It is directly readable. The rules are the model, not a summary of it — the "
         "distinction lecture 2 draws between an interpretable model and an explainable "
         "one. There is no gap between what the tree does and what you are told it does."]),
       ("Where it goes wrong", [
         "Grown without constraint, a tree keeps splitting until every leaf is pure, which "
         "on 333 rows means a private rule for each unusual bird. Training accuracy hits "
         "1.0 and test accuracy does not follow."])],
     advice="Read the tree as sentences rather than as a diagram; it is the same model, and the sentences stay readable for longer.",
     related=["leaves", "overfitting", "complexity", "logistic"])

_add(slug="logistic", title="Multinomial logistic regression", group="Models",
     short="A weighted sum of the features per class, turned into probabilities by the softmax.",
     sections=[("The model", [
         "Each class $c$ gets a weight vector $w_c$ and a bias $b_c$. The score for class "
         "$c$ is the linear combination $w_c^⊤x + b_c$, and the scores are converted into "
         "probabilities that sum to one by the softmax function,"]),
       ("The softmax", [
         Math("P(y = c | x)  =  exp(w_cᵀx + b_c) ⁄ Σⱼ exp(w_jᵀx + b_j)"),
         "Exponentiating makes every score positive; dividing by the sum makes them add "
         "to one. The prediction is the class with the largest probability, which is also "
         "the class with the largest score, since exp is increasing.",
         "For two classes this collapses to the familiar logistic sigmoid σ(z) = 1/(1+e⁻ᶻ) "
         "applied to the difference of the two scores."]),
       ("How it is fitted", [
         "By minimising the cross-entropy loss over the training set plus a penalty, which "
         "is exactly the shape of lecture 1's objective:",
         Math("argmin_w  −(1/n) Σᵢ log P(yᵢ | xᵢ)  +  λ‖w‖"),
         "Cross-entropy is used rather than accuracy because the 0–1 loss behind accuracy "
         "is neither convex nor differentiable, so it cannot be optimised directly. "
         "Cross-entropy is a convex surrogate that is minimised by the same predictor."]),
       ("Why it is the contrast to a tree", [
         "A tree partitions the input space into axis-aligned boxes and is constant inside "
         "each. Logistic regression draws straight boundaries and varies smoothly. They "
         "are readable in different ways — one as rules you follow, the other as weights "
         "you compare — which is exactly why the brief asks for a separate complexity "
         "measure for each."])],
     advice="Because it is differentiable, it is also the model for which the ALE derivative can be computed exactly.",
     related=["coefficient", "sparsity", "complexity", "ale"])

_add(slug="leaves", title="Leaves", group="Complexity",
     short="The number of terminal nodes — how many distinct rules the tree contains.",
     sections=[("What it counts", [
         "Every path from the root to a leaf is one rule, and every leaf is one verdict. "
         "A tree with three leaves is three sentences; one with fourteen is fourteen.",
         "For a binary tree the count of leaves L and of internal decision nodes are "
         "related by internal = L − 1, so counting leaves counts questions too."]),
       ("Why it is the right measure here", [
         "Ω is meant to capture how much of a model a person must read. Leaves count "
         "exactly that, which is why the brief fixes Ω(f) = number of leaves for trees.",
         "Depth would be a poor substitute: a tree of depth 5 can have anywhere between 6 "
         "and 32 leaves, so depth constrains complexity only loosely."])],
     advice="Watch where accuracy stops improving as leaves increase — that gap is what the λ slider is for.",
     related=["complexity", "tree", "lambda"])

_add(slug="complexity", title="Complexity, Ω", group="Complexity",
     short="How much of a model you must read before you can say what it does.",
     sections=[("Where it comes from", [
         "Lecture 1 states supervised learning as minimising a penalised empirical loss. "
         "The penalty term R(h) reappears in lecture 2 under the name Ω, with a different "
         "justification: not to prevent overfitting but to promote interpretability."]),
       ("The objective", [
         Math("argmin_f  (1/n) Σᵢ ℓ(f(xᵢ), yᵢ)  +  λ Ω(f)"),
         Aside("Equation (1) of the project brief. It is the same expression as lecture 1's, with R renamed Ω."),
         "Two models can be equally accurate and completely different to live with. Ω puts "
         "a number on that difference so it can be traded against accuracy explicitly "
         "rather than by taste."]),
       ("It means different things per model", [
         "For a tree the brief fixes it as the number of leaves. For logistic regression "
         "the brief asks you to choose — the measure used here is the number of features "
         "with a non-zero coefficient. Both count how many things must be inspected."])],
     advice="Ω is a modelling decision, not a property of the data. A different Ω gives a different frontier.",
     related=["leaves", "sparsity", "lambda", "reachable"])

_add(slug="lambda", title="The trade-off parameter λ", group="Complexity",
     short="The exchange rate between accuracy and simplicity — how much accuracy one extra unit of complexity must earn.",
     sections=[("What the slider does", [
         "Among the trained models, the interface shows the one maximising"]),
       ("The selection criterion", [
         Math("score(f) = acc_test(f) − λ · Ω(f)"),
         "At λ = 0 complexity is free and the most accurate model always wins. As λ grows, "
         "each unit of Ω costs λ points of accuracy, and simpler models overtake.",
         "The units matter: λ is measured in accuracy per leaf. λ = 0.01 says one extra "
         "leaf must buy at least one percentage point of accuracy to be worth having."]),
       ("Why it cannot be computed", [
         "There is no correct λ. It encodes how much a particular person, for a particular "
         "purpose, values being able to read the model. A clinician explaining a decision "
         "to a patient and a researcher maximising held-out accuracy have different λ, and "
         "neither is wrong.",
         "That is the whole reason it is a slider rather than a constant in the code."])],
     advice="Sweep it end to end once. The set of models it can reach is smaller than you expect.",
     related=["complexity", "reachable", "leaves"])

_add(slug="reachable", title="Which models the slider can reach", group="Complexity",
     short="Most trained models can never be selected, at any λ. The selectable ones lie on a concave hull.",
     sections=[("Each model is a line", [
         "Fix a model f. Its score as a function of λ is",
         Math("s_f(λ) = acc(f) − λ·Ω(f)"),
         "which is a straight line with intercept acc(f) and slope −Ω(f). With ten "
         "candidate models you have ten lines, and the slider reports whichever is highest "
         "at the current λ."]),
       ("The consequence", [
         "The winner is always on the upper envelope of the set of lines — the upper "
         "concave hull of the points (Ω(f), acc(f)). Anything strictly inside the hull is "
         "beaten everywhere and can never be selected.",
         "In particular a model that is both less accurate and more complex than another "
         "is dominated and is invisible to the slider no matter what you do.",
         "Typically only three or four of ten trained trees are ever reachable. The others "
         "were trained, scored, and can never be chosen."]),
       ("Reading the frontier", [
         "The λ at which two adjacent hull models swap is the slope of the segment joining "
         "them: Δacc ⁄ ΔΩ. That number is the exact price of the simplification — the "
         "accuracy given up per leaf removed."])],
     advice="The switch points are more informative than the slider position; they say what each simplification costs.",
     related=["lambda", "complexity", "leaves"])

_add(slug="sparsity", title="Sparsity", group="Complexity",
     short="How many features carry a non-zero weight — the complexity measure used for the linear model.",
     sections=[("The measure", [
         "For logistic regression, Ω(f) is taken to be the number of coefficients that are "
         "not zero, written",
         Math("Ω(f) = ‖w‖₀ = #{ j : w_j ≠ 0 }"),
         Aside("The ℓ₀ 'norm' — a count, not a norm in the mathematical sense, since it is not homogeneous."),
         "A feature with weight zero is never consulted, so it need not be understood. The "
         "count is therefore the number of things you must look at, which is the same "
         "quantity leaves measure for a tree."]),
       ("Why not the size of the weights", [
         "‖w‖₂² is the standard ridge penalty, but a model with two hundred tiny "
         "coefficients scores low on it and is completely unreadable. It measures "
         "magnitude, not how much there is to read."]),
       ("How zeros are produced", [
         "Fitting with an ℓ₁ penalty drives coefficients exactly to zero rather than merely "
         "shrinking them. Geometrically the ℓ₁ ball has corners on the axes, and the "
         "optimum tends to land on one. ℓ₂ has no corners and shrinks everything smoothly "
         "without ever reaching zero."])],
     advice="The brief itself calls the slider “sparsity”, which is a strong hint about the intended Ω.",
     related=["coefficient", "complexity", "logistic"])

_add(slug="coefficient", title="Coefficient", group="Models",
     short="The weight a linear model gives one feature for one class.",
     sections=[("Reading one", [
         "In $w_c^⊤x$, the entry $w_{c,j}$ says how strongly feature $j$ pushes towards "
         "class $c$. "
         "Positive pushes towards, negative away, and the magnitude says how hard.",
         "Because the features here are standardised, the coefficients are directly "
         "comparable: each is the change in the class score produced by a one-standard-"
         "deviation change in that feature."]),
       ("What it is not", [
         "A coefficient is not a causal effect and not an importance ranking that survives "
         "correlated features. If flipper length and body mass move together, the fit can "
         "place the weight on either, or split it between them, with little effect on "
         "predictions. Do not read a small coefficient as “this feature does not matter”."])],
     advice="Compare coefficients within one class, not across classes; the softmax is invariant to adding a constant to every class score.",
     related=["logistic", "sparsity", "scaling", "correlation"])

_add(slug="accuracy", title="Accuracy", group="Measuring",
     short="The fraction of test penguins classified correctly.",
     sections=[("Definition", [
         Math("acc = (1/n) Σᵢ 1[ f(xᵢ) = yᵢ ]"),
         Aside("1[·] is the indicator function: 1 when the condition holds, 0 otherwise."),
         "Simple, and adequate here because the three species are reasonably balanced — "
         "roughly 44%, 36% and 20% of the data."]),
       ("When it misleads", [
         "With imbalanced classes accuracy is dominated by the majority class. If one "
         "species were 90% of the data, always predicting it scores 0.9 while being "
         "useless. Macro-averaged F1 or a confusion matrix says more in that case.",
         "Accuracy is also the reason the objective uses cross-entropy for fitting: the "
         "0–1 loss underlying accuracy is neither convex nor differentiable, so it cannot "
         "be optimised directly."])],
     advice="Always read accuracy next to the class proportions; on its own it is not interpretable.",
     related=["test-set", "overfitting", "penguins"])

_add(slug="test-set", title="The test set", group="Measuring",
     short="Rows held back from fitting, used to estimate performance on data the model has not seen.",
     sections=[("Why it exists", [
         "Training error is a biased estimate of future error, because part of what the "
         "model fitted was noise specific to those rows. Only data that played no part in "
         "fitting gives an unbiased estimate."]),
       ("A caveat specific to this project", [
         "The brief's criterion is $acc_test − λΩ(f)$, so the test set is used to choose "
         "which model the slider shows. Strictly, once a set has been used for selection "
         "it is no longer a clean estimate of generalisation — the reported accuracy is "
         "the best of several attempts on that sample.",
         "The brief asks for it this way and it is the right call pedagogically, since the "
         "frontier is the object of interest. It is worth knowing that a fully rigorous "
         "version would select on a validation split and report on a third, untouched set."])],
     advice="Treat the accuracies here as comparable to each other, not as unbiased estimates of field performance.",
     related=["accuracy", "overfitting"])

_add(slug="overfitting", title="Overfitting", group="Measuring",
     short="Fitting the noise in the training data rather than the pattern.",
     sections=[("What happens", [
         "As a model gains capacity, training error falls monotonically towards zero while "
         "test error falls, reaches a minimum, and then rises. The rise is the model "
         "memorising particulars that do not recur.",
         "The decomposition usually quoted is that expected error splits into bias², "
         "variance and irreducible noise. More capacity lowers bias and raises variance; "
         "the minimum sits where the two trade off."]),
       ("How you see it here", [
         "Grow the tree and watch training accuracy climb to 1.0 while test accuracy "
         "plateaus or dips. The gap between the two curves is the overfitting."])],
     advice="A training accuracy of exactly 1.0 on 333 rows is a warning, not an achievement.",
     related=["tree", "test-set", "complexity"])

_add(slug="counterfactual", title="Counterfactual explanation", group="Counterfactuals",
     short="The smallest change to an example that flips the model's prediction to a chosen class.",
     sections=[("The question it answers", [
         "Lecture 3 frames it as what a person actually asks after an adverse automated "
         "decision: what could I have done differently? The canonical example is a loan "
         "refusal answered with “you would have been approved with €10,000 more income”.",
         "Here the same question is asked of a penguin: what would have to change about "
         "this bird for the model to call it a Gentoo?"]),
       ("The formal version", [
         "Wachter et al. pose it as a minimisation balancing achieving the target class "
         "against staying close to the original point,",
         Math("argmin_{x′}  λ ( f̂(x′) − y′ )²  +  d(x, x′)"),
         "The variable being optimised is $x′$, the counterfactual point itself — you are "
         "searching over possible penguins, not over model parameters.",
         "The first term is squared error between the model's output at $x′$ and the target "
         "outcome $y′$, so it is zero when the counterfactual really does get the class you "
         "asked for and grows as it falls short. The second term $d(x, x′)$ is the distance "
         "back to the original example, so it grows as the counterfactual drifts away.",
         "Minimising the sum therefore pulls in two directions at once, and $λ$ sets the "
         "exchange rate: large $λ$ insists on achieving the target class even at the cost "
         "of a distant, implausible point; small $λ$ prefers staying close even if the "
         "flip is only marginal."]),
       ("What is implemented here", [
         "Rather than optimising, this project samples: draw many perturbed points around "
         "x, keep those the model assigns to the target class, and rank them by distance. "
         "It is easier to reason about and easier to make honest, at the cost of needing "
         "enough samples to find anything."]),
       ("An important limitation", [
         "A counterfactual can be valid and useless. “You would be a Gentoo if you had "
         "hatched on another island” is true of the model and impossible in fact. The "
         "literature calls this actionability, and a distance function alone cannot "
         "capture it."])],
     advice="Read the list as “nearest neighbours on the other side of the boundary”, not as advice.",
     related=["mad", "noising", "tree"])

_add(slug="mad", title="MAD-weighted L¹ distance", group="Counterfactuals",
     short="A distance that measures each feature's change in units of that feature's own spread.",
     sections=[("Why a plain distance fails", [
         "Body mass runs to thousands of grams; bill depth spans a few millimetres. Under "
         "an unweighted L¹ or L² distance a 500 g change looks larger than a 2 mm change, "
         "when in penguin terms the second may be the more drastic."]),
       ("The distance used", [
         Math("d(x, x′) = Σⱼ |xⱼ − x′ⱼ| ⁄ MADⱼ"),
         "Each change is divided by how much that feature normally varies, so the sum is "
         "in comparable units across features."]),
       ("Median absolute deviation", [
         Math("MADⱼ = medᵢ | x⁽ⁱ⁾ⱼ − medₗ x⁽ˡ⁾ⱼ |"),
         "The median of the absolute deviations from the median. Replace both medians by "
         "means and the L¹ by an L², and you recover the sample variance — lecture 3 draws "
         "exactly this comparison.",
         "The reason for the median is robustness. A single absurd value inflates a "
         "standard deviation substantially and barely moves the MAD. An inflated scale in "
         "the denominator would make real changes in that feature look negligible."])],
     advice="MAD is computed on the training data once and held fixed; it describes the dataset, not the example.",
     related=["counterfactual", "scaling", "noising"])

_add(slug="noising", title="Noising categorical features", group="Counterfactuals",
     short="Categorical features cannot be nudged, so they are resampled instead.",
     sections=[("Why a nudge is meaningless", [
         "The sampling procedure asks for points near x. For a numeric feature that is "
         "clear: a bill 2 mm longer is a nearby bill.",
         "For island ∈ {Torgersen, Biscoe, Dream} there is no nearby island. There is no "
         "value between Biscoe and Dream, no ordering, and no arithmetic. Adding Gaussian "
         "noise to a one-hot encoding produces 0.3 of an island, which is not a penguin "
         "and is a point the model has never seen anything like."]),
       ("What is done instead", [
         "The perturbation becomes a different kind of operation — a replacement applied "
         "with some probability p, drawing the new level from the empirical distribution "
         "of that feature rather than uniformly.",
         "Both choices matter. Drawing from the empirical distribution keeps proposals "
         "realistic, since rare combinations are proposed rarely. Applying it only with "
         "probability p keeps most draws close to x, which is the point of local sampling; "
         "resampling every categorical feature every time would scatter the proposals."]),
       ("Binary and integer features", [
         "A binary feature is the two-level case of the same rule. Integers that are really "
         "labels — a year of observation, say — are better treated as categorical than as "
         "numbers, since the distance between 2007 and 2009 is not meaningful in the way "
         "two millimetres is."])],
     advice="If no counterfactual is found, widen the search — more samples, or a larger variance — rather than accepting that none exists.",
     related=["counterfactual", "mad"])

_add(slug="pdp", title="Partial dependence plot (PDP)", group="Feature effects",
     short="The average predicted probability as one feature is swept, holding the others at their observed values.",
     sections=[("The definition", [
         "Split the features into the one of interest, $A$, and the rest, $B$. The partial "
         "dependence is the expected model output when $x_A$ is held at a chosen value and "
         "$x_B$ is allowed to vary as it does in the data:",
         Math("PD_f(x_A) = E_{x_B ∼ D|B} [ f(x_A, x_B) ]"),
         "The subscript is the part that matters: the expectation is over the marginal "
         "distribution of $x_B$ — the distribution of the other features on their own, "
         "ignoring what $x_A$ happens to be.",
         "Estimating it needs no theory. Replace the expectation with an average over the "
         "dataset:",
         Math("PD_f(v) ≈ (1⁄n) Σ_i f(v, x_B^{(i)})"),
         "So the recipe is: pin the feature to $v$ on every row, leave every other column "
         "untouched, predict, average. Repeat for each $v$ on a grid and join the points. "
         "That is genuinely the whole algorithm, which is why the brief can reasonably ask "
         "you to write it yourself."]),
       ("The problem", [
         "Because the expectation is over the marginal distribution, the procedure ignores "
         "that $x_B$ depends on $x_A$. Setting $x_A = v$ on every row constructs rows that "
         "may be impossible.",
         "Lecture 3's example is apartment price. The PDP for size evaluates the model at "
         "30 m² on every row, including the rows describing eight-room apartments. You have "
         "asked what a 30 m² apartment with eight rooms costs.",
         "The model answers, because models always answer. That answer is an extrapolation "
         "into a region containing no training data, so it is essentially arbitrary — and "
         "it is then averaged in with the sensible ones, at equal weight."]),
       ("In this dataset", [
         "Flipper length and body mass are strongly correlated: larger birds are larger in "
         "every dimension. The PDP for flipper length therefore asks the model about "
         "penguins with 230 mm flippers and a 3,000 g body — a bird shaped like nothing in "
         "the data.",
         "The distortion is worst at the ends of the range, where the conflict between the "
         "pinned value and the untouched columns is sharpest. A PDP that bends oddly near "
         "its edges is usually showing you extrapolation rather than a real effect."])],
     advice="Compare the PDP against the ALE curve; where they diverge is where correlation was distorting it.",
     related=["ale", "correlation"])

_add(slug="ale", title="Accumulated local effects (ALE)", group="Feature effects",
     short="Averages the model's local slope within narrow bins, then accumulates — so it never evaluates unrealistic combinations.",
     sections=[("The idea", [
         "Instead of asking for the model's value at an arbitrary combination, ask for its "
         "rate of change, using only points that are actually in that neighbourhood. Then "
         "add those local changes up across the range."]),
       ("Formally", [
         Math("ALE_f(x_A) = ∫ E_{x_B | z_A} [ ∂f(z_A, x_B) ⁄ ∂z_A ] dz_A − C"),
         "Read it from the inside out. The innermost piece, $∂f(z_A, x_B) ⁄ ∂z_A$, is the "
         "slope of the model with respect to the feature of interest at a particular point "
         "— how fast the predicted probability moves if you nudge $x_A$ and change nothing "
         "else.",
         "The expectation $E_{x_B | z_A}$ averages that slope over the other features, but "
         "conditionally: only over the values of $x_B$ that actually occur alongside "
         "$z_A$. This is the step that keeps the model away from combinations that do not "
         "exist. A PDP averages over the marginal distribution instead and therefore has "
         "no such protection.",
         "The integral then accumulates those averaged slopes from the left edge of the "
         "range up to $x_A$. Slopes are local statements; the integral turns a sequence of "
         "them into a curve you can read as a level.",
         Aside("C is subtracted so the curve has mean zero. Integrating a derivative leaves an arbitrary constant, and centring is simply a convention for pinning it down."),
         "In practice the integral is a sum. The feature's range is cut into bins; within "
         "each bin the model is evaluated at the two edges for every point that falls "
         "there, and the differences are averaged; those bin-level averages are then "
         "cumulatively summed. So the continuous formula above describes a computation "
         "that is a handful of numpy lines."]),
       ("Why M-plots are not the fix", [
         "The obvious repair to the PDP is to use the conditional distribution, "
         "$E[· | x_A]$, giving an M-plot. It only ever probes realistic points, but it "
         "mixes effects: when flipper length rises, body mass rises with it, so the curve "
         "reflects both. Lecture 3 puts it as “they do not isolate the feature effects of "
         "$x_A$ alone”.",
         "ALE gets both properties at once. Conditioning on $z_A$ keeps the data realistic, "
         "exactly as an M-plot does. Taking a derivative is what removes the correlated "
         "contribution — a derivative measures the change caused by moving $x_A$, and "
         "anything that merely happens to sit alongside $x_A$ contributes no slope."]),
       ("The derivative", [
         "Whether that derivative is available in closed form depends on the model, which "
         "is exactly what the brief asks you to work out.",
         "Logistic regression is differentiable everywhere, and the softmax derivative can "
         "be written down:",
         Math("∂P(y=c|x) ⁄ ∂x_j = P(y=c|x) · ( w_{c,j} − Σ_k P(y=k|x) w_{k,j} )"),
         "The structure is worth reading. The class's own weight $w_{c,j}$ pushes the "
         "probability up, but the average weight across all classes — weighted by their "
         "current probabilities — is subtracted. So what matters is not how much feature "
         "$j$ favours class $c$ in absolute terms, but how much more it favours $c$ than "
         "the competition. That subtraction is why the probabilities continue to sum to "
         "one as the feature moves.",
         "A decision tree gives no such expression. Its prediction is piecewise constant: "
         "flat inside every leaf, jumping at the boundaries. The derivative is therefore "
         "zero almost everywhere and undefined precisely at the split points, which is "
         "where all the behaviour lives.",
         "So for a tree the only honest route is a finite difference — evaluate the model "
         "at both edges of a bin and divide by the width. Both paths are implemented, and "
         "the page states which one produced the curve you are looking at."])],
     advice="ALE curves are centred at zero, so read them as deviations from the average prediction rather than as probabilities.",
     related=["pdp", "correlation", "logistic"])

_add(slug="correlation", title="Correlated features", group="Feature effects",
     short="When two features move together, effects attributed to one may belong to the other.",
     sections=[("The measure", [
         "The Pearson correlation of two features is their covariance divided by the product "
         "of their standard deviations:",
         Math("ρ(u,v) = cov(u,v) ⁄ (σ_u σ_v)"),
         "The covariance in the numerator is large when the two are above their means "
         "together and negative when one is high while the other is low. On its own it is "
         "in the product of the two features’ units, so it cannot be compared across "
         "pairs.",
         "Dividing by both standard deviations strips those units away and confines the "
         "result to $[−1, 1]$: $+1$ is a perfect increasing linear relationship, $−1$ a "
         "perfect decreasing one, $0$ no linear relationship at all.",
         "In these data flipper length and body mass sit high on that scale — bigger "
         "penguins have both."]),
       ("Why it matters for explanations", [
         "It makes the model under-determined in a specific way. Two nearly collinear "
         "features can trade weight between them with almost no change in predictions, so "
         "the fitted coefficients are unstable and should not be read as importances.",
         "It also breaks the PDP, which constructs feature combinations the correlation says "
         "should not occur. ALE exists precisely because of this."]),
       ("What it does not tell you", [
         "Correlation captures only linear dependence. Two features can be perfectly related "
         "— one the square of the other — and still show $ρ$ near zero. And it says nothing "
         "about causation: two features may move together because one drives the other, or "
         "because a third drives both."])],
     advice="Check the correlation matrix before reading any feature-effect plot; it tells you which curves to distrust.",
     related=["pdp", "ale", "coefficient"])

_add(slug="scaling", title="Standardisation", group="Models",
     short="Rewriting each feature as how many standard deviations it sits from the mean.",
     sections=[("The transformation", [
         Math("z = (x − μ) ⁄ σ"),
         "Subtracting the mean $μ$ moves the feature so it is centred on zero; dividing by "
         "the standard deviation $σ$ rescales it so its spread is one. The transformation "
         "is linear and reversible, so no information is lost — only the units change.",
         "Afterwards every feature has mean 0 and standard deviation 1, so a change of 1 "
         "means the same thing everywhere: one standard deviation of that feature as it "
         "occurs in this dataset."]),
       ("Why it is needed", [
         "Logistic regression with a penalty is not scale-invariant: the penalty ‖w‖ "
         "treats all coefficients alike, so a feature measured in grams would be penalised "
         "differently from the same feature measured in kilograms. Without standardising, "
         "which features get driven to zero depends on the units they happen to be in.",
         "Decision trees are unaffected — a threshold test gives the same partition under "
         "any monotone rescaling — which is why the two model families are treated "
         "differently here."]),
       ("A practical point", [
         "μ and σ must be computed on the training set and then applied unchanged to the "
         "test set. Computing them over all the data lets information about the test rows "
         "leak into the fit."])],
     advice="Standardisation is what makes the coefficients comparable to one another, and so what makes the sparsity measure meaningful.",
     related=["coefficient", "sparsity", "mad"])


def groups():
    ordered = {}
    for topic in TOPICS.values():
        ordered.setdefault(topic.group, []).append(topic)
    return ordered
