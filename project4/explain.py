"""Plain-language explanations for project 4, in the same shape as 1, 2 and 3.

These are linked from inside the interface at the moment a word first appears,
not collected in a glossary nobody visits. Participants in a real study would
see the same pages: a study that needs a statistics degree to consent to has
not really obtained consent.
"""

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


def groups():
    out = {}
    for topic in TOPICS.values():
        out.setdefault(topic.group, []).append(topic)
    return out


# --- the data --------------------------------------------------------------

_add(slug="movies", title="The film pool", group="The data",
     short="Around 2,700 well-known films, described only by metadata.",
     sections=[("What is in it", [
         "The IMDB 5000 dataset: roughly five thousand films with genres, release year, "
         "running time, certificate, average score and vote count. It contains no ratings "
         "by individual users at all.",
         "That absence is the reason the project exists. A conventional recommender learns "
         "from a matrix of user-by-item ratings; here there is no such matrix, and nothing "
         "to learn from until somebody sits down and tells us what they like."]),
       ("Why it was trimmed", [
         "Films with fewer than 25,000 votes are dropped. Somebody comparing two films "
         "they have never heard of is not expressing a preference — they are reading the "
         "card and guessing, and that guessing enters the model as response noise "
         "attributed to their taste.",
         "That is a validity decision rather than data cleaning. About 2,700 films survive, "
         "spanning 1927 to 2016."])],
     advice="Every film shown to you is drawn uniformly at random from this pool.",
     related=["cold-start", "feature-vector", "validity"])

_add(slug="cold-start", title="The cold start", group="The data",
     short="A new user with no history — the situation this whole project addresses.",
     sections=[("The problem", [
         "You have just signed up. The system knows nothing about you, so it cannot say "
         "“people like you enjoyed this” — it has no idea who is like you.",
         "Collaborative filtering, which learns from the pattern of other users' ratings, "
         "has nothing to work with. The only remaining option is to ask directly."]),
       ("The constraint", [
         "You will not answer two hundred questions. Realistically the system gets twenty "
         "or so, which is why the whole design is shaped by the budget rather than by what "
         "would best describe a film."])],
     advice="This is the question the user study is built to answer: which way of asking learns more per minute.",
     related=["elicitation", "movies", "designs"])

# --- the model -------------------------------------------------------------

_add(slug="feature-vector", title="Feature vector, x", group="The model",
     short="A film reduced to 20 numbers.",
     sections=[("What it holds", [
         "Fourteen genre indicators, each 0 or 1; four standardised continuous features — "
         "how recent, how long, how well reviewed, how widely seen; and two further "
         "indicators for adult certification and black-and-white. So x ∈ ℝ²⁰.",
         "The model sees nothing else. Anything not in x is invisible to it."]),
       ("Why so few", [
         "Every dimension needs evidence. Twenty weights can be pinned down by twenty-odd "
         "answers; two thousand — one per director — could not be pinned down by any "
         "number of answers a person would tolerate giving.",
         "A second rule: if it is not on the card, it is not in x. You cannot have chosen "
         "a film for its budget if the budget was never shown to you."])],
     advice="The features page lists all twenty dimensions and everything deliberately left out.",
     related=["preference-vector", "utility", "identifiability", "standardise"])

_add(slug="preference-vector", title="Preference vector, w", group="The model",
     short="One number per feature describing your taste — the quantity being estimated.",
     sections=[("What it is", [
         "For each of the twenty features, a weight saying how much that feature appeals to "
         "you. Positive means you like it, negative means you avoid it, and magnitude says "
         "how strongly. w ∈ ℝ²⁰, and it differs from person to person.",
         "“Loves horror, dislikes long films, mildly prefers older ones” is a description "
         "of a w."]),
       ("Why it must be inferred rather than asked", [
         "Nobody can state their own w. Ask someone their weight on “how widely seen a film "
         "is” and you will get a shrug. But ask them to choose between two films twenty "
         "times and the number falls out.",
         "It is also never observed for any real person, which is why the study cannot "
         "measure how close an estimate is and has to measure prediction instead."])],
     advice="Your fitted w is shown at the end of the study.",
     related=["utility", "feature-vector", "holdout", "map"])

_add(slug="utility", title="Utility, U(x) = wᵀx", group="The model",
     short="A single score per film for a given person: multiply and add.",
     sections=[("The model", [
         Math("U(x) = wᵀx = Σⱼ wⱼ xⱼ"),
         "Take each of the film's twenty numbers, multiply by your weight for that feature, "
         "sum. Higher means you should prefer it. That is the entire preference model — "
         "everything else is about recovering w."]),
       ("What it assumes", [
         "That features contribute additively and independently. Liking horror and liking "
         "short films means liking short horror films by exactly the sum of the two, no "
         "more. Real taste has interactions this cannot express.",
         "It is a deliberate simplification, and it is what makes twenty answers enough. "
         "Lecture 9's remark applies: all models are wrong, but some are useful."]),
       ("An identifiability note", [
         "Only differences in utility matter to any of the models here, so adding a "
         "constant to every film's utility changes nothing observable."])],
     advice="Two films with equal utility should feel like a coin toss to you — that is what the model predicts.",
     related=["preference-vector", "bradley-terry", "feature-vector"])

_add(slug="bradley-terry", title="The Bradley–Terry model", group="The model",
     short="Converts two utilities into the probability of choosing one over the other.",
     sections=[("The model", [
         Math("P(i ≻ j) = e^{U_i} ⁄ ( e^{U_i} + e^{U_j} ) = σ( wᵀ(xᵢ − xⱼ) )"),
         Aside("σ(z) = 1/(1 + e⁻ᶻ) is the logistic sigmoid."),
         "Equal utilities give one half. As one film pulls ahead the probability rises "
         "smoothly towards, but never reaching, certainty. A gap of one utility point is "
         "about 73/27; a gap of three is about 95/5."]),
       ("Where it comes from in the course", [
         "Lecture 9 gives the general Luce model for choice among finitely many options,",
         Math("p(a | s, θ) = U(a | s, θ) ⁄ Σ_{a′} U(a′ | s, θ)"),
         "and notes that with two actions and utility $exp(αθ^⊤q_a)$ this is the logit model. "
         "Bradley–Terry is exactly that two-option case."]),
       ("Why not deterministic", [
         "Because people are not consistent, and a model asserting the higher utility always "
         "wins cannot learn from being contradicted. Allowing surprising answers is what "
         "lets a handful of responses say anything useful."])],
     advice="This is the model the brief supplies. Task 2 is about extending it past two options.",
     related=["plackett-luce", "utility", "iia"])

_add(slug="plackett-luce", title="The Plackett–Luce model", group="The model",
     short="Bradley–Terry extended from a single pair to a full ranking.",
     sections=[("The construction", [
         "Read a ranking not as one object but as a sequence of choices: a favourite out of "
         "ten, then a favourite of the nine remaining, then of the eight after that. Each "
         "stage is an ordinary Luce choice over whatever is still available, so the "
         "probability of the whole ranking is the product of the stages:",
         Math("P(i_1 ≻ i_2 ≻ ⋯ ≻ i_n) = ∏_k  e^{U_{i_k}} ⁄ Σ_{l ≥ k} e^{U_{i_l}}"),
         "Take the pieces in turn. The numerator $e^{U_{i_k}}$ is the exponentiated utility "
         "of the film actually chosen at stage $k$. The denominator sums that same quantity "
         "over $l ≥ k$ — over the films still in the running at that stage, the chosen one "
         "included.",
         "The condition $l ≥ k$ is the whole mechanism. At stage 1 the sum runs over all "
         "ten films; at stage 2 the first has been removed so it runs over nine; and so on. "
         "Each factor is a softmax over a shrinking set, which is what choosing without "
         "replacement means written as probability.",
         "The product runs to $n − 1$, not $n$. Once nine films are placed the tenth is "
         "determined, and its factor would be a sum over a single item divided by itself — "
         "that is, 1. Multiplying by it changes nothing, so it is dropped.",
         Aside("With ten films this gives nine factors, so a single ranking contributes nine terms to the log-likelihood rather than one."),
     ]),
       ("Why this is the right extension", [
         "At n = 2 the product has a single factor and is exactly Bradley–Terry, so nothing "
         "was replaced.",
         "Its pairwise marginals are the Bradley–Terry probabilities: the chance that i "
         "appears before j in a sampled ranking is $e^{U_i} ⁄ (e^{U_i} + e^{U_j})$, whatever the "
         "other films in the set. This is what makes the study legitimate — both interfaces "
         "estimate the same w, so scoring them on the same held-out pairs is a fair test "
         "rather than a category error."]),
       ("The alternative that fails", [
         "A ranking of ten implies C(10,2) = 45 pairwise preferences, and it is tempting to "
         "treat them as 45 independent observations and multiply. This is wrong twice: the "
         "comparisons are dependent, since i₁ ≻ i₃ is partly entailed by i₁ ≻ i₂ and "
         "i₂ ≻ i₃, so the same evidence is counted repeatedly; and the product does not sum "
         "to one over the n! orderings, so it is not a likelihood at all.",
         "Plackett–Luce sums to one by construction, being a product of normalised choices."])],
     advice="The full derivation and its justification are in the PDF report.",
     related=["bradley-terry", "ranking", "iia", "fisher"])

_add(slug="iia", title="Luce's choice axiom", group="The model",
     short="The assumption that adding options does not change the relative odds of the others.",
     sections=[("The property", [
         "Independence of irrelevant alternatives: the odds of preferring $i$ to $j$ do not "
         "depend on what else happens to be in the set.",
         Math("P(i ≻ j) ⁄ P(j ≻ i) = e^{U_i} ⁄ e^{U_j}"),
         "The reason is visible in the Luce model itself. Both probabilities are formed by "
         "dividing by the same normalising sum over the whole choice set, so taking their "
         "ratio cancels it. Whatever else is present scales numerator and denominator "
         "identically and drops out.",
         "That cancellation is convenient — it is what makes the pairwise marginals clean, "
         "and therefore what lets the two study designs be compared at all. It is also a "
         "strong empirical claim about how people behave, and it is false."]),
       ("Where it fails", [
         "The standard counterexample is the red-bus/blue-bus problem. Someone choosing "
         "between a car and a bus, indifferent at 50/50, should not become less likely to "
         "take the car merely because the bus company repaints half its fleet. The axiom "
         "says the car keeps a third of the vote once there are three options; intuition "
         "says it keeps half, because the two buses are the same option wearing different "
         "paint.",
         "The film version: put two near-identical sequels into a set of ten and people tend "
         "to push both down, splitting their appeal between them. Bradley–Terry and "
         "Plackett–Luce cannot represent that, because it is precisely what the axiom "
         "forbids."]),
       ("Why it matters for this study", [
         "Design 2 shows ten films at once and is therefore far more exposed to the "
         "violation than Design 1, which never shows more than two. So the flaw is "
         "confounded with the manipulation being tested — any disadvantage measured for "
         "ranking might be the interface, or might be the model failing on ranked sets.",
         "That is recorded as a threat to validity rather than fixed, since fixing it would "
         "mean abandoning the Luce family the brief specifies."])],
     advice="If a set of ten ever felt like it contained duplicates, this is why that matters.",
     related=["plackett-luce", "bradley-terry", "validity"])

# --- fitting ---------------------------------------------------------------

_add(slug="map", title="How w is fitted", group="Fitting",
     short="Choose the w that makes your actual answers most probable, pulled gently toward zero.",
     sections=[("The objective", [
         "Every candidate $w$ assigns a probability to the answers you actually gave. Take "
         "the one that maximises it, penalised so the search stays finite:",
         Math("ŵ = argmax_w  Σ log P(response | w)  −  ‖w‖² ⁄ (2σ²)"),
         "The first term is the log-likelihood — the Plackett–Luce probability of your "
         "answers, summed over responses because independent probabilities multiply and "
         "logs turn products into sums. The second is the penalty from the prior, which "
         "grows with the squared length of $w$ and so charges for extreme opinions.",
         "This is maximum a posteriori estimation: maximum likelihood with a prior attached. "
         "Setting $σ → ∞$ removes the penalty and recovers plain maximum likelihood."]),
       ("How it is found", [
         "By gradient ascent. Differentiating the log of the Plackett–Luce product gives a "
         "form worth reading:",
         Math("∇_w = Σ_k ( x_{i_k} − Σ_{l ≥ k} p_l x_{i_l} )"),
         "At each stage the gradient is the chosen film\u2019s features minus a "
         "probability-weighted average of the features still available. So it is a "
         "comparison: what you picked, against what the model currently expected you to "
         "pick.",
         "If the model already gives almost all the probability to the film you chose, that "
         "weighted average is nearly the chosen film\u2019s own features, the difference is "
         "close to zero, and your answer moves $w$ hardly at all. The gradient is largest "
         "when you choose something the model thought unlikely — the formal version of the "
         "obvious idea that surprising answers are the informative ones.",
         "The objective is strictly concave: a concave log-likelihood plus a strictly "
         "concave quadratic penalty. A vanishing gradient therefore certifies the global "
         "optimum, with no local maxima to escape, which is why plain gradient ascent with a "
         "line search suffices and no cleverer optimiser is needed."])],
     advice="More answers means the pull toward zero matters less, so a longer session gives sharper weights.",
     related=["prior", "preference-vector", "identifiability"])

_add(slug="prior", title="The prior", group="Fitting",
     short="An assumption of no strong opinions, overruled by evidence.",
     sections=[("What it assumes", [
         "Before you answer anything, w is taken to be drawn from a spherical Gaussian "
         "centred at zero,",
         Math("w ∼ 𝒩(0, σ²I),   σ = 1"),
         "which contributes the ‖w‖²/(2σ²) term above. σ = 1 says a one-standard-deviation "
         "move in a feature is worth about one unit of utility — roughly a 73/27 preference. "
         "A weak claim, deliberately."]),
       ("Why it is necessary rather than optional", [
         "Without it the estimate frequently does not exist. With twenty dimensions and "
         "twenty-five responses there are usually directions in w that no response "
         "constrains, and the likelihood increases without bound along them.",
         "This is the same phenomenon as separation in logistic regression: when the data "
         "can be perfectly divided, the maximum-likelihood coefficients diverge. The penalty "
         "is not a refinement; it is what makes the problem well posed."]),
       ("Its lineage in the course", [
         "It is lecture 1's step 3 — the penalisation R(h) — appearing for the third time. "
         "Project 1 used it against overfitting, project 2 as a complexity measure, and "
         "here as a prior making estimation possible from very little data."])],
     advice="It is why your fitted profile looks moderate even if your answers felt decisive.",
     related=["map", "identifiability"])

_add(slug="identifiability", title="Identifiability", group="Fitting",
     short="Whether the data could determine a parameter at all, even in principle.",
     sections=[("The problem", [
         "Add one weight per director — 2,398 of them. Two films drawn at random share a "
         "director with probability well under 0.001, so essentially every response "
         "multiplies those weights by zero. No number of answers would ever constrain them.",
         "They are not weak features. They are unmeasurable at this budget, and including "
         "them merely gives the prior room to invent preferences nobody expressed."]),
       ("The rule used for genres", [
         "A genre earns a dimension only if random pairs actually differ on it. For "
         "prevalence p that happens with probability",
         Math("P(differ) = 2p(1 − p)"),
         "Requiring one informative pair in ten gives 2p(1−p) ≥ 0.1, whose lower root is "
         "p ≥ 0.0528. Fourteen genres clear it; Western, at p ≈ 0.02, would be informative "
         "in about one pair in twenty-six — one observation across a whole session."])],
     advice="The features page shows the threshold and everything that fell below it.",
     related=["feature-vector", "prior", "map"])

_add(slug="standardise", title="Standardising", group="Fitting",
     short="Putting features on a common scale so their weights mean the same thing.",
     sections=[("The transformation", [
         Math("z = (x − μ) ⁄ σ"),
         "Year, running time, score and vote count are in wildly different units. After "
         "standardising, a weight of 0.5 means the same amount of preference on any of "
         "them, and a single spherical prior over w is a coherent assumption."]),
       ("Why genre flags are left alone", [
         "Scaling a genre with prevalence 0.06 to unit variance makes flipping it a roughly "
         "four-standard-deviation move, so the same weight would produce a far larger "
         "utility swing than for a common genre. An isotropic prior would then permit rare "
         "genres — the ones with least evidence — the largest effects, which is backwards.",
         "Left as plain 0/1, $w_g$ is simply the utility of carrying the tag, and every genre "
         "is shrunk equally."])],
     advice="It is also what lets the profile bars at the end be compared with one another.",
     related=["feature-vector", "prior"])

_add(slug="fisher", title="How much a task is worth", group="Fitting",
     short="A measure of how sharply one answer narrows down w — and a warning about reading it naively.",
     sections=[("The quantity", [
         "Fisher information measures how much an observation constrains a parameter. For a "
         "softmax choice over a set S with probabilities p_s,",
         Math("I = Σ_{s∈S} p_s x_s x_s^⊤  −  x̄ x̄^⊤"),
         Aside("with $x̄ = Σ_s p_s x_s$, the average feature vector under the choice distribution."),
         "It is the covariance of the features under the choice distribution. A ranking "
         "contributes one such term per stage; a pairwise comparison is the n = 2 case."]),
       ("The naive reading, and why it misleads", [
         "One ranking of ten is worth around sixty pairwise comparisons by trace. That "
         "looks decisive until the budget is fixed by time rather than by task count: eight "
         "minutes buys roughly eighty comparisons or five rankings, and the accumulated "
         "traces then come out at about 118 against 129 — a ratio of 1.09, not 60."]),
       ("Why the trace is the wrong summary", [
         "Total information is not the same as useful information. Eighty comparisons touch "
         "160 different films and constrain w in many directions. Five rankings touch fifty, "
         "and the 45 comparisons inside one ranking all lie in the span of that set's ten "
         "feature vectors.",
         "Measured by log-determinant — the D-optimality criterion, which governs the volume "
         "of the confidence region — ranking comes out 17.6 nats lower, with a smallest "
         "eigenvalue about six times smaller. Whole directions of taste go unconstrained, "
         "and the prior supplies the answer there."])],
     advice="This reversal is why the study's hypothesis favours pairwise; the arithmetic that seemed to favour ranking does not survive the time correction.",
     related=["plackett-luce", "designs", "confound"])

# --- the study -------------------------------------------------------------

_add(slug="elicitation", title="Preference elicitation", group="The study",
     short="Asking a few well-chosen questions to infer what someone likes.",
     sections=[("What it means", [
         "Not inferring taste from past behaviour — there is none yet — but asking directly, "
         "in a form people can actually answer.",
         "People are poor at stating parameters and good at making comparisons. Elicitation "
         "is the business of converting the second into the first."])],
     advice="Both designs in this study are elicitation methods; which one works better is the question.",
     related=["cold-start", "designs", "bradley-terry"])

_add(slug="designs", title="The two designs", group="The study",
     short="Design 1: choose one of two. Design 2: put ten in order.",
     sections=[("Design 1 — pairwise choice", [
         "Two films, pick one, repeat. Each answer carries little information but the task "
         "is fast, easy and hard to get wrong. Modelled directly by Bradley–Terry."]),
       ("Design 2 — rank ten", [
         "Ten films dragged into order. One answer carries far more, but the task is slow "
         "and ordering the middle of a list of ten is genuinely difficult. Needs the "
         "Plackett–Luce extension."]),
       ("Why the answer is not obvious", [
         "Ranking wins decisively on information per task. Pairwise wins on tasks per "
         "minute, and on how carefully each answer is given. Which effect dominates is an "
         "empirical question about people rather than a fact about the algebra."])],
     advice="You were assigned one of these at random when you started.",
     related=["fisher", "confound", "between-subjects"])

_add(slug="holdout", title="The held-out block", group="The study",
     short="Ten fresh comparisons at the end, used to score how well your taste was learned.",
     sections=[("Why it exists", [
         "The natural outcome measure is how close the estimated ŵ is to your true w. It "
         "cannot be computed: w is latent and never observed for a real person, so there is "
         "no ground truth to compare against.",
         "What can be measured is prediction. Fit w on the elicitation answers alone, then "
         "see how well it calls ten comparisons it never saw."]),
       ("Why it is identical in both arms", [
         "Same task, same count, same wording, whichever design you were given. If the test "
         "itself differed between arms, any difference in the result could be the test "
         "rather than the design being tested."])],
     advice="Two of the held-out pairs are repeats of earlier ones with the sides swapped.",
     related=["log-loss", "preference-vector", "attention-check"])

_add(slug="log-loss", title="Log loss", group="The study",
     short="Scores a prediction on its confidence, not merely on whether it was right.",
     sections=[("Definition", [
         Math("ℒ = −(1/m) Σ log σ( wᵀ(x_chosen − x_rejected) )"),
         "Predicting the chosen film with 90% confidence scores better than predicting it "
         "with 55%; predicting the wrong one confidently scores much worse than getting it "
         "wrong hesitantly."]),
       ("The reference value", [
         "A model that has learned nothing answers 0.5 every time and scores −log(0.5) = "
         "log 2 ≈ 0.693. Below that line something was learned; above it, worse than a coin."]),
       ("Why not accuracy", [
         "On ten held-out items accuracy takes eleven possible values and discards all the "
         "confidence information, so it needs far more participants to separate two arms. "
         "Log loss uses the whole prediction and is therefore the more powerful test at a "
         "fixed sample size."])],
     advice="0.693 is the line to beat, and it is marked on your debrief.",
     related=["holdout", "power"])

_add(slug="between-subjects", title="Between subjects", group="The study",
     short="Each participant does one design, not both.",
     sections=[("The trade-off", [
         "A within-subjects design, where everyone does both, is more sensitive — each "
         "person acts as their own control, so differences between people drop out of the "
         "comparison. It also needs fewer participants.",
         "It is rejected here for a reason no counterbalancing removes: having ranked ten "
         "films, a participant understands their own taste better and would approach the "
         "second condition differently for that reason alone. Order effects can be "
         "balanced; genuine learning about oneself cannot be undone."]),
       ("The cost", [
         "Between-subjects needs more participants, because person-to-person variability now "
         "sits on top of the effect being measured. That is what the power calculation "
         "accounts for."]),
       ("Blocking", [
         "Assignment is randomised within strata defined by viewing frequency. Frequent "
         "viewers have firmer preferences and lower response noise, and leaving that to "
         "simple randomisation risks it landing unevenly across the two arms."])],
     advice="You were assigned at random; the report explains the blocking used in the full study.",
     related=["designs", "power", "confound"])

_add(slug="confound", title="The time confound", group="The study",
     short="Comparing 40 pairs against 8 rankings compares two arbitrary numbers.",
     sections=[("The trap", [
         "Give one arm forty comparisons and the other eight rankings, and whichever wins, "
         "the finding is about those two numbers. Change them to eighty and four and the "
         "result may reverse.",
         "A confound is any variable that differs between conditions alongside the one you "
         "meant to manipulate. Here task count would differ, so the design and the amount "
         "of work would be varied together and could not be separated."]),
       ("What the study does", [
         "Both arms receive the same wall-clock elicitation budget and complete as many "
         "tasks as fit. What a recommender actually spends is the user's patience, and "
         "eight minutes is eight minutes however it is divided.",
         "A consequence worth noting: the number of tasks is then an outcome of the "
         "experiment rather than a setting of it, and it will vary between participants."])],
     advice="The timer on the study page is that budget running down.",
     related=["designs", "fisher", "validity"])

_add(slug="power", title="How many participants", group="The study",
     short="Deciding the sample size before running anything.",
     sections=[("The calculation", [
         "For a two-sample comparison of means with standardised effect size d, one-sided α "
         "and power 1−β, the required sample per arm is",
         Math("n = 2 ( z_{1−α} + z_{1−β} )² ⁄ d²"),
         "With d = 0.5, α = .05 and power 0.8 this gives 2(1.645 + 0.842)²/0.25 ≈ 49.5, so "
         "50 per arm — inflated for exclusions and dropout to 75 per arm, 150 recruited."]),
       ("Why it must come first", [
         "Deciding afterwards how many participants you needed is how researchers find "
         "whatever they were hoping for. Choose d by asking how big a difference would "
         "actually change what someone builds; a difference too small to act on is not "
         "worth powering for."]),
       ("Pre-registration", [
         "The hypothesis, the primary outcome, the exclusion rules and the statistical test "
         "are all recorded before the first participant. Otherwise a dozen defensible "
         "analyses exist and the one that gets reported is the one that worked."])],
     advice="Section 4 of the PDF works the calculation through.",
     related=["log-loss", "between-subjects", "validity"])

_add(slug="attention-check", title="Attention checks", group="The study",
     short="Repeated questions that reveal who was clicking at random.",
     sections=[("How they work here", [
         "Two of the held-out pairs return a second time with the films swapped left to "
         "right. Someone genuinely deciding will mostly answer the same way twice; someone "
         "clicking through will not.",
         "The swap matters. A participant favouring one side out of habit would pass an "
         "identical re-presentation and fails this one."]),
       ("Why the rule is fixed in advance", [
         "Disagreeing on both repeats is the pre-registered exclusion criterion. Deciding it "
         "before collection is what stops it becoming a way to remove whoever spoiled the "
         "result."])],
     advice="Your own consistency is reported on the debrief page.",
     related=["holdout", "power", "validity"])

_add(slug="validity", title="Threats to validity", group="The study",
     short="The ways a study can give the right answer to the wrong question.",
     sections=[("Internal and external", [
         "Internal validity asks whether the observed difference was really caused by the "
         "manipulation. External validity asks whether the finding generalises beyond the "
         "people and setting studied. Randomisation buys the first; it does nothing for the "
         "second."]),
       ("The ones that matter here", [
         "Unrecognised films turn preference into guessing — handled by the vote threshold "
         "on the pool, though not eliminated.",
         "Ranking ten items invites carelessness toward the bottom of the list, so the "
         "fitted w may partly be learning noise from positions eight to ten. The analysis "
         "models position-dependent noise rather than assuming it away.",
         "Sets of ten can contain near-duplicates, which the model is formally unable to "
         "handle, and this affects Design 2 specifically — so it is confounded with the "
         "manipulation.",
         "Participants recruited online are not a random sample of humanity, so the finding "
         "is about these designs on that population."])],
     advice="A study listing none of these has not looked for them.",
     related=["iia", "attention-check", "power"])

_add(slug="consent", title="Informed consent", group="The study",
     short="You are told what will happen and what is recorded before anything begins.",
     sections=[("What is recorded", [
         "Which design you were given, which films you chose or how you ordered them, how "
         "long each task took, and one coarse answer about how often you watch films. "
         "Nothing identifying — no name, no email, no IP address, and no cookie beyond the "
         "one keeping the session alive.",
         "Under GDPR that makes the dataset pseudonymous, with consent as the lawful basis "
         "and withdrawal deleting the record."]),
       ("What you can do", [
         "Stop at any point using the button on the study page; the run ends there and what "
         "you have already answered is still fitted. In the real study, withdrawing deletes "
         "the data and payment is made regardless."]),
       ("Why these pages exist", [
         "Lecture 7 is explicit that even a seemingly harmless study is a study on human "
         "beings. Consent that requires a statistics degree to understand is not informed "
         "consent, which is why every technical term on the participant-facing screens links "
         "to an explanation."])],
     advice="Nothing here is required of you; you can close the tab at any point.",
     related=["movies", "designs", "validity"])

_add(slug="ranking", title="Ranking ten films", group="The study",
     short="Ordering a list from most to least preferred.",
     sections=[("How to do it", [
         "Drag a row to move it, or use the up and down buttons. The buttons are not a "
         "fallback for the clumsy: dragging is unusable with a keyboard or a screen reader, "
         "and an interface working only one way excludes participants — invisibly, since "
         "they appear in the data as dropouts.",
         "Position one is your favourite, position ten your least preferred."]),
       ("Doing it honestly", [
         "The top and bottom are usually easy and the middle is genuinely hard. Roughly "
         "right is fine — the model expects imprecision, and measuring how much of it there "
         "is happens to be one of the study's secondary questions."])],
     advice="Reorder as much as you like before submitting; nothing is recorded until you do.",
     related=["plackett-luce", "designs", "iia"])
