"""Plain-language explanations for project 3.

Same idea as project 1: nobody should have to already know what a word means in
order to use the page it appears on. Written for someone who has not met any of
this before.
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
    topic = Topic(**kw)
    TOPICS[topic.slug] = topic


_add(
    slug="deferral", title="Deferring", group="The idea",
    short="The system declines to answer and hands the case to a human instead.",
    sections=[("The setting", [
        "Rather than choosing between full automation and none, build a team: the "
        "classifier answers what it can and passes the rest to an expert. Lecture 5 "
        "motivates this with legal judgment prediction, where four things make full "
        "automation unacceptable — the decision carries real risk, unfamiliar cases are "
        "misclassified at high rates, the model may reproduce biases in its data, and the "
        "human often has context the model never saw."]),
      ("The loss", [
        "In the rejection formulation the system pays a fixed cost c(x) for abstaining. "
        "Deferral replaces that abstract cost with something concrete: whatever the human "
        "then gets wrong. Writing m(x) for the expert's prediction,",
        Math("L(h, r, x, y) = 1[y ≠ h(x)]·1[not deferred]  +  1[y ≠ m(x)]·1[deferred]"),
        "You pay for being wrong when you answer, and for the expert being wrong when you "
        "hand over. An optional extra term c(x) charges for the expert's time."]),
      ("The observation that makes this hard", [
        "Lecture 5 states it directly: this is not a three-class problem with labels "
        "world, sports and pass. There is no label anywhere saying whether an article "
        "should have been deferred — nobody ever wrote one down.",
        "The training signal for the deferral decision has to be constructed out of things "
        "that can be observed: whether the classifier was right, and whether the expert "
        "was."])],
    advice="If a deferral rule only ever looks at the classifier's confidence, it is solving rejection, not deferral.",
    related=["expert", "advantage", "threshold", "outcomes"])

_add(
    slug="expert", title="The simulated expert", group="The idea",
    short="A stand-in for a human annotator — deliberately imperfect, and better in some regions than others.",
    sections=[("Why simulate", [
        "Lecture 7 makes this a general principle: test on simulated users first, because "
        "they give full control over behaviour and make results interpretable, and real "
        "users are expensive. Here it also makes the experiments repeatable."]),
      ("What makes an expert useful for this project", [
        "Competence must vary with the input. An expert who is right 70% of the time "
        "uniformly at random gives a deferral problem with a constant optimal answer — "
        "always defer, or never — and nothing to learn.",
        "So the expert is right in identifiable regions and wrong elsewhere, mimicking a "
        "person with a specialism."]),
      ("The subtle constraint", [
        "Competence must depend on the input x, not on the true label y. An expert defined "
        "as “95% accurate whenever the article is about sports” cannot be exploited at "
        "prediction time, because the topic is precisely what is unknown then.",
        "Making competence a function of the text — its vocabulary, its length — keeps the "
        "problem well posed, and is what allows the active learning in the third stage to "
        "discover anything at all."])],
    advice="Check that competence really varies by region before trusting any downstream result.",
    related=["region", "competence", "deferral"])

_add(
    slug="region", title="Regions of the input space", group="The idea",
    short="Groups of similar articles, used to describe where the expert is strong or weak.",
    sections=[("What a region is", [
        "A subset of inputs that resemble one another — here, articles sharing vocabulary. "
        "They are found by clustering the vectorised text, so a region is defined by the "
        "words it contains and can be identified without knowing the topic."]),
      ("Why they are useful", [
        "They make competence describable. “The expert is reliable on articles about "
        "markets and earnings” is a statement about a region, and it can be checked, "
        "reported and acted on.",
        "They also give the active learning something to generalise over: knowing the "
        "expert did well on one article says something about nearby articles only if "
        "nearby means something."]),
      ("Their limit", [
        "A region-level rule is coarse. If competence varies within a region, no rule that "
        "only knows which region an article is in can capture it — the residual has to be "
        "handled per item, and measuring that gap is informative in itself."])],
    advice="Compare a region-level deferral rule against a per-item one; the difference tells you how much structure the regions actually capture.",
    related=["expert", "competence", "tfidf"])

_add(
    slug="competence", title="Competence model", group="How it works",
    short="A second model that predicts whether the expert will be right, from the article alone.",
    sections=[("What it estimates", [
        "For each article, the probability that the expert answers it correctly:",
        Math("q(x) = P( m(x) = y | x )"),
        Aside("m(x) is the expert's prediction and y the true label, so the event inside is “the expert got this one right”."),
        "Notice what sits after the conditioning bar. The probability is a function of $x$ "
        "alone, and it has to be: at the moment the deferral decision is taken, the article "
        "is all anyone has. The true label $y$ is precisely what is unknown.",
        "It is fitted as an ordinary binary classification problem. The inputs are the "
        "article\u2019s features; the target is a 0 or 1 recording whether the expert was "
        "right on it. Any classifier will do, and its predicted probability is $q(x)$."]),
      ("Where the labels come from", [
        "This is where the missing “should we defer” label gets manufactured. You cannot "
        "observe whether deferral was the correct decision, but for any article you have "
        "actually queried you can observe whether the expert was right — and that binary "
        "outcome is a legitimate training target."]),
      ("Why it must be honest", [
        "The same care is needed as for the classifier: correctness has to be measured on "
        "predictions the model did not fit, or the competence model learns that the expert "
        "is better than they are."])],
    advice="A competence model that predicts a near-constant is telling you the expert has no structure worth exploiting.",
    related=["expert", "advantage", "out-of-fold"])

_add(
    slug="advantage", title="The advantage score", group="How it works",
    short="How much more likely the expert is to be right than the classifier, for this article.",
    sections=[("The quantity", [
        Math("a(x) = q(x) − p(x)"),
        Aside("q(x) is the expert's estimated chance of being right; p(x) the classifier's."),
        "Positive means the expert is the better bet on this article; negative means the "
        "machine is. Deferring is worthwhile when a(x) is large enough to justify the "
        "interruption."]),
      ("Why the difference and not either term", [
        "Rejection asks whether the machine is unsure. Deferral asks whether somebody else "
        "is better. A hard article that the expert also gets wrong has low p and low q, so "
        "a small advantage — deferring it costs attention and buys nothing.",
        "Conversely an article the classifier is confidently wrong about has high p and yet "
        "a large true advantage. Confidence-based rules never catch those, which is the "
        "practical difference between the two framings."]),
      ("The connection to the Bayes rule", [
        "Lecture 5's Bayes-optimal deferral rule compares the machine's expected error "
        "with the expert's expected cost: defer when",
        Math("1 − max_y P(y|x)   ≥   E_{y|x}[ c_u(x, y) ]"),
        "The advantage score is a direct estimate of the gap between those two sides."])],
    advice="Sort the test set by advantage and read both ends; it is the quickest way to see what the policy has learned.",
    related=["competence", "threshold", "deferral"])

_add(
    slug="threshold", title="The threshold τ", group="How it works",
    short="How large the expert's advantage must be before the case is actually handed over.",
    sections=[("What it controls", [
        "The rule is to defer when a(x) > τ. At τ = 0 the system defers whenever the "
        "expert is even marginally better. Raising τ demands the expert be substantially "
        "better, so fewer articles are handed over.",
        "τ therefore absorbs the query cost c(x) from lecture 5: it is the price of the "
        "expert's attention, expressed in units of accuracy."]),
      ("Why it is exposed to the user", [
        "There is no correct value. It encodes how much a human's time is worth relative "
        "to an error, which is a fact about the deployment and not about the data. A "
        "newsroom with one sub-editor and a newsroom with ten want different τ."]),
      ("Reading the sweep", [
        "Sweeping τ traces the accuracy–coverage curve. That curve, rather than any single "
        "operating point, is what a decision-maker needs: it answers what team accuracy "
        "you get for 10% of an expert's time."])],
    advice="Quote τ together with the deferral rate it produces; on its own the number means little.",
    related=["advantage", "deferral-rate", "team-accuracy"])

_add(
    slug="outcomes", title="The four outcomes", group="Measuring",
    short="Every deferred article falls into one of four cases, and only one of them is a win.",
    sections=[("The cases", [
        "For a deferred article, the machine would have been right or wrong, and the "
        "expert is right or wrong. That is a 2×2 table:",
        "Expert right, machine would have been wrong — a genuine win, the reason deferral "
        "exists. Expert right, machine also right — wasted, the answer was already "
        "available. Expert wrong, machine would have been right — actively harmful, the "
        "handover destroyed a correct answer. Both wrong — no loss, no gain."]),
      ("Why the breakdown matters", [
        "Team accuracy sums these into one number and hides which is happening. Two "
        "policies with identical accuracy can have completely different profiles, and the "
        "one wasting less of the expert's time is the better system."])],
    advice="Report the four counts, not just the total; the harmful cell is the one worth minimising.",
    related=["team-accuracy", "deferral-precision", "oracle"])

_add(
    slug="deferral-precision", title="Deferral precision and recall", group="Measuring",
    short="Of the cases handed over, how many needed it; and of the cases that needed it, how many were caught.",
    sections=[("The definitions", [
        "Call an article one that “needed deferral” if the expert is right on it and the "
        "machine wrong. Then",
        Math("precision = (needed ∧ deferred) ⁄ deferred"),
        Math("recall = (needed ∧ deferred) ⁄ needed"),
        "Both share a numerator — the cases handed over that genuinely warranted it — and "
        "differ only in what they divide by.",
        "Precision divides by everything deferred, so it answers: of the interruptions you "
        "caused, what share were worth causing? That is the expert\u2019s perspective, and "
        "it measures wasted attention.",
        "Recall divides by everything that needed deferring, so it answers: of the cases "
        "where handing over would have helped, how many did you catch? That is the "
        "system\u2019s perspective, and it measures missed benefit."]),
      ("The trade-off", [
        "Lowering the threshold raises recall and lowers precision — you catch more of the "
        "useful cases by handing over more of everything. A policy that defers the entire "
        "queue has recall 1 and precision equal to the base rate of useful deferrals, which "
        "is usually low; a policy that defers one case it is certain about has precision 1 "
        "and recall near zero.",
        "Neither extreme is any good, which is exactly why a single number cannot describe "
        "a deferral policy. The harmonic mean of the two is the F1 score if one summary is "
        "wanted."])],
    advice="High recall with poor precision means the policy is deferring a lot rather than deferring well.",
    related=["outcomes", "threshold", "oracle"])

_add(
    slug="team-accuracy", title="Team accuracy", group="Measuring",
    short="How often the combined system is right, counting the classifier's answers and the expert's together.",
    sections=[("Definition", [
        Math("acc_team = (1/n) Σᵢ 1[ ŷᵢ = yᵢ ]"),
        Aside("ŷᵢ is whichever of the two answered article i."),
        "It is the headline number, and it is the one to be most careful with."]),
      ("Why it is not enough", [
        "If the expert is better overall than the classifier, the policy “always defer” "
        "scores well. It has made no decisions, consumed an entire person's attention, and "
        "would still look respectable in a table.",
        "The brief is explicit that evaluation must reflect the quality of the deferral "
        "decisions and not only classification accuracy."])],
    advice="Never quote team accuracy without the deferral rate beside it.",
    related=["deferral-rate", "outcomes", "oracle"])

_add(
    slug="deferral-rate", title="Deferral rate", group="Measuring",
    short="The fraction of articles handed to the expert — the price of the accuracy gain.",
    sections=[("What it measures", [
        "Coverage of the human. A system deferring 5% of a queue is asking for a few "
        "minutes; one deferring 60% is asking for most of a job.",
        "It converts directly into cost, which is what makes it the natural x-axis when "
        "reporting results."]),
      ("Reading it with accuracy", [
        "Plotting team accuracy against deferral rate as τ sweeps gives the curve that "
        "actually answers the deployment question. A steep initial rise means the policy "
        "finds the valuable cases first, which is exactly what a good policy should do."])],
    advice="Compare policies at equal deferral rate; comparing them at their own preferred rates compares nothing.",
    related=["threshold", "team-accuracy", "oracle"])

_add(
    slug="oracle", title="The oracle ceiling", group="Measuring",
    short="The best any deferral policy could possibly do, computed by cheating.",
    sections=[("How it is computed", [
        "Defer exactly when the expert is right and the machine wrong — using the true "
        "labels, which no real policy has. That is the maximum achievable team accuracy at "
        "the minimum possible deferral rate."]),
      ("Why it is the number to report against", [
        "An improvement of two points over the classifier alone means one thing if the "
        "ceiling is three points away and something quite different if it is twenty. The "
        "oracle converts an unanchored gain into a fraction of what was available.",
        "It also bounds the problem honestly. If the oracle headroom is small, no policy "
        "can do much, and saying so is more useful than reporting a small win as a success."])],
    advice="Report your policy's gain as a percentage of the oracle headroom; it is the only scale-free way to judge it.",
    related=["team-accuracy", "outcomes", "deferral-precision"])

_add(
    slug="active-learning", title="Active learning", group="Asking questions",
    short="Choosing which examples to get labelled, instead of labelling everything.",
    sections=[("The motivation", [
        "Lecture 6 opens with speech recognition for low-resource languages: audio is easy "
        "to obtain, transcriptions are the bottleneck. The same asymmetry holds here — "
        "articles are free, expert opinions are not."]),
      ("Why choosing helps so much", [
        "For the 0–1 loss and a finite hypothesis class, passive learning obeys",
        Math("P( |L(h) − L_emp(h)| > ε )  ≤  2|ℋ| e^(−2Nε²)"),
        "Lecture 6 works the numbers: for |ℋ| = 100, ε = 10⁻³ and 95% confidence, that "
        "demands almost three million labelled points.",
        "That is the cost of labelling blindly. Active learning is the claim that most "
        "points teach you nothing, so choosing well does far better than the bound "
        "suggests."]),
      ("The setting in this project", [
        "The classifier can be trained on the full labelled dataset. The expert's opinions "
        "are what must be bought, a few at a time, and the goal is to learn where "
        "deferring pays."])],
    advice="Always include a random-query baseline; lecture 6 names uniform random as a legitimate strategy, and it is a hard one to beat.",
    related=["acquisition", "query-budget", "competence"])

_add(
    slug="acquisition", title="How the next question is chosen", group="Asking questions",
    short="A utility function scores every unlabelled article by how much its answer would teach.",
    sections=[("The framework", [
        "A query strategy is a rule for picking the next instance to label, and it is "
        "defined by a utility u(x) encoding informativeness. Lecture 6 divides strategies "
        "into information-based ones, which target uncertainty, and representation-based "
        "ones, which target coverage of the input space."]),
      ("The classical uncertainty measures", [
        "For a model producing class probabilities q(y|x), lecture 6 gives three:",
        Math("u_LC(x) = 1 − max_y q(y|x)"),
        Math("u_margin(x) = 1 − ( q(ŷ₁|x) − q(ŷ₂|x) )"),
        Math("u_Ent(x) = − Σₖ q(k|x) log q(k|x)"),
        "Least confidence uses only the top class and discards the rest of the "
        "distribution. Margin compares the top two, so it captures genuine ambiguity "
        "between competing labels. Entropy uses the whole distribution."]),
      ("The trap specific to this project", [
        "Those measure uncertainty about the label. The goal here is to learn when "
        "deferral is beneficial, which is a property of the difference between two models, "
        "not of either alone.",
        "The classifier's hardest articles are not the ones where deferring helps most: an "
        "article the classifier agonises over and the expert also fails is worthless to "
        "query. The informative queries are those where the estimated advantage a(x) is "
        "closest to the threshold — near the deferral boundary, not the classification "
        "one."])],
    advice="Plot each strategy against random on the same axes; a strategy that does not beat random is a result worth reporting.",
    related=["active-learning", "advantage", "confidence"])

_add(
    slug="query-budget", title="Query budget", group="Asking questions",
    short="The number of expert opinions you are allowed to buy.",
    sections=[("Why it is fixed", [
        "Expert time is the scarce resource, so results are reported as a function of how "
        "much of it was spent. The natural presentation is a curve: queries used on the "
        "x-axis, team accuracy on the y-axis."]),
      ("The cold start", [
        "At the first query there are no expert labels, so every utility-based strategy is "
        "undefined. Something has to seed the process — a random or stratified first round "
        "is normal — and the choice should be deliberate rather than accidental, since it "
        "affects every strategy equally and can dominate short budgets."]),
      ("Reading the curves", [
        "These curves are noisy: a single run can suggest anything. Averaging over several "
        "random seeds and showing the spread is the difference between a result and an "
        "anecdote."])],
    advice="Compare strategies at equal budget, and show the variation across seeds rather than a single run.",
    related=["active-learning", "acquisition"])

_add(
    slug="out-of-fold", title="Out-of-fold scoring", group="Under the bonnet",
    short="Judging a model only on predictions made by a version of it that never saw that row.",
    sections=[("The problem it solves", [
        "A model asked about its own training data is far more accurate than it will ever "
        "be on new data — it has already seen those labels. Using that inflated view to "
        "decide when the classifier needs help teaches the system that it almost never "
        "does."]),
      ("The procedure", [
        "Split the training data into k folds. For each fold, train on the other k−1 and "
        "predict the held-out one. Every prediction then comes from a model that never saw "
        "that row, and every row still receives a prediction."]),
      ("The size of the effect", [
        "It is not a small correction. On this dataset the classifier makes roughly 3,400 "
        "errors in-sample against 11,350 out-of-fold — the in-sample view understates "
        "failure by more than a factor of three."])],
    advice="Every quantity that feeds the deferral decision must be computed out-of-fold, not just the headline accuracy.",
    related=["classifier", "competence", "deferral"])

_add(
    slug="classifier", title="The classifier", group="Under the bonnet",
    short="The model that answers whenever the system does not defer.",
    sections=[("What it is", [
        "A linear model over TF-IDF features of the article text, trained on the full "
        "labelled training set. AG News has four balanced topics — world, sports, business "
        "and science/technology — so chance is 25%."]),
      ("Why something simple", [
        "It is retrained many times across the deferral and active learning experiments, so "
        "training cost multiplies. A sparse linear model is fast, strong on text, and "
        "produces calibrated-enough probabilities to be usable as confidence.",
        "It also sets the baseline the human–AI team must beat, which is the point of "
        "task 1: a team that does not beat the machine alone is not worth assembling."])],
    advice="Its accuracy is the number every later result should be compared against.",
    related=["tfidf", "confidence", "out-of-fold"])

_add(
    slug="confidence", title="Confidence", group="Under the bonnet",
    short="How sure the classifier is — and why that alone is the wrong basis for deferring.",
    sections=[("What it means", [
        "The probability assigned to the predicted class, max_y q(y|x). High confidence "
        "means the model considers the alternatives unlikely."]),
      ("Confidence-based rejection", [
        "Lecture 5 derives the optimal rejection rule for a known cost c: abstain when the "
        "chance of being wrong exceeds the cost of abstaining. With calibrated "
        "probabilities that reduces to a threshold on confidence — Chow's rule."]),
      ("Why it is not enough for deferral", [
        "Rejection with a fixed cost only needs to know whether the machine is likely to be "
        "wrong. Deferral needs to know whether somebody else is likely to be right, which "
        "confidence says nothing about.",
        "Worse, confidence is systematically wrong where it matters most. Models are often "
        "confidently wrong, and those cases — where deferring pays most — look safest to a "
        "confidence rule."])],
    advice="Use confidence as one input to the advantage score, never as the deferral rule itself.",
    related=["advantage", "acquisition", "classifier"])

_add(
    slug="tfidf", title="Turning text into numbers", group="Under the bonnet",
    short="Each article becomes a vector of weighted word counts.",
    sections=[("Term frequency, inverse document frequency", [
        "A word's weight rises with how often it appears in this article and falls with how "
        "many articles contain it at all:",
        Math("tfidf(t, d) = tf(t, d) · log( N ⁄ df(t) )"),
        Aside("tf is the count of term t in document d; df(t) the number of documents containing t; N the total."),
        "The second factor is the point. “The” appears everywhere and carries no "
        "information about topic, so log(N/df) is near zero for it. “Quarterly” appears in "
        "few articles and is highly indicative, so its weight stays large."]),
      ("What it throws away", [
        "Word order and syntax entirely — the representation is a bag of words. “Dog bites "
        "man” and “man bites dog” are identical vectors. For topic classification that "
        "loss is largely harmless, which is why the method survives.",
        "The result is very high-dimensional and extremely sparse: tens of thousands of "
        "columns, of which each article uses a few dozen."])],
    advice="The same vectorisation defines the regions, so its vocabulary determines what “similar article” can mean.",
    related=["classifier", "region"])


def groups():
    ordered = {}
    for topic in TOPICS.values():
        ordered.setdefault(topic.group, []).append(topic)
    return ordered
