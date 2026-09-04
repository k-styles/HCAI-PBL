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
     short="2,734 well-known films, described only by their metadata.",
     sections=[("What is in it", [
         "The IMDB 5000 dataset: about five thousand films with their genres, year, "
         "running time, certificate, average score and how many people voted on it. "
         "There are no ratings by individual people in it at all.",
         "That absence is the reason this project exists. A normal recommender learns "
         "from millions of past ratings. Here there is nothing to learn from until "
         "somebody sits down and tells us what they like."]),
       ("Why it was trimmed", [
         "Films with fewer than 25,000 votes were dropped. Someone comparing two films "
         "they have never heard of is not expressing a preference, they are guessing "
         "from the card, and that guessing is noise we can never model away.",
         "2,734 films survive, from 1927 to 2016."])],
     advice="Every film you are shown comes from this pool, drawn at random.",
     related=["cold-start", "feature-vector"])

_add(slug="cold-start", title="The cold start", group="The data",
     short="A brand new user with no history at all — and the whole problem here.",
     sections=[("The situation", [
         "You have just signed up. The system knows nothing about you. It cannot say "
         "\"people like you enjoyed this\", because it has no idea who is like you.",
         "The only way out is to ask. But you will not sit through two hundred questions, "
         "so the system gets maybe twenty. Which twenty, and asked how?"]),
       ("What this project measures", [
         "Two ways of asking. Pick one of two films, over and over; or put ten films in "
         "order. Both cost you time. The question is which one tells the system more "
         "per minute you spend."])],
     advice="This is the question the whole user study is built to answer.",
     related=["elicitation", "movies", "designs"])

# --- the model -------------------------------------------------------------

_add(slug="feature-vector", title="Feature vector, x", group="The model",
     short="A film boiled down to 20 numbers.",
     sections=[("What it is", [
         "Every film becomes a list of numbers: is it a comedy (1 or 0), is it a thriller, "
         "how recent, how long, how well reviewed, how widely seen, is it for adults, is "
         "it black and white. Twenty numbers in all.",
         "The model only ever sees these numbers. If something about a film is not in "
         "them, the model is blind to it."]),
       ("Why only twenty", [
         "Because every number needs evidence. Twenty numbers can be pinned down by "
         "twenty-odd answers. Two thousand — one per director, say — could not be pinned "
         "down by any number of answers you would tolerate giving.",
         "There is a second rule: if you cannot see it on the card, it is not in x. You "
         "cannot have chosen a film for its budget if we never showed you the budget."])],
     advice="The features page lists all twenty and what was deliberately left out.",
     related=["preference-vector", "utility", "identifiability", "standardise"])

_add(slug="preference-vector", title="Preference vector, w", group="The model",
     short="Your taste, as one number per feature — the thing we are trying to find.",
     sections=[("What it is", [
         "For each of the twenty features, one number saying how much that feature "
         "appeals to you. Positive means you like it, negative means you avoid it, "
         "large means you care a lot.",
         "\"Loves horror, hates long films, mildly prefers older ones\" is a w."]),
       ("Why it is hard", [
         "Nobody can tell you their own w. Ask someone their weight on \"how widely seen\" "
         "and you will get a shrug. But ask them to choose between two films twenty times "
         "and the number falls out.",
         "It is also never observed, which is exactly why the study cannot measure "
         "\"how close did we get\" and has to measure something else instead."])],
     advice="At the end of the study you are shown the w that was fitted to your answers.",
     related=["utility", "feature-vector", "holdout", "map"])

_add(slug="utility", title="Utility, U(x) = wᵀx", group="The model",
     short="One score per film, for one person: multiply and add.",
     sections=[("How it works", [
         "Take each of the film's twenty numbers, multiply by your weight for that "
         "feature, add it all up. One score. Higher means you should like it more.",
         "That is the entire preference model. Everything else is about working out the "
         "weights."]),
       ("What it assumes", [
         "That features add up independently — that liking horror and liking short films "
         "means liking short horror films exactly that much and no more. Real taste has "
         "interactions this cannot express. It is a deliberate simplification, and it is "
         "what makes twenty answers enough."])],
     advice="Two films with the same score should feel like a coin toss to you.",
     related=["preference-vector", "bradley-terry", "feature-vector"])

_add(slug="bradley-terry", title="The Bradley–Terry model", group="The model",
     short="Turns two utility scores into the chance of picking one over the other.",
     sections=[("The idea", [
         "If two films score the same, you pick either with probability one half. As one "
         "pulls ahead, the chance of picking it rises — smoothly, never quite reaching "
         "certainty.",
         "Concretely the probability is one divided by one plus e to the minus the score "
         "difference. A one-point gap gives about 73 to 27; a three-point gap about 95 to 5."]),
       ("Why not just say the higher score always wins", [
         "Because people are not that consistent, and a model that says \"always\" cannot "
         "learn from being wrong. Allowing for the occasional surprising answer is what "
         "lets a handful of responses say something useful."])],
     advice="This is the model the brief gives us. Task 2 is about stretching it.",
     related=["plackett-luce", "utility", "iia"])

_add(slug="plackett-luce", title="The Plackett–Luce model", group="The model",
     short="Bradley–Terry for a whole ranking instead of a single pair.",
     sections=[("How the extension works", [
         "Read your ranking of ten as a sequence of choices. First you picked a favourite "
         "out of ten. Then a favourite out of the nine left. Then out of eight. Nine "
         "choices in all.",
         "Each of those is Bradley–Terry with more than two options, so multiply the nine "
         "probabilities together. That product is Plackett–Luce."]),
       ("Why this one and not another", [
         "With only two films the product has one factor and it is Bradley–Terry exactly, "
         "so nothing was replaced.",
         "The chance that one film ends up above another in the ranking works out to be "
         "the plain Bradley–Terry probability, whatever the other eight films are. So the "
         "pairwise interface and the ranking interface are estimating the same thing — "
         "without which comparing them would be meaningless.",
         "The tempting shortcut is to chop a ranking of ten into all forty-five pairs and "
         "treat them as forty-five separate answers. That counts the same evidence over "
         "and over: if you put A above B and B above C, then A above C came free. It also "
         "does not add up to one across the possible orderings, so it is not a probability "
         "at all."])],
     advice="This derivation is Task 2, and it is set out in full in the PDF report.",
     related=["bradley-terry", "ranking", "iia", "fisher"])

_add(slug="iia", title="Luce's choice axiom", group="The model",
     short="The assumption that adding options doesn't change how you rank the others.",
     sections=[("What it says", [
         "If you prefer A to B, then throwing C into the mix should not flip that. The "
         "odds between A and B are the odds between A and B, whoever else is present."]),
       ("Where it breaks", [
         "Put two near-identical sequels in a set of ten and people often push both down "
         "— the pair split their appeal. Bradley–Terry and Plackett–Luce cannot represent "
         "that, because it is exactly what the axiom forbids.",
         "This is a genuine limitation, and the study design treats it as one: the "
         "ranking interface exposes ten films at a time and so is more vulnerable to it "
         "than the pairwise one. The report lists it as a threat to validity."])],
     advice="If a set of ten ever felt like it contained duplicates, this is why that matters.",
     related=["plackett-luce", "bradley-terry", "validity"])

# --- fitting ---------------------------------------------------------------

_add(slug="map", title="How w is fitted", group="Fitting",
     short="Pick the w that makes your actual answers most likely — with a nudge toward zero.",
     sections=[("The search", [
         "Every candidate w assigns a probability to the answers you gave. Some w make "
         "your answers look likely, others make them look like a fluke. Take the one that "
         "makes them likeliest.",
         "It is found by walking uphill: start at all zeros, work out which direction "
         "improves things, step, repeat until stepping stops helping."]),
       ("Why the nudge", [
         "With twenty numbers and twenty-five answers, there are usually directions the "
         "answers say nothing about, and left alone the search runs off to infinity along "
         "them. A gentle pull toward zero stops that, and it says something sensible: in "
         "the absence of evidence, assume no strong opinion."])],
     advice="More answers means the pull toward zero matters less.",
     related=["prior", "preference-vector", "identifiability"])

_add(slug="prior", title="The prior", group="Fitting",
     short="A starting assumption of no strong opinions, overruled by evidence.",
     sections=[("What it is here", [
         "Before you answer anything, we assume your weights are smallish and centred on "
         "zero — no violent opinions in any direction. Each answer you give pushes back "
         "against that.",
         "The strength is set so a one standard deviation change in a feature is worth "
         "about one unit of utility, which is roughly a 73/27 preference. Weak, on purpose."]),
       ("Why it is necessary rather than optional", [
         "Without it the best-fitting w often does not exist — there is always a more "
         "extreme w that fits slightly better. That is not a computational problem to "
         "engineer around, it is the honest answer that twenty-five responses do not pin "
         "down twenty numbers."])],
     advice="It is why your fitted profile looks moderate even if your answers were decisive.",
     related=["map", "identifiability"])

_add(slug="identifiability", title="Identifiability", group="Fitting",
     short="Whether the answers could ever tell you a particular weight, even in principle.",
     sections=[("The problem", [
         "Suppose you added one weight per director — 2,398 of them. Two films drawn at "
         "random almost never share a director, so essentially every answer you give "
         "multiplies those weights by zero. No number of answers would ever say anything "
         "about them.",
         "They are not weak features. They are unmeasurable ones at this budget, and "
         "including them just lets the prior make things up."]),
       ("How the feature set was chosen", [
         "A genre earns a place only if random pairs of films actually differ on it. For "
         "a genre appearing in a fraction p of films, that happens about 2p(1−p) of the "
         "time; requiring one informative pair in ten leaves fourteen genres and drops "
         "Western, Musical, Documentary and the rest."])],
     advice="The features page shows the cut and everything that fell below it.",
     related=["feature-vector", "prior", "map"])

_add(slug="standardise", title="Standardising", group="Fitting",
     short="Putting features on a common scale so their weights are comparable.",
     sections=[("What was done", [
         "Year, running time, score and vote count are measured in wildly different units. "
         "Each is rewritten as \"how many standard deviations from the average film\", so a "
         "weight of 0.5 means the same amount of preference whichever one it sits on."]),
       ("Why genres were left alone", [
         "Genre flags stay as plain 0 or 1. Scaling a rare genre to unit variance would "
         "let the same weight produce a much bigger swing in utility than for a common "
         "one, which is backwards — rare genres have less evidence behind them, not more."])],
     advice="It is also why the profile bars at the end can be compared to each other.",
     related=["feature-vector", "prior"])

_add(slug="fisher", title="How much a task is worth", group="Fitting",
     short="A way to measure how much one question narrows down w.",
     sections=[("The idea", [
         "Some questions are informative and some are not. Comparing two near-identical "
         "romances tells you almost nothing; comparing a 1950s black-and-white drama with "
         "a modern animated comedy tells you a lot.",
         "Fisher information puts a number on that: how sharply the answer's probability "
         "changes as w changes. Big number, informative question."]),
       ("What it says about the two designs", [
         "A ranking of ten contributes nine of these terms, so one ranking is worth "
         "around sixty pairwise comparisons. That sounds decisive, and it is misleading. "
         "Eight minutes buys eighty comparisons or five rankings, and counted that way "
         "the totals come out almost equal.",
         "Then a second thing goes wrong for ranking. Total information is not the same "
         "as useful information. Eighty comparisons touch 160 different films and pin "
         "down taste in every direction; five rankings touch fifty, and everything one "
         "ranking teaches you is about those ten films. Whole directions of taste go "
         "unmeasured, and the prior quietly fills them in.",
         "So the arithmetic that looked like it favoured ranking ends up favouring "
         "pairs. Whether that survives contact with real people is what the study is "
         "for — all of it assumes people answer exactly as the model says."])],
     advice="The report works the numbers through.",
     related=["plackett-luce", "designs", "confound"])

# --- the study -------------------------------------------------------------

_add(slug="elicitation", title="Preference elicitation", group="The study",
     short="Asking someone a few well-chosen questions to work out what they like.",
     sections=[("What it means", [
         "Not \"guessing from what you watched\" — there is nothing to guess from yet. "
         "Directly asking, in a form people can actually answer.",
         "People are bad at saying \"I weight thrillers at 0.6\" and good at saying \"that "
         "one, not that one\". Elicitation is the business of turning the second into the "
         "first."])],
     advice="Both designs in this study are elicitation methods. That is what is being compared.",
     related=["cold-start", "designs", "bradley-terry"])

_add(slug="designs", title="The two designs", group="The study",
     short="Design 1: pick one of two. Design 2: put ten in order.",
     sections=[("Design 1 — pairwise choice", [
         "Two films, choose one, repeat. Each answer is tiny but it is easy, fast and "
         "hard to get wrong."]),
       ("Design 2 — rank ten", [
         "Ten films, drag them into order. One answer carries far more information, but "
         "it takes much longer, and ordering the middle of a list of ten is genuinely "
         "difficult."]),
       ("Why it is not obvious which wins", [
         "Ranking wins on information per answer by a wide margin. Pairwise wins on "
         "answers per minute and on how carefully each one is given. Which effect is "
         "bigger is an empirical question about people, not a fact about the maths."])],
     advice="You were assigned one of these at random when you started.",
     related=["fisher", "confound", "between-subjects"])

_add(slug="holdout", title="The held-out block", group="The study",
     short="Ten fresh comparisons at the end, used to score how well we learned you.",
     sections=[("Why it exists", [
         "We want to know which design learned your taste better. But your true taste is "
         "never observed, so \"how close is the estimate\" cannot be measured.",
         "What can be measured is prediction. Fit w on the elicitation answers only, then "
         "see how well it calls ten comparisons it never saw. A w that has genuinely "
         "learned something gets these right."]),
       ("Why it is identical in both arms", [
         "Same task, same number, same instructions, whichever design you were given. If "
         "the test itself differed between the arms, any difference in the result could be "
         "the test rather than the design."])],
     advice="Two of the held-out pairs are repeats of earlier ones, shown with the sides swapped.",
     related=["log-loss", "preference-vector", "attention-check"])

_add(slug="log-loss", title="Log loss", group="The study",
     short="Scores a prediction on how confident it was, not just whether it was right.",
     sections=[("What it measures", [
         "Predicting the right film with 90% confidence scores better than predicting it "
         "with 55%. Predicting the wrong one with 90% confidence scores much worse than "
         "getting it wrong hesitantly.",
         "A model that knows nothing says 50/50 every time and scores 0.693. Anything "
         "below that has learned something; above it, worse than a coin."]),
       ("Why not just count how many were right", [
         "Because with only ten held-out comparisons, plain accuracy takes eleven possible "
         "values and throws away all the confidence information. Log loss uses the whole "
         "prediction, so it can tell two designs apart on far fewer participants."])],
     advice="0.693 is the line to beat, and it is marked on your debrief.",
     related=["holdout", "power"])

_add(slug="between-subjects", title="Between subjects", group="The study",
     short="Each participant does one design, not both.",
     sections=[("Why not both", [
         "It would be more sensitive — each person acting as their own comparison — but "
         "by the time you have ranked ten films you understand your own taste better, and "
         "you would do the second design differently for that reason alone. There is no "
         "ordering that removes it.",
         "So each participant gets one design, chosen at random, and the arms are "
         "compared across people."]),
       ("The cost", [
         "It needs more participants, because differences between people now sit on top of "
         "the difference you are looking for. That is what the power calculation is for."])],
     advice="You were assigned at random. The report explains the blocking used in the real study.",
     related=["designs", "power", "confound"])

_add(slug="confound", title="The time confound", group="The study",
     short="Comparing 40 pairs to 8 rankings compares two arbitrary numbers.",
     sections=[("The trap", [
         "Give one arm forty comparisons and the other eight rankings and whichever wins, "
         "the answer is about those two numbers, not about the designs.",
         "What a recommender is actually spending is the user's patience. Eight minutes is "
         "eight minutes whichever way it is spent."]),
       ("What the study does", [
         "Both arms get the same wall-clock elicitation budget and do as many tasks as fit. "
         "The comparison is then about information per minute, which is the quantity "
         "anyone building this would care about."])],
     advice="The timer on the study page is that budget running down.",
     related=["designs", "fisher", "validity"])

_add(slug="power", title="How many participants", group="The study",
     short="Working out the sample size before running anything, not after.",
     sections=[("The calculation", [
         "For a moderate difference between the two arms, a two-sided test at the "
         "conventional 5% level with an 80% chance of detecting it needs about 64 people "
         "per arm — 128, plus a margin for exclusions, so 150 recruited.",
         "Doing this first is the point. Deciding afterwards how many people you needed "
         "is how you end up finding whatever you were hoping for."]),
       ("Pre-registration", [
         "The hypothesis, the primary outcome, the exclusions and the test are all written "
         "down before the first participant. Otherwise there are a dozen defensible "
         "analyses and the one that gets reported is the one that worked."])],
     advice="Section 4 of the PDF has the full calculation.",
     related=["log-loss", "between-subjects", "validity"])

_add(slug="attention-check", title="Attention checks", group="The study",
     short="Repeated questions that reveal who was clicking at random.",
     sections=[("How they work here", [
         "Two of the held-out pairs come round a second time with the films swapped left "
         "to right. Someone actually deciding will mostly answer the same way twice; "
         "someone clicking through will not.",
         "Disagreeing on both is the pre-registered reason to exclude a participant — "
         "decided in advance, so it cannot be applied selectively to whoever spoils the "
         "result."])],
     advice="Your own consistency is reported on the debrief page.",
     related=["holdout", "power", "validity"])

_add(slug="validity", title="Threats to validity", group="The study",
     short="The ways this study could give the right answer to the wrong question.",
     sections=[("The ones that matter here", [
         "Films nobody recognises turn preference into guessing — handled by the vote "
         "threshold on the pool.",
         "Ranking ten items invites carelessness at the bottom of the list, so the fitted "
         "w may be learning noise from positions eight to ten. The analysis models "
         "position-dependent noise rather than assuming it away.",
         "Sets of ten can contain near-duplicates, which the model is formally unable to "
         "handle. It is recorded as a limitation, not fixed.",
         "Prolific participants are not a random sample of humanity. The finding is about "
         "the designs on that population, and the report says so."])],
     advice="A study that lists none of these has not looked.",
     related=["iia", "attention-check", "power"])

_add(slug="consent", title="Informed consent", group="The study",
     short="You are told what will happen and what is recorded, before anything starts.",
     sections=[("What is recorded", [
         "Which design you were given, which films you chose or how you ordered them, how "
         "long each task took, and one coarse answer about how often you watch films. "
         "Nothing that identifies you.",
         "No name, no email, no IP address, no cookie beyond the one that keeps this "
         "session going."]),
       ("What you can do", [
         "Stop at any point with the button on the study page, and the run ends there. In "
         "the real study, withdrawing means the data is deleted and the payment is still "
         "made."])],
     advice="Consent that requires understanding the maths is not consent, which is why these pages exist.",
     related=["movies", "designs"])

_add(slug="ranking", title="Ranking ten films", group="The study",
     short="Putting a list in order from most to least preferred.",
     sections=[("How to do it here", [
         "Drag a film to move it, or use the up and down buttons on each row. The buttons "
         "are not a fallback for the clumsy: dragging is unusable with a keyboard or a "
         "screen reader, and an interface that only works one way excludes participants.",
         "Position one is your favourite, position ten your least favourite."]),
       ("Doing it honestly", [
         "The top and bottom are usually easy and the middle is genuinely hard. Getting "
         "the middle roughly right is fine — the model expects imprecision and this study "
         "is partly about measuring how much of it there is."])],
     advice="You can reorder as much as you like before submitting.",
     related=["plackett-luce", "designs", "iia"])
