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


TOPICS = {}


def _add(**kw):
    topic = Topic(**kw)
    TOPICS[topic.slug] = topic


_add(
    slug="deferral", title="Deferring", group="The idea",
    short="Letting the computer hand a decision to a person instead of answering it itself.",
    sections=[
        ("The everyday version", [
            "A call centre robot that answers what it can and puts you through to a human "
            "when it cannot. That handover is a deferral.",
            "Here the job is sorting news articles into four topics. For each article the "
            "system either answers, or passes it to a human expert.",
        ]),
        ("Why it is not obvious", [
            "The system has to decide who is more likely to be right on this particular "
            "article, before knowing the answer. Handing over too much wastes a person's "
            "time and can make things worse; handing over too little wastes the person.",
        ]),
    ],
    advice="Everything on these pages is about learning when to hand over.",
    related=["expert", "team-accuracy", "threshold"],
)

_add(
    slug="expert", title="The simulated expert", group="The idea",
    short="A stand-in for a human specialist, who is good at some kinds of article and poor at others.",
    sections=[
        ("Why simulated", [
            "Training this kind of system needs to know what a human expert would have "
            "said about thousands of articles. Collecting that for real is expensive, so "
            "the expert here is a program that behaves like one.",
        ]),
        ("What makes them realistic", [
            "They are not perfect and they are not uniformly mediocre. Like a real "
            "journalist, they have a beat: reliable on their own subjects, weak outside "
            "them.",
            "Crucially their competence depends on what the article looks like, never on "
            "the right answer. An expert who was reliable 'whenever the answer is Sports' "
            "could never be found out by asking questions, because the answer is exactly "
            "what you do not know when you are deciding whether to ask.",
        ]),
    ],
    advice="Two experts are offered so you can see the method behave differently on each.",
    related=["deferral", "region", "competence"],
)

_add(
    slug="region", title="Regions of the input space", group="The idea",
    short="Groups of similar articles, found automatically by clustering the text.",
    sections=[
        ("What it means", [
            "Articles about the same sort of thing use the same sort of words. Grouping "
            "them by word usage produces clusters -- one might be full of company results "
            "and share prices, another full of match reports.",
            "These groups are the 'regions' the expert specialises in. Nobody labelled "
            "them; they fall out of the text.",
        ]),
        ("Why it matters here", [
            "It gives 'this expert is good at some things' a precise meaning that a "
            "computer can check, and one you can read: each region is shown with the words "
            "that most define it.",
        ]),
    ],
    advice="Look at the region table to see what each expert is actually good at.",
    related=["expert", "competence"],
)

_add(
    slug="competence", title="Competence model", group="How it works",
    short="A small model that predicts how likely someone is to be right on a given article, without knowing the answer.",
    sections=[
        ("Two of them", [
            "One estimates how likely the classifier is to be right. One estimates how "
            "likely the expert is to be right. Both look only at the article.",
            "Neither predicts the topic. They predict whether an answer would be correct, "
            "which is a different and harder question.",
        ]),
        ("Where the classifier's one comes from free", [
            "We already know when the classifier was wrong on the training articles, so "
            "its competence model costs nothing to build. The expert's is the expensive "
            "one, because every data point means asking a person.",
        ]),
    ],
    advice="This asymmetry is why the active learning is about the expert, not the classifier.",
    related=["out-of-fold", "advantage", "active-learning"],
)

_add(
    slug="advantage", title="The advantage score", group="How it works",
    short="How much more likely the expert is to be right than the classifier, on this article.",
    sections=[
        ("The rule in one line", [
            "advantage = (chance the expert is right) − (chance the classifier is right).",
            "Positive means hand over. The bigger it is, the clearer the case.",
        ]),
        ("Why the size matters, not just the sign", [
            "An advantage of 0.01 means it is nearly a coin flip and probably not worth a "
            "person's time. Having a number rather than a yes/no lets you set how sure you "
            "want to be before bothering someone.",
        ]),
    ],
    advice="The threshold slider is what turns this number into a decision.",
    related=["threshold", "competence"],
)

_add(
    slug="threshold", title="The threshold", group="How it works",
    short="How much better the expert has to look before the system actually hands over.",
    sections=[
        ("What moving it does", [
            "At a low threshold the system hands over readily -- more articles reach the "
            "expert, and more of those handovers turn out to have been unnecessary.",
            "At a high threshold it hands over only the clearest cases. Fewer wasted "
            "handovers, but it keeps articles it would have been better off passing on.",
        ]),
        ("Whose choice it is", [
            "Nobody can pick this for you, because it depends on how expensive the "
            "expert's time is compared to a mistake. That is why it is a control rather "
            "than a constant buried in the code.",
        ]),
    ],
    advice="Drag it and watch the four outcome counts move.",
    related=["advantage", "deferral-rate", "outcomes"],
)

_add(
    slug="outcomes", title="The four outcomes", group="Measuring",
    short="Every article falls into one of four cases, and only one of them is a reason to hand over.",
    sections=[
        ("The four cases", [
            "Both would be right: handing over changes nothing and wastes the expert's time.",
            "Only the classifier is right: handing over actively makes the answer worse.",
            "Only the expert is right: this is the one case worth handing over.",
            "Neither is right: handing over is harmless and pointless.",
        ]),
        ("Why not just count correct answers", [
            "A single accuracy figure treats all four the same. Two systems with identical "
            "accuracy can be wasting wildly different amounts of a person's time.",
        ]),
    ],
    advice="This table is the honest picture of what a deferral policy is doing.",
    related=["deferral-precision", "team-accuracy"],
)

_add(
    slug="deferral-precision", title="Deferral precision and recall", group="Measuring",
    short="Of the articles handed over, how many needed to be; and of those that needed handing over, how many were.",
    sections=[
        ("Precision", [
            "Out of everything sent to the expert, the share where the expert really was "
            "right and the classifier really was wrong. Low precision means the person is "
            "being interrupted for nothing.",
        ]),
        ("Recall", [
            "Out of all the articles where the expert would have rescued a wrong answer, "
            "the share actually sent. Low recall means the system is keeping articles it "
            "cannot handle.",
        ]),
        ("Why both", [
            "You can get perfect recall by handing over everything, and near-perfect "
            "precision by handing over almost nothing. Only together do they describe a "
            "sensible policy.",
        ]),
    ],
    advice="A good policy holds both up at once; watch them trade off as you move the threshold.",
    related=["outcomes", "team-accuracy", "oracle"],
)

_add(
    slug="team-accuracy", title="Team accuracy", group="Measuring",
    short="How often the answer is right once the classifier and the expert have divided the work.",
    sections=[
        ("How it is counted", [
            "For each article, whoever was assigned it gives the answer, and we check "
            "whether it is right. The share correct is the team's accuracy.",
        ]),
        ("What it hides", [
            "It says nothing about how much of the expert's time was spent. A team that "
            "hands over everything can look respectable while having learned nothing.",
        ]),
    ],
    advice="Read it next to the deferral rate; alone it is easy to game.",
    related=["outcomes", "deferral-rate", "oracle"],
)

_add(
    slug="deferral-rate", title="Deferral rate", group="Measuring",
    short="The share of articles handed to the expert.",
    sections=[
        ("Why it is a cost", [
            "Each handover is a person reading an article. A policy that gains half a "
            "point of accuracy by handing over 60% of the work is usually a bad deal.",
        ]),
    ],
    advice="Compare policies at similar deferral rates, or the comparison is unfair.",
    related=["team-accuracy", "threshold"],
)

_add(
    slug="oracle", title="The oracle", group="Measuring",
    short="A cheating policy that always knows who was right. Not achievable — it marks the ceiling.",
    sections=[
        ("What it does", [
            "It hands over exactly the articles where the expert is right and the "
            "classifier is wrong, and keeps everything else. To do that it has to know the "
            "correct answers in advance, which no real system does.",
        ]),
        ("What it is for", [
            "It says how much room there is. If the oracle scores 97.7% and the classifier "
            "alone scores 92.3%, then 5.4 points are theoretically available and a real "
            "policy can be judged by how much of that it captures.",
        ]),
    ],
    advice="Always read a result as a fraction of the oracle's headroom, not on its own.",
    related=["team-accuracy", "outcomes"],
)

_add(
    slug="active-learning", title="Active learning", group="Asking questions",
    short="Choosing which questions to ask, when asking is expensive.",
    sections=[
        ("The setting", [
            "At the start we have no idea what the expert is good at. We can find out only "
            "by showing them an article and seeing what they say. Every question costs "
            "their time, so we cannot ask about all 120,000.",
        ]),
        ("The idea", [
            "Some articles teach you much more than others. Choosing them deliberately, "
            "instead of at random, is active learning.",
        ]),
    ],
    advice="The comparison to beat is simply picking articles at random.",
    related=["acquisition", "query-budget", "competence"],
)

_add(
    slug="acquisition", title="How the next question is chosen", group="Asking questions",
    short="Ask about the articles where it is least clear whether handing over is the right call.",
    sections=[
        ("The rule", [
            "For each article we have a guess at how much better the expert would be. "
            "Where that guess sits near zero, the decision is on a knife edge and an answer "
            "would settle it. Those are the articles worth asking about.",
        ]),
        ("The trap this avoids", [
            "The obvious idea is to ask about whatever the classifier finds hardest. But "
            "an article that baffles the classifier is only worth a question if the expert "
            "might do better. If both are lost, learning that teaches nothing about when to "
            "hand over.",
            "That alternative is included on the page as a comparison, so you can see "
            "whether the reasoning holds up rather than taking it on trust.",
        ]),
        ("Starting from nothing", [
            "With no answers yet there is no guess to be near the edge of, so the first "
            "batch is spread evenly across the regions. Otherwise the search would chase "
            "whatever its first few answers happened to suggest and never look elsewhere.",
        ]),
    ],
    advice="Compare the four strategies on the curve; the differences are real but modest.",
    related=["active-learning", "query-budget", "region"],
)

_add(
    slug="query-budget", title="Query budget", group="Asking questions",
    short="How many articles the expert has been asked about so far.",
    sections=[
        ("Reading the curve", [
            "The horizontal axis is how many questions have been asked; the vertical axis "
            "is how well the resulting team does. A strategy is better if its line is "
            "higher for the same number of questions.",
            "Every line starts at the same point because all strategies share the same "
            "opening batch, so early on there is nothing to separate them.",
        ]),
    ],
    advice="Look at where lines separate, not just where they end.",
    related=["active-learning", "acquisition"],
)

_add(
    slug="out-of-fold", title="Out-of-fold scoring", group="Under the bonnet",
    short="Testing the classifier on articles it was not trained on, so its error rate is honest.",
    sections=[
        ("The problem it fixes", [
            "A model scored on the very examples it learned from looks far better than it "
            "is. Here it appeared to make 3,403 mistakes that way, against 11,350 when "
            "tested properly.",
            "The competence model learns from those mistakes, so using the flattering "
            "number would have taught it the classifier almost never needs help.",
        ]),
        ("How it works", [
            "Split the data into five parts. Train on four, predict the fifth, rotate. "
            "Every article ends up predicted by a model that never saw it.",
        ]),
    ],
    advice="Nothing to do; it is why the numbers here are trustworthy.",
    related=["competence", "classifier"],
)

_add(
    slug="classifier", title="The classifier", group="Under the bonnet",
    short="The machine that sorts articles into topics on its own, correct 92.3% of the time.",
    sections=[
        ("What it is", [
            "It counts which words and word pairs appear in an article and weighs up which "
            "topic those words point to. No neural network, no pretrained model.",
        ]),
        ("Why this one", [
            "A slightly more accurate alternative was available (92.9%) but it reports only "
            "its guess, not how sure it is. Everything on these pages depends on knowing "
            "how confident it was, so half a point of accuracy was traded for that.",
        ]),
    ],
    advice="Its 92.3% is the number every other policy has to beat.",
    related=["out-of-fold", "confidence", "tfidf"],
)

_add(
    slug="confidence", title="Confidence", group="Under the bonnet",
    short="How sure the classifier is, which turns out to predict its mistakes better than the article's words do.",
    sections=[
        ("Where it comes from", [
            "The classifier scores all four topics and the gap between the top two says how "
            "close the call was. A wide gap is a confident answer.",
        ]),
        ("A result worth knowing", [
            "That single gap predicts whether the classifier is right about as well as all "
            "50,000 word features put together, and slightly better. When a model is "
            "unsure, it is usually unsure for a reason.",
        ]),
    ],
    advice="It is fed to the competence models directly rather than left to be rediscovered.",
    related=["competence", "classifier"],
)

_add(
    slug="tfidf", title="Turning text into numbers", group="Under the bonnet",
    short="Counting words, weighted so that common words count for little.",
    sections=[
        ("The idea", [
            "'The' appears everywhere and tells you nothing. 'Nasdaq' appears rarely and "
            "tells you a lot. Weighting each word by how rare it is across all articles "
            "gives every article a long list of numbers a model can work with.",
        ]),
        ("And then squashing it", [
            "That list is 50,000 numbers long, which is far too many to learn from a few "
            "hundred expert answers. It is compressed to about a hundred summary numbers "
            "first, which loses very little and makes the competence models learnable.",
        ]),
    ],
    advice="Nothing to set; it is why a few hundred questions are enough.",
    related=["classifier", "competence"],
)


def groups():
    ordered = {}
    for topic in TOPICS.values():
        ordered.setdefault(topic.group, []).append(topic)
    return ordered
