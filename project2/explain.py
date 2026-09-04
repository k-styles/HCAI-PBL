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


TOPICS = {}


def _add(**kw):
    TOPICS[kw["slug"]] = Topic(**kw)


_add(slug="penguins", title="The penguins", group="The data",
     short="344 penguins from three species, measured on four body dimensions.",
     sections=[("What is in it", [
         "Real measurements from Palmer Station, Antarctica: bill length and depth, "
         "flipper length, body mass, plus which island the bird was on, its sex and the "
         "year. The job is to work out the species from the rest.",
         "Eleven of the 344 rows are missing something and are set aside, leaving 333."]),
       ("Why it is a good example", [
         "Two of the three species are easy to tell apart and two are genuinely similar, "
         "so a model has to do real work without the problem being hopeless."])],
     advice="Everything on these pages is about this one table.",
     related=["accuracy", "test-set"])

_add(slug="tree", title="Decision tree", group="Models",
     short="A flowchart of yes/no questions that ends in an answer.",
     sections=[("How it works", [
         "\"Is the flipper shorter than 207 mm? If yes, is the bill shorter than 43 mm? "
         "If yes, it's an Adelie.\" The model works out the questions itself.",
         "You can read it. That is unusual for a model and it is the reason this project "
         "uses one."]),
       ("Where it goes wrong", [
         "Left alone it keeps asking questions until it has a private rule for every bird "
         "in the training data, including the odd ones. That looks perfect on data it has "
         "seen and disappoints on anything new."])],
     advice="Read the tree as sentences rather than as a diagram; it is the same model.",
     related=["leaves", "overfitting", "complexity"])

_add(slug="logistic", title="Logistic regression", group="Models",
     short="Gives each measurement a weight, adds them up, and turns the total into a probability.",
     sections=[("How it works", [
         "Each feature gets a number saying how much it pushes towards each species. Add "
         "them up and you get a score per species; the biggest wins."]),
       ("Why it is here", [
         "It is the natural contrast to a tree. A tree asks questions in sequence; this "
         "weighs everything at once. They are readable in different ways, which is why the "
         "brief asks for a complexity measure for each."])],
     advice="Its readability is about how many features carry a non-zero weight.",
     related=["coefficient", "sparsity", "complexity"])

_add(slug="leaves", title="Leaves", group="Complexity",
     short="The number of endings a tree has — how many different answers it can give.",
     sections=[("What it counts", [
         "Every path through the questions ends at a leaf, and each leaf is one verdict. "
         "Three leaves means three rules to read; fourteen means fourteen."]),
       ("Why count them", [
         "It is a direct measure of how much you have to take in before you can say what "
         "the model does. The brief fixes this as the complexity measure for trees."])],
     advice="More leaves is not automatically better; watch the accuracy stop improving.",
     related=["complexity", "tree", "lambda"])

_add(slug="complexity", title="Complexity, Ω", group="Complexity",
     short="How much of a model you have to read before you understand it.",
     sections=[("The idea", [
         "Two models can be equally accurate and wildly different to live with. One is "
         "three sentences; the other is fourteen branching rules. Ω puts a number on that."]),
       ("It means different things per model", [
         "For a tree it is the number of leaves — the brief says so. For logistic "
         "regression we chose the number of features with a non-zero weight, because a "
         "feature weighted zero costs the reader nothing."])],
     advice="Ω is the quantity the slider trades against accuracy.",
     related=["leaves", "sparsity", "lambda"])

_add(slug="lambda", title="Lambda, the slider", group="Complexity",
     short="How much accuracy one unit of complexity has to buy before it is worth having.",
     sections=[("What it does", [
         "The app picks whichever model scores highest on accuracy minus λ times "
         "complexity. At λ = 0 complexity is free and the most accurate model wins. Turn λ "
         "up and each extra leaf has to earn its place."]),
       ("Why it is yours to set", [
         "It encodes how much you care about being able to read the model, which depends "
         "on what you need it for. Nobody can pick it for you, so it is a control rather "
         "than a number in the code."])],
     advice="Watch the table of λ ranges: there are only a handful of models it can ever give you.",
     related=["complexity", "reachable", "overfitting"])

_add(slug="reachable", title="Which models the slider can reach", group="Complexity",
     short="Most trained models can never be selected, at any setting of the slider.",
     sections=[("Why", [
         "For a fixed model, accuracy minus λ times complexity is a straight line as λ "
         "changes. Picking the best model is picking the highest line, and only the lines "
         "on the top edge are ever highest.",
         "A model that is both more complicated and less accurate than another is beaten "
         "everywhere. So is one that sits below the line joining two others."]),
       ("What it buys you", [
         "It turns a dial you drag blindly into a short list with exact boundaries: here "
         "are the four models available, and here is the range of λ that gives each."])],
     advice="Greyed points in the plot are models no setting will ever produce.",
     related=["lambda", "complexity"])

_add(slug="sparsity", title="Sparsity and the L1 penalty", group="Complexity",
     short="Pushing a model to use as few features as possible by driving weights to exactly zero.",
     sections=[("The mechanism", [
         "An L1 penalty charges the model for the size of its weights in a way that makes "
         "the cheapest option often exactly zero. A feature with a zero weight is simply "
         "not consulted."]),
       ("Why not the other penalty", [
         "The more common L2 penalty shrinks weights towards zero but never reaches it, so "
         "every feature stays in the model, and the count we use as Ω would never move. "
         "The slider would do nothing."])],
     advice="Look at the coefficient table — dropped features are greyed out.",
     related=["coefficient", "complexity", "logistic"])

_add(slug="coefficient", title="Coefficients", group="Models",
     short="The weight each measurement carries, and in which direction.",
     sections=[("Reading them", [
         "A large positive number means that measurement pushes strongly towards that "
         "species; a large negative one pushes away. Zero means it is ignored."]),
       ("A catch", [
         "Raw weights are not comparable across features measured in millimetres and "
         "grams, so the model is fitted on rescaled measurements and the numbers shown are "
         "on that scale."])],
     advice="Compare their sizes, not their raw units.",
     related=["sparsity", "logistic", "scaling"])

_add(slug="accuracy", title="Accuracy", group="Measuring",
     short="Of all the predictions made, the share that were right.",
     sections=[("The arithmetic", [
         "Count correct predictions, divide by the total. That is all."]),
       ("What it hides", [
         "It says nothing about which species the mistakes fall on. With three species of "
         "unequal size, a model can look respectable while being useless on the smallest."])],
     advice="Always read it next to how complicated the model is.",
     related=["test-set", "penguins"])

_add(slug="test-set", title="Test set", group="Measuring",
     short="Birds held back from training, used to check the model honestly.",
     sections=[("Why", [
         "A model can always recite the examples it learned from. The only meaningful "
         "question is how it does on birds it has never seen, so 30% are set aside."]),
       ("Used consistently", [
         "Every model on these pages is scored on the same held-back birds, so a model "
         "cannot look better simply by having been given an easier test."])],
     advice="Nothing to set; it is why the numbers are comparable.",
     related=["accuracy", "overfitting"])

_add(slug="overfitting", title="Overfitting", group="Measuring",
     short="When a model memorises the examples it was shown instead of the pattern behind them.",
     sections=[("The exam analogy", [
         "A student who memorises last year's paper aces last year's paper and fails this "
         "year's. They learned the answers, not the subject."]),
       ("How to spot it here", [
         "Give a tree more leaves and its accuracy on the training birds keeps climbing "
         "towards perfect while its accuracy on held-back birds stops improving. That gap "
         "is the memorising."])],
     advice="This is the whole reason for penalising complexity at all.",
     related=["lambda", "test-set", "leaves"])

_add(slug="counterfactual", title="Counterfactual", group="Explaining a prediction",
     short="The smallest change that would have made the model say something else.",
     sections=[("The question it answers", [
         "Not \"why did it say Adelie\" but \"what would have had to be different for it to "
         "say Chinstrap\". That is usually the more useful question, because it is the one "
         "you could act on."]),
       ("How they are found", [
         "Thousands of slightly altered penguins are generated around the real one, the "
         "model is asked about each, and the ones that get the answer you wanted are ranked "
         "by how little had to change."])],
     advice="Prefer the ones that change one measurement; they are the readable answers.",
     related=["mad", "noising"])

_add(slug="mad", title="MAD-weighted distance", group="Explaining a prediction",
     short="A way of measuring how big a change is when the things changing are in different units.",
     sections=[("The problem", [
         "Is 200 grams of body mass a bigger change than 2 millimetres of bill length? In "
         "raw numbers 200 dwarfs 2, but that is only because grams are small units."]),
       ("The fix", [
         "Divide each change by how much that measurement normally varies between penguins "
         "— its median absolute deviation. Then a change counts as large only if it is "
         "large for that measurement."]),
       ("Categories", [
         "An island is not 0.4 of another island, so a change of category counts as one "
         "unit and no change counts as zero."])],
     advice="It is what ranks the counterfactuals; lower is a smaller ask.",
     related=["counterfactual", "noising"])

_add(slug="noising", title="How the alternatives are generated", group="Explaining a prediction",
     short="Numbers get nudged; categories get swapped, because there is nothing in between two islands.",
     sections=[("Two kinds of feature", [
         "A bill length can be nudged slightly. An island cannot — there is no value "
         "halfway between Biscoe and Dream — so it either changes or it does not.",
         "Each generated penguin leaves most categories alone and, when it does change one, "
         "picks a replacement from what real penguins actually have."]),
       ("One more choice", [
         "Each attempt is allowed to alter only some of the four measurements, chosen at "
         "random. Without that, every answer changes all four slightly, and \"adjust "
         "everything a bit\" is useless as advice."])],
     advice="This is why the best counterfactuals name one or two things, not seven.",
     related=["counterfactual", "mad"])

_add(slug="pdp", title="Partial dependence", group="What a feature does",
     short="Force every penguin to the same bill length, ask the model, and average the answers.",
     sections=[("How it is built", [
         "Pick a value. Pretend every bird in the data had exactly that bill length, leave "
         "everything else alone, and average what the model says. Repeat across the range."]),
       ("Its weakness", [
         "Flipper length and body mass go together in real penguins. Setting a flipper to "
         "230 mm while leaving body mass alone asks the model about an animal that does not "
         "exist, and the curve is an average over impossible birds as well as real ones."])],
     advice="Compare it with the ALE curve; where they disagree, the ALE is the safer read.",
     related=["ale", "correlation"])

_add(slug="ale", title="Accumulated local effects", group="What a feature does",
     short="Like partial dependence, but only ever asks the model about birds that could exist.",
     sections=[("How it is built", [
         "Split the range into bands. Inside each band use only the birds whose own bill "
         "length falls in that band, see how much the answer shifts across the band, and "
         "add those shifts up along the range."]),
       ("What you get and give up", [
         "Nothing impossible is ever constructed. The price is that the curve shows change "
         "relative to a typical bird rather than an absolute probability, so it is centred "
         "on zero."]),
       ("Two ways of computing it", [
         "For logistic regression the slope can be written down exactly. A tree is flat "
         "between its split points, so it has no slope to speak of and the shift has to be "
         "measured across each band instead. The page says which was used."])],
     advice="Steps in the curve for a tree are its split points; that is the correct shape.",
     related=["pdp", "tree", "correlation"])

_add(slug="correlation", title="Correlation", group="What a feature does",
     short="How strongly two measurements move together, from −1 to +1.",
     sections=[("Reading it", [
         "Near +1 means when one goes up so does the other. Near 0 means no straight-line "
         "relationship. Petal-like measurements on penguins are strongly related."]),
       ("Why it matters here", [
         "Two measurements that say nearly the same thing make partial dependence "
         "misleading, and they make a linear model's weights unstable."])],
     advice="Check it before trusting a partial dependence curve.",
     related=["pdp", "ale"])

_add(slug="scaling", title="Putting features on the same scale", group="Models",
     short="Rescaling so a measurement in grams does not drown out one in millimetres.",
     sections=[("The problem", [
         "Body mass runs to thousands; bill depth is around fifteen. Any model that weighs "
         "features against each other would be dominated by body mass purely because its "
         "numbers are bigger."]),
       ("What is done", [
         "Every measurement is shifted and stretched to a comparable range before fitting "
         "logistic regression. Trees are left alone, because they only ask whether a value "
         "is above or below a threshold and that does not change."])],
     advice="Handled automatically; it explains why coefficients are comparable.",
     related=["coefficient", "logistic"])


def groups():
    ordered = {}
    for topic in TOPICS.values():
        ordered.setdefault(topic.group, []).append(topic)
    return ordered
