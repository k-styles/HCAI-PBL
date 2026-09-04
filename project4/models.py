"""What the study records.

Session storage would be simpler and would also lose every response the moment
a participant closed the tab, which is not a defensible way to run a study --
so responses go in the database. Nothing here identifies anybody: a random
token, the arm they were assigned, and one coarse self-report used for
stratification.
"""

from django.db import models


class Participant(models.Model):
    PAIR, RANK = "pair", "rank"
    ARMS = [(PAIR, "Design 1 - pairwise choice"), (RANK, "Design 2 - rank ten")]

    FREQUENCY = [("rarely", "Less than once a month"),
                 ("monthly", "One to three films a month"),
                 ("weekly", "About one a week"),
                 ("often", "Several a week")]

    token = models.CharField(max_length=16, unique=True)
    arm = models.CharField(max_length=8, choices=ARMS)
    frequency = models.CharField(max_length=10, choices=FREQUENCY, blank=True)
    seed = models.IntegerField()
    preview = models.BooleanField(default=False)
    started = models.DateTimeField(auto_now_add=True)
    elicitation_seconds = models.FloatField(default=0.0)
    finished = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-started"]

    def __str__(self):
        return f"{self.token} ({self.arm})"

    @property
    def elicitations(self):
        return self.responses.filter(phase=Response.ELICIT).order_by("position")

    @property
    def holdouts(self):
        return self.responses.filter(phase=Response.HOLDOUT).order_by("position")


class Response(models.Model):
    ELICIT, HOLDOUT = "elicit", "holdout"
    PHASES = [(ELICIT, "Elicitation"), (HOLDOUT, "Held-out test")]

    participant = models.ForeignKey(Participant, on_delete=models.CASCADE,
                                    related_name="responses")
    phase = models.CharField(max_length=8, choices=PHASES)
    position = models.IntegerField()
    # Film indices into the pool, best first. A pairwise choice is the n = 2
    # case of a ranking, which is the whole point of Task 2 -- so one column
    # holds both and the fitting code never has to branch on the arm.
    order = models.JSONField()
    seconds = models.FloatField()
    # Set on the two repeated held-out pairs, so consistency can be scored.
    repeat_of = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["participant", "phase", "position"]
        constraints = [models.UniqueConstraint(fields=["participant", "phase", "position"],
                                               name="one_response_per_slot")]
