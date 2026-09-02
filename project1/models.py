from django.db import models


class TrainingRun(models.Model):
    """A record of every sweep, so runs on the same data can be compared
    afterwards instead of being remembered."""

    created = models.DateTimeField(auto_now_add=True)
    dataset = models.CharField(max_length=200)
    kind = models.CharField(max_length=20)
    algorithm = models.CharField(max_length=60)
    param = models.CharField(max_length=40)
    grid = models.CharField(max_length=300)
    chosen = models.FloatField()
    peak = models.FloatField()
    rule = models.CharField(max_length=20)
    score_name = models.CharField(max_length=40)
    cv_score = models.FloatField()
    test_score = models.FloatField()
    test_fraction = models.FloatField()
    n_folds = models.PositiveSmallIntegerField()
    seed = models.IntegerField()
    automated = models.BooleanField(default=False)

    class Meta:
        ordering = ["-created"]

    def __str__(self):
        return f"{self.algorithm} ({self.param}={self.chosen:g}) on {self.dataset}"

    @classmethod
    def record(cls, dataset_name, result, test_fraction, seed, automated=False):
        return cls.objects.create(
            dataset=dataset_name,
            kind=result.algo.kind,
            algorithm=result.algo.label,
            param=result.algo.param,
            grid=", ".join(f"{r.value:g}" for r in result.rows),
            chosen=result.winner.value,
            peak=result.peak.value,
            rule=result.rule,
            score_name=result.score.label,
            cv_score=result.winner.mean,
            test_score=result.headline,
            test_fraction=test_fraction,
            n_folds=result.n_folds,
            seed=seed,
            automated=automated,
        )
