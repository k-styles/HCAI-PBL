from django import forms

from . import learning

RULES = [
    ("one_se", "One standard error - prefer the simpler model among the ties"),
    ("best", "Best mean - take the top score outright"),
]


class UploadForm(forms.Form):
    file = forms.FileField(
        label="CSV file",
        help_text="First row holds the column names, the last column is the value to predict.")
    kind = forms.ChoiceField(
        label="Problem type", required=False,
        choices=[("", "Work it out from the data"),
                 ("classification", "Classification"),
                 ("regression", "Regression")])

    def clean_file(self):
        uploaded = self.cleaned_data["file"]
        if not uploaded.name.lower().endswith((".csv", ".txt")):
            raise forms.ValidationError("Please upload a .csv file.")
        if uploaded.size > 8 * 1024 * 1024:
            raise forms.ValidationError("That file is larger than 8 MB.")
        return uploaded


class TrainingForm(forms.Form):
    algorithm = forms.ChoiceField(label="Model")
    score = forms.ChoiceField(label="Score to select on")
    grid = forms.CharField(label="Values to try", required=False)
    test_percent = forms.IntegerField(label="Test set (%)", min_value=10, max_value=50, initial=25)
    n_folds = forms.IntegerField(label="Cross-validation folds", min_value=2, max_value=10, initial=5)
    rule = forms.ChoiceField(label="How to pick the winner", choices=RULES, initial="one_se")
    seed = forms.IntegerField(label="Random seed", min_value=0, max_value=99999, initial=0)

    def __init__(self, kind, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["algorithm"].choices = [(a.key, a.label) for a in learning.algorithms_for(kind)]
        self.fields["score"].choices = [(s.key, s.label) for s in learning.scores_for(kind)]

    def clean(self):
        cleaned = super().clean()
        key = cleaned.get("algorithm")
        if key:
            try:
                cleaned["values"] = learning.parse_grid(learning.ALGORITHMS[key], cleaned.get("grid"))
            except ValueError as problem:
                self.add_error("grid", str(problem))
        return cleaned
