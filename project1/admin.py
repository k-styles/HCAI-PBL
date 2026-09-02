from django.contrib import admin

from .models import TrainingRun


@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    list_display = ("created", "dataset", "algorithm", "chosen", "score_name",
                    "test_score", "automated")
    list_filter = ("kind", "algorithm", "automated")
