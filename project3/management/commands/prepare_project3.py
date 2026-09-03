from django.core.management.base import BaseCommand

from project3 import prepare


class Command(BaseCommand):
    help = "Train and cache everything project 3 needs. Takes several minutes."

    def add_arguments(self, parser):
        parser.add_argument("--stage", default="all",
                            choices=["all", "baseline", "world", "experiments"])

    def handle(self, *args, **options):
        stage = options["stage"]
        if stage in ("all", "baseline"):
            self.stdout.write("Baseline classifier and out-of-fold predictions...")
            prepare.build_baseline()
        if stage in ("all", "world"):
            self.stdout.write("Regions, reduced features and simulated experts...")
            prepare.build_world()
        if stage in ("all", "experiments"):
            self.stdout.write("Deferral sweep and active learning curves...")
            prepare.build_experiments()
        self.stdout.write(self.style.SUCCESS("done"))
