### SA functionality

from statistics import mean
from typing import List

import numpy as np
from numpy import float64

from classes.schrock_paralell import Schrock
from sa.sascorer import sa_target_score_clipped


class SaScorer:
    def get_sa(self, molecules: List[Schrock]) -> None:
        """Get the SA score of the population."""
        # Get the scores
        # Scale with gaussian. We prefer molecules with sa scores above 0.80.
        sa_scores_all = [
            [sa_target_score_clipped(lig.mol) for lig in ind.ligands]
            for ind in molecules
        ]
        # TODO Better way of combining the individual scores

        sa_scores = [mean(scores) for scores in sa_scores_all]

        # Scale with gaussian
        sa_scores = [
            np.exp(-0.5 * np.power((score - 1) / 0.8, 2.0)) for score in sa_scores
        ]

        # Set the scores
        self.set_sa(sa_scores, molecules)

    def set_sa(self, sa_scores: List[float64], molecules: List[Schrock]) -> None:
        """Set sa score.

        If score is high, then score is not modified
        """
        for individual, sa_score in zip(molecules, sa_scores):
            individual.sa_score = sa_score

            individual.score = sa_score * individual.score
