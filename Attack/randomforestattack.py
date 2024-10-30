from __future__ import absolute_import, division, print_function, unicode_literals, annotations

import numpy as np
import warnings


from sklearn.ensemble import RandomForestClassifier



import logging
from tqdm.auto import trange
from tqdm.auto import tqdm
import random
from art.attacks.attack import EvasionAttack
from art.estimators.classification.scikitlearn import ScikitlearnDecisionTreeClassifier
from art.attacks.evasion import DecisionTreeAttack
from art.utils import check_and_transform_label_format
warnings.filterwarnings('ignore')





logger = logging.getLogger(__name__)

class RandomForestAttack(EvasionAttack):
    """
    This an attempt to implement Papernot's attack on random forest.

    | Paper link: https://arxiv.org/abs/1605.07277
    """

    attack_params = ["classifier", "offset", "verbose", "n_estimators"]
    _estimator_requirements = (RandomForestClassifier,)
    def __init__(
        self,
        classifier: RandomForestClassifier,
        n_estimators: int =1,
        offset: float = 0.001,
        verbose: bool = True,

    ) -> None:
        """
        :param classifier: A trained scikit-learn random forest model.
        :param list_of_estimators: a list of the estimators of the random forest model.
        :param offset: How much the value is pushed away from tree's threshold.
        :param verbose: Show progress bars.
        """
        super().__init__(estimator=classifier)
        self.classifier = classifier
        self.offset = offset
        self.verbose = verbose
        self._check_params()


    def generate(self, x: np.ndarray, y: np.ndarray | None = None,  n_estimators: int | None = None,**kwargs) -> np.ndarray:
        """
        Generate adversarial examples and return them as an array.

        :param x: An array with the original inputs to be attacked.
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :param n_estimators: an integer that specificy the number of estimators object of a decision tree attack.
        :return: An array holding the adversarial examples.
        """
        if y is not None:
            y = check_and_transform_label_format(y, nb_classes=self.estimator.nb_classes, return_one_hot=False)
        x_adv = x.copy()
        list_of_estimators = self.classifier.estimators_
        if n_estimators is None:
          random_estimator = random.randint(0, len(list_of_estimators))
          art_classifier = ScikitlearnDecisionTreeClassifier(model=list_of_estimators[random_estimator])
          DTA = DecisionTreeAttack(classifier=art_classifier, verbose=False)
          x_adv = DTA.generate(x_adv)
        else:
          random_estimators = [random.randint(0, len(list_of_estimators)) for _ in range(n_estimators)]
          for estimator in tqdm(random_estimators, desc="Random Forest attack", disable=not self.verbose):
            art_classifier = ScikitlearnDecisionTreeClassifier(model=list_of_estimators[estimator])
            DTA = DecisionTreeAttack(classifier=art_classifier, verbose=False)
            x_gen_adv = DTA.generate(x_adv)
            x_adv = x_gen_adv


        return x_adv

    def _check_params(self) -> None:

        if self.offset <= 0:
            raise ValueError("The offset parameter must be strictly positive.")

        if not isinstance(self.verbose, bool):
            raise ValueError("The argument `verbose` has to be of type bool.")