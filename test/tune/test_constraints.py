import unittest
from functools import partial

import numpy as np
from fairlearn.metrics import equalized_odds_difference
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from flaml import tune, BlendSearch
from flaml.automl import SearchState
from flaml.automl.model import LGBMEstimator

from ray.tune.integration.lightgbm import TuneReportCheckpointCallback
from ray.tune.schedulers import AsyncHyperBandScheduler

from flaml.automl.task.factory import task_factory
from flaml.tune.scheduler.stratum_asha import StratumAsyncHyperBandScheduler

dataset = "default-of-credit-card-clients"
seed = 42
SEX = 'x2'
EVAL_NAME = 'eval'


def preprocess(X, y):
    # Make 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6' categorical
    X = X.astype(dict((f'x{i}', 'category') for i in [3, 4, 6, 7, 8, 9, 10, 11]))
    y = y.astype('float64')
    # Add a synthetic feature
    dist_scale = 0.5
    np.random.seed(seed)
    LIMIT_BAL = 'x1'
    # Make 'LIMIT_BAL' informative of the target
    X[LIMIT_BAL] = y + np.random.normal(scale=dist_scale, size=X.shape[0])
    # But then make it uninformative for the male clients
    X.loc[X[SEX] == 1, LIMIT_BAL] = np.random.normal(scale=dist_scale, size=X[X[SEX] == 1].shape[0])
    return X, y


def train_credit_card_default(config, X, y):
    X, y = preprocess(X, y)
    sensitive_str = X[SEX].map({2: "female", 1: "male"})
    X_train, X_test, y_train, y_test, _, sensitive_str_test = train_test_split(
        X.drop(columns=[SEX]),
        y,
        sensitive_str,
        test_size=0.33,
        random_state=seed,
        stratify=y)

    def fair_metric(labels, preds):
        threshold = np.mean(y_train)
        y_pred = (preds >= threshold) * 1
        eod = equalized_odds_difference(labels, y_pred, sensitive_features=sensitive_str_test)
        return "eod", eod, False

    lgb_params = {
        'task': 'binary',
        # "n_jobs": 1,
    }
    lgb_params.update(config)
    model = LGBMEstimator(**lgb_params)
    model.fit(X_train, y_train,
              eval_set=(X_test, y_test),
              eval_names=[EVAL_NAME],
              eval_metric=['auc', fair_metric],
              callbacks=[TuneReportCheckpointCallback()])


def run_metric_constraint(exp_name, scheduler_name):
    metric = f"{EVAL_NAME}-auc"
    constraint_metric = f"{EVAL_NAME}-eod"
    time_budget = 120
    mode = "max"
    metric_constraints = [(constraint_metric, "<=", 0.3)]
    # default-of-credit-card-clients
    X, y = fetch_openml(return_X_y=True, data_id=42477)
    search_state = SearchState(learner_class=LGBMEstimator, data=X, task=task_factory('binary'), budget=time_budget)
    if scheduler_name == "stratum_asha":
        scheduler = StratumAsyncHyperBandScheduler(max_t=X.shape[0], metric_constraints=metric_constraints)
    elif scheduler_name == "asha":
        scheduler = AsyncHyperBandScheduler(max_t=X.shape[0])
    else:
        scheduler = None
    analysis = tune.run(
        partial(train_credit_card_default, X=X, y=y),
        name=exp_name,
        scheduler=scheduler,
        search_alg=BlendSearch(
            metric=metric,
            mode=mode,
            space=search_state.search_space,
            points_to_evaluate=search_state.init_config,
            low_cost_partial_config=search_state.low_cost_partial_config,
            metric_constraints=metric_constraints,
            seed=seed,
            time_budget_s=time_budget,
        ),
        num_samples=5,
        max_concurrent_trials=1,
        checkpoint_score_attr=metric,
        keep_checkpoints_num=1,
        raise_on_failed_trial=False,
        use_ray=True,
        verbose=3,
    )
    print(f"best result: {analysis.best_result}")


class TestConstraint(unittest.TestCase):
    def test_config_constraint(self):
        # Test dict return value
        def evaluate_config_dict(config):
            metric = (round(config["x"]) - 85000) ** 2 - config["x"] / config["y"]
            return {"metric": metric}

        def config_constraint(config):
            if config["y"] >= config["x"]:
                return 1
            else:
                return 0

        analysis = tune.run(
            evaluate_config_dict,
            config={
                "x": tune.qloguniform(lower=1, upper=100000, q=1),
                "y": tune.qrandint(lower=2, upper=100000, q=2),
            },
            config_constraints=[(config_constraint, "<", 0.5)],
            metric="metric",
            mode="max",
            num_samples=100,
            log_file_name="logs/config_constraint.log",
        )

        assert analysis.best_config["x"] > analysis.best_config["y"]
        assert analysis.trials[0].config["x"] > analysis.trials[0].config["y"]

    def test_metric_constraint_asha(self):
        run_metric_constraint(exp_name="test_metric_constraint", scheduler_name="asha")

    def test_metric_constraint_stratum_asha(self):
        run_metric_constraint(exp_name="test_metric_constraint_stratum_asha", scheduler_name="stratum_asha")


if __name__ == '__main__':
    unittest.main()
