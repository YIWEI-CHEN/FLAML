import logging
from typing import Dict, Optional, Union, List, Tuple

import numpy as np
from ray.tune.schedulers import AsyncHyperBandScheduler, TrialScheduler
from ray.tune.schedulers.async_hyperband import _Bracket
from ray.tune.experiment import Trial

logger = logging.getLogger(__name__)


class StratumAsyncHyperBandScheduler(AsyncHyperBandScheduler):
    def __init__(self,
                 time_attr: str = "training_iteration",
                 metric: Optional[str] = None,
                 mode: Optional[str] = None,
                 max_t: int = 100,
                 grace_period: int = 1,
                 reduction_factor: float = 4,
                 brackets: int = 1,
                 stop_last_trials: bool = True,
                 metric_constraints: Optional[List[Tuple[str, str, float]]] = None):
        super(StratumAsyncHyperBandScheduler, self).__init__(time_attr, metric, mode, max_t, grace_period,
                                                             reduction_factor, brackets, stop_last_trials)
        self._metric_constraints = metric_constraints
        if metric_constraints:
            assert all(x[1] in ["<=", ">="] for x in metric_constraints), "sign of metric constraints must be <= or >=."
        self._brackets = [
            ConstraintBracket(grace_period, max_t, reduction_factor, s, stop_last_trials=stop_last_trials)
            for s in range(brackets)
        ]

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        action = TrialScheduler.CONTINUE
        if self._time_attr not in result or self._metric not in result:
            return action
        if result[self._time_attr] >= self._max_t and self._stop_last_trials:
            action = TrialScheduler.STOP
        else:
            bracket = self._trial_info[trial.trial_id]
            cur_violation_amount = self.get_violation_amount(result)
            action = bracket.on_result(
                trial, result[self._time_attr], self._metric_op * result[self._metric], cur_violation_amount
            )
        if action == TrialScheduler.STOP:
            self._num_stopped += 1
        return action

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        if self._time_attr not in result or self._metric not in result:
            return
        bracket = self._trial_info[trial.trial_id]
        cur_violation_amount = self.get_violation_amount(result)
        bracket.on_result(
            trial, result[self._time_attr], self._metric_op * result[self._metric], cur_violation_amount
        )
        del self._trial_info[trial.trial_id]

    def get_violation_amount(self, result):
        if self._metric_constraints:
            if all(result.get(metric) is None for metric, _, _ in self._metric_constraints):
                return None
            violation_amount = 0
            for metric, sign, threshold in self._metric_constraints:
                value = result.get(metric)
                if value is not None:
                    if sign == "<=" and value > threshold:
                        violation_amount += value - threshold
                    elif sign == ">=" and value < threshold:
                        violation_amount += threshold - value
            return violation_amount
        else:
            return None

    def debug_string(self) -> str:
        out = "Using StratumAsyncHyperBand: num_stopped={}".format(self._num_stopped)
        out += "\n" + "\n".join([b.debug_str() for b in self._brackets])
        return out


class ConstraintBracket(_Bracket):
    def on_result(self, trial: Trial, cur_iter: int, cur_rew: Optional[float],
                  cur_violation_amount: Optional[float]) -> str:
        action = TrialScheduler.CONTINUE
        for milestone, recorded in self._rungs:
            if (
                cur_iter >= milestone
                and trial.trial_id in recorded
                and not self._stop_last_trials
            ):
                # If our result has been recorded for this trial already, the
                # decision to continue training has already been made. Thus we can
                # skip new cutoff calculation and just continue training.
                # We can also break as milestones are descending.
                break
            if cur_iter < milestone or trial.trial_id in recorded:
                continue
            else:
                selected_recorded = self.filter_by_violation(recorded, violation_amount=cur_violation_amount)
                if cur_violation_amount is None or cur_violation_amount <= 0:
                    cutoff = self.cutoff(selected_recorded)
                    if cutoff is not None and cur_rew < cutoff:
                        action = TrialScheduler.STOP
                else:
                    violation_recorded = dict([(k, -v[1]) for k, v in selected_recorded.items()])
                    violation_cutoff = self.cutoff(violation_recorded, q=(1 - 1 / self.rf * 2) * 100)
                    if violation_cutoff is not None:
                        violation_cutoff *= -1
                        small_violation_recorded = dict([(k, v[0]) for k, v in selected_recorded.items()
                                                         if v[1] <= violation_cutoff])
                        reward_cutoff = self.cutoff(small_violation_recorded, q=(1 - 1 / self.rf * 2) * 100)
                        if cur_violation_amount > violation_cutoff or \
                                (reward_cutoff is not None and cur_rew < reward_cutoff):
                            action = TrialScheduler.STOP
                if cur_rew is None:
                    logger.warning("Reward attribute is None! Consider"
                                   " reporting using a different field.")
                else:
                    recorded[trial.trial_id] = (cur_rew, cur_violation_amount)
                break
        return action

    def filter_by_violation(self, recorded, violation_amount):
        if violation_amount is None:
            return dict([(k, v[0]) for k, v in recorded.items() if v[1] is None])
        elif violation_amount > 0:
            return dict([(k, v) for k, v in recorded.items() if v[1] is not None and v[1] > 0])
        else:
            return dict([(k, v[0]) for k, v in recorded.items() if v[1] is not None and v[1] <= 0])

    def cutoff(self, recorded, q=None) -> Union[None, int, float, complex, np.ndarray]:
        if q is None:
            q = (1 - 1 / self.rf) * 100
        if not recorded:
            return None
        return np.nanpercentile(list(recorded.values()), q)

    def debug_str(self) -> str:
        iters = " | ".join([
            "Iter {:.3f}: {}".format(milestone, self.cutoff(
                dict([(k, v[0]) for k, v in recorded.items()])
            ))
            for milestone, recorded in self._rungs
        ])
        return "Bracket: " + iters
