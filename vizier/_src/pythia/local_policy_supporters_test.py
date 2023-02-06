# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

"""Tests for vizier.pythia.base.local_policy_supporters."""

import numpy as np

from vizier import pyvizier as vz
from vizier._src.pythia import local_policy_supporters
from absl.testing import absltest
from absl.testing import parameterized


InRamPolicySupporter = local_policy_supporters.InRamPolicySupporter
_GUID = '31'


def _runner_with_10trials():
  runner = InRamPolicySupporter(vz.ProblemStatement(), study_guid=_GUID)
  runner.AddTrials([vz.Trial() for _ in range(10)])
  return runner


class LocalPolicySupportersTest(parameterized.TestCase):

  def test_add_and_get_trials(self):
    runner = _runner_with_10trials()
    trials = runner.GetTrials()
    self.assertLen(trials, 10)
    # The 10 trials are assigned ids 1 through 10 automatically.
    self.assertSequenceEqual([t.id for t in trials], range(1, 11))

  def test_push_metadata(self):
    runner = _runner_with_10trials()
    trial1 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]

    mu = vz.MetadataDelta()
    mu.assign('ns', 'key', 'value')
    mu.assign('ns', 'key', 'value', trial_id=1)
    # Metadata update is not immediate.
    self.assertEmpty(runner.GetStudyConfig(study_guid=_GUID).metadata.ns('ns'))
    self.assertEmpty(trial1.metadata)
    runner._UpdateMetadata(mu)

    self.assertEqual(
        runner.GetStudyConfig(study_guid=_GUID).metadata.ns('ns').get('key'),
        'value')
    trial0 = runner.GetTrials(min_trial_id=1, max_trial_id=1)[0]
    self.assertSequenceEqual(trial0.metadata.ns('ns'), {'key': 'value'})

  # TODO: Need a test for LocalPolicySupporter.SuggestTrials().


class LocalPolicySupportersGetBestTrialsTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(goal=vz.ObjectiveMetricGoal.MAXIMIZE, count=2, best_values=[9, 8]),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, count=2, best_values=[0, 1]),
      dict(goal=vz.ObjectiveMetricGoal.MINIMIZE, count=None, best_values=[
          0,
      ]))
  def test_get_best_trials_single_objective(self, goal, count, best_values):
    runner = InRamPolicySupporter(
        vz.ProblemStatement(
            vz.SearchSpace(),
            metric_information=vz.MetricsConfig(
                [vz.MetricInformation(name='objective', goal=goal)])))

    trials = [
        vz.Trial().complete(vz.Measurement({'objective': i})) for i in range(10)
    ]
    np.random.shuffle(trials)
    runner.AddTrials(trials)

    objectives = np.array([
        t.final_measurement.metrics['objective'].value
        for t in runner.GetBestTrials(count=count)
    ])
    np.testing.assert_array_equal(objectives, np.array(best_values))

  @parameterized.parameters(
      dict(
          sin_goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          cos_goal=vz.ObjectiveMetricGoal.MAXIMIZE,
          best_r=5),
      dict(
          sin_goal=vz.ObjectiveMetricGoal.MINIMIZE,
          cos_goal=vz.ObjectiveMetricGoal.MINIMIZE,
          best_r=1))
  def test_get_best_trials_multi_objective(self, sin_goal, cos_goal, best_r):
    runner = InRamPolicySupporter(
        vz.ProblemStatement(
            vz.SearchSpace(),
            metric_information=vz.MetricsConfig([
                vz.MetricInformation(name='sin', goal=sin_goal),
                vz.MetricInformation(name='cos', goal=cos_goal)
            ])))

    def build_measurement(r, theta):
      return vz.Measurement({
          'r': r,
          'theta': theta,
          'cos': r * np.cos(theta),
          'sin': r * np.sin(theta)
      })

    # Generate many trials for each radius.
    for r in range(1, 6):
      for theta in np.linspace(0, np.pi / 2, 5):
        runner.AddTrials([vz.Trial().complete(build_measurement(r, theta))])

    # Check the radius.
    rs = np.array([
        t.final_measurement.metrics['r'].value for t in runner.GetBestTrials()
    ])
    self.assertEqual(rs.size, 5)
    np.testing.assert_array_equal(rs, np.ones_like(rs) * best_r)


if __name__ == '__main__':
  absltest.main()
