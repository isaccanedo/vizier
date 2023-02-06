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

"""Tests for cmaes."""
from vizier import benchmarks
from vizier._src.algorithms.designers import cmaes
from vizier._src.algorithms.testing import test_runners

from absl.testing import absltest


class CmaesTest(absltest.TestCase):

  def setUp(self):
    self.experimenter = benchmarks.BBOBExperimenterFactory('Sphere', 2)()
    super().setUp()

  def test_e2e_and_serialization(self):
    designer = cmaes.CMAESDesigner(self.experimenter.problem_statement())

    trials = test_runners.run_with_random_metrics(
        designer,
        self.experimenter.problem_statement(),
        iters=10,
        batch_size=3,
        verbose=1,
        validate_parameters=True)
    self.assertLen(trials, 30)

    new_designer = cmaes.CMAESDesigner(self.experimenter.problem_statement())
    new_designer.load(designer.dump())

    suggestions = designer.suggest(10)
    same_suggestions = new_designer.suggest(10)

    self.assertEqual(suggestions, same_suggestions)


if __name__ == '__main__':
  absltest.main()
