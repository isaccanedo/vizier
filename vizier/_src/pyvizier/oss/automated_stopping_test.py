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

"""Tests for automated_stopping."""

from vizier._src.pyvizier.oss import automated_stopping
from vizier.service import study_pb2
from vizier._src.pyvizier.oss import compare
from absl.testing import absltest


class AutomatedStoppingTest(absltest.TestCase):

  def testDefaultStoppingConfig(self):
    config = automated_stopping.AutomatedStoppingConfig.default_stopping_spec()
    proto = study_pb2.StudySpec.DefaultEarlyStoppingSpec()
    compare.assertProto2Equal(self, proto, config.to_proto())


if __name__ == '__main__':
  absltest.main()
