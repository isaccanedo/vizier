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

"""Sets up the Vizier Service gRPC server.

This should be done on a server machine:

```
python run_vizier_server.py
```

After running the command, the address of the server, formatted as:
"localhost:[PORT]" will be logged to stdout.
This address should be used as a command line argument to run_vizier_client.py
"""

import time
from typing import Sequence

from absl import app
from absl import flags
from absl import logging

from vizier.service import vizier_server

flags.DEFINE_string(
    'host',
    'localhost',
    'Host location for the server. For distributed cases, use the IP address.',
)

FLAGS = flags.FLAGS

_ONE_DAY_IN_SECONDS = 60 * 60 * 24


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  server = vizier_server.DefaultVizierServer(host=FLAGS.host)
  logging.info('Address to Vizier Server is: %s', server.endpoint)

  # prevent the main thread from exiting
  try:
    while True:
      time.sleep(_ONE_DAY_IN_SECONDS)
  except KeyboardInterrupt:
    del server


if __name__ == '__main__':
  app.run(main)
