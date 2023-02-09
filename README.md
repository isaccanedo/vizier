<figure>
<img src="docs/assets/vizier_logo2.png" width=20% align="right"/>
</figure>

# Open Source Vizier: Otimização de caixa preta confiável e flexível.
[![PyPI version](https://badge.fury.io/py/google-vizier.svg)](https://badge.fury.io/py/google-vizier)
![Continuous Integration (Core)](https://github.com/google/vizier/workflows/pytest_core/badge.svg)
![Continuous Integration (Clients)](https://github.com/google/vizier/workflows/pytest_clients/badge.svg)
![Continuous Integration (Algorithms)](https://github.com/google/vizier/workflows/pytest_algorithms/badge.svg)
![Continuous Integration (Benchmarks)](https://github.com/google/vizier/workflows/pytest_benchmarks/badge.svg)
![Continuous Integration (Docs)](https://github.com/google/vizier/workflows/docs/badge.svg)

  [**Google AI Blog**](https://ai.googleblog.com/2023/02/open-source-vizier-towards-reliable-and.html)
| [**Getting Started**](#getting_started)
| [**Documentation**](#documentation)
| [**Installation**](#installation)
| [**Citing Vizier**](#citing_vizier)

## O que é Vizier de código aberto (OSS)?
[OSS Vizier](https://arxiv.org/abs/2207.13676) é um serviço baseado em Python para otimização e pesquisa de caixa preta, baseado no [Google Vizier](https://dl.acm.org/doi/10.1145/3097983.3098043), um dos primeiros serviços de ajuste de hiperparâmetros projetados para funcionar em escala .
<figure>
<p align="center" width=65%>
<img src="docs/assets/oss_vizier_service.gif"/>
  <br>
  <em><b>Sistema cliente-servidor distribuído do OSS Vizier. Animação de Tom Small.</b></em>
</p>
</figure>

## Começando <a name="getting_started"></a>
Como um exemplo básico para os usuários, abaixo mostramos como ajustar um objetivo simples usando todos os tipos de espaço de pesquisa plana:

```python
from vizier.service import clients
from vizier.service import pyvizier as vz

# Objective function to maximize.
def evaluate(w: float, x: int, y: float, z: str) -> float:
  return w**2 - y**2 + x * ord(z)

# Algorithm, search space, and metrics.
study_config = vz.StudyConfig(algorithm=vz.Algorithm.GAUSSIAN_PROCESS_BANDIT)
study_config.search_space.root.add_float_param('w', 0.0, 5.0)
study_config.search_space.root.add_int_param('x', -2, 2)
study_config.search_space.root.add_discrete_param('y', [0.3, 7.2])
study_config.search_space.root.add_categorical_param('z', ['a', 'g', 'k'])
study_config.metric_information.append(vz.MetricInformation('metric_name', goal=vz.ObjectiveMetricGoal.MAXIMIZE))

# Setup client and begin optimization. Vizier Service will be implicitly created.
study = clients.Study.from_study_config(study_config, owner='my_name', study_id='example')
for i in range(10):
  suggestions = study_client.suggest(count=1)
  for suggestion in suggestions:
    params = suggestion.parameters
    objective = evaluate(params['w'], params['x'], params['y'], params['z'])
    suggestion.complete(vz.Measurement({'metric_name': objective}))
```

## Documentation <a name="documentation"></a>
OSS Vizier's interface consists of [three main APIs](https://oss-vizier.readthedocs.io/en/latest/guides/index.html):

* [**User API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-users) Allows a user to setup an OSS Vizier Server, which can host black-box optimization algorithms to serve multiple clients simultaneously in a fault-tolerant manner to tune their objective functions.
* [**Developer API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-developers) Defines abstractions and utilities for implementing new optimization algorithms for research and to be hosted in the service.
* [**Benchmarking API:**](https://oss-vizier.readthedocs.io/en/latest/guides/index.html#for-benchmarking) A wide collection of objective functions and methods to benchmark and compare algorithms.

Additionally, it contains [advanced API](https://oss-vizier.readthedocs.io/en/latest/advanced_topics/index.html) for:

* [**Tensorflow Probability:**](https://oss-vizier.readthedocs.io/en/latest/advanced_topics/index.html#tensorflow-probability) For writing Bayesian Optimization algorithms using Tensorflow Probability and Flax.
* [**PyGlove:**](https://oss-vizier.readthedocs.io/en/latest/advanced_topics/index.html#pyglove) For large-scale evolutionary experimentation and program search using OSS Vizier as a distributed backend.

Please see OSS Vizier's [ReadTheDocs documentation](https://oss-vizier.readthedocs.io/) for detailed information.

## Installation <a name="installation"></a>
**Most common:** To tune objectives using our default state-of-the-art JAX-based Bayesian Optimizer, run:

```bash
pip install google-vizier[jax]
```

To install a **minimal version** that consists of only the core service and client API from `requirements.txt`, run:

```bash
pip install google-vizier
```

For **full installation** to support all algorithms and benchmarks, run:

```bash
pip install google-vizier[extra]
```

For **specific installations**, you can run:

```bash
pip install google-vizier[X]
```

which will install additional packages from `requirements-X.txt`, such as:

* `requirements-jax.txt`: Jax libraries shared by both algorithms and benchmarks.
* `requirements-tf.txt`: Tensorflow libraries used by benchmarks.
* `requirements-algorithms.txt`: Additional repositories (e.g. EvoJAX) for algorithms.
* `requirements-benchmarks.txt`: Additional repositories (e.g. NASBENCH-201) for benchmarks.
* `requirements-test.txt`: Libraries needed for testing code.

Check if all unit tests work by running `run_tests.sh` after a full installation. OSS Vizier requires Python 3.10+, while client-only packages require Python 3.7+.

## Citing Vizier <a name="citing_vizier"></a>
If you found this code useful, please consider citing the [OSS Vizier paper](https://arxiv.org/abs/2207.13676) as well as the [Google Vizier paper](https://dl.acm.org/doi/10.1145/3097983.3098043). Thanks!

```
@inproceedings{oss_vizier,
  author    = {Xingyou Song and
               Sagi Perel and
               Chansoo Lee and
               Greg Kochanski and
               Daniel Golovin},
  title     = {Open Source Vizier: Distributed Infrastructure and API for Reliable and Flexible Black-box Optimization},
  booktitle = {Automated Machine Learning Conference, Systems Track (AutoML-Conf Systems)},
  year      = {2022},
}
@inproceedings{google_vizier,
  author    = {Daniel Golovin and
               Benjamin Solnik and
               Subhodeep Moitra and
               Greg Kochanski and
               John Karro and
               D. Sculley},
  title     = {Google Vizier: {A} Service for Black-Box Optimization},
  booktitle = {Proceedings of the 23rd {ACM} {SIGKDD} International Conference on
               Knowledge Discovery and Data Mining, Halifax, NS, Canada, August 13
               - 17, 2017},
  pages     = {1487--1495},
  publisher = {{ACM}},
  year      = {2017},
  url       = {https://doi.org/10.1145/3097983.3098043},
  doi       = {10.1145/3097983.3098043},
}
```
