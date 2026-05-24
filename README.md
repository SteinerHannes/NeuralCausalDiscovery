# NeuralCausalDiscovery

This repository contains the experimental pipeline for the master's thesis
"Toward Causal Explainability of Neural Networks: Recovering and Validating Global Causal Structure".

The project studies whether causal discovery on neural-network representations
can recover causal structure and interventional behaviour from synthetic
structural equation models (SEMs). The pipeline generates SEM data, trains small
feed-forward neural networks, extracts internal representations, estimates PAGs
with causal discovery, compares recovered structures, and evaluates
interventional fidelity.

## Repository Structure

- `pipeline.py`: main orchestration entrypoint
- `train.py`, `test.py`: model training and held-out evaluation
- `extraction.py`: representation extraction and optional alignment
- `causaldiscovery.py`: data-level and model-level causal discovery
- `pag_cleanup.py`: combined PAG aggregation and cleanup
- `interventions.py`: SEM-level and model-level intervention analyses
- `reporting.py`: final figure, table, statistics, and manifest generation
- `config/`: Hydra configuration files for datasets, models, experiments, and
  pipeline steps
- `reporting/figures/`: report figure and table builders
- `data/`: synthetic SEM datasets and intervention-ready dataset interfaces
- `models/`: MLP model definitions
- `metrics/`: structural and performance metrics
- `experiment_utils/`: shared utilities for reproducibility, logging, alignment,
  CI tests, and artifact reuse
- `outputs/experiments/`: experiment outputs, fold-level results, metrics,
  intervention results, and graph artifacts
- `outputs/reporting/thesis_reports/`: final figures, tables, statistics, and
  report manifests

## Python Setup

Python 3.9 is recommended. The package versions used for the experiments are
listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some dependencies, especially `pygraphviz`, may require Graphviz system
libraries. On macOS with Homebrew:

```bash
brew install graphviz
```

## Adding causality-lab from the Command Line

The pipeline uses IntelLabs `causality-lab`. Clone it next to this repository and
add both repositories to `PYTHONPATH`.

```bash
cd ..
git clone https://github.com/SteinerHannes/causality-lab.git
cd NeuralCausalDiscovery
export PYTHONPATH="$(pwd):$(cd ../causality-lab && pwd):$PYTHONPATH"
```

For a one-off command, the same path setup can be prepended directly:

```bash
PYTHONPATH="$(pwd):$(cd ../causality-lab && pwd):$PYTHONPATH" \
  python pipeline.py --config-name experiments/linear_sem_shallow_mlp_align_observable_state
```

To make the setup persistent for the current shell profile, add the export line
to `~/.zshrc` or `~/.bashrc`, adjusting the paths if the repositories are stored
elsewhere.

## Running Experiments

Experiment profiles are defined in `config/experiments/`. A full pipeline run can
be started with Hydra:

```bash
python pipeline.py --config-name experiments/linear_sem_shallow_mlp_align_observable_state
```

Pipeline steps are controlled in the selected experiment config and in
`config/pipeline/Default.yaml`. New runs write their results to
`outputs/experiments/<experiment.name>/`.

## Generating Reporting Artifacts

The reporting entrypoint builds the thesis figures, summary tables, statistics,
and JSON manifests from the experiment outputs:

```bash
python reporting.py --config-name reporting_config
```

The output directory is configured in `config/reporting_config.yaml`.

## Final Artifacts

The final result artifacts are available under:

- `outputs/experiments/`: model checkpoints, fold-level CSV/JSON results,
  learned PAGs, structural metrics, and intervention summaries
- `outputs/reporting/thesis_reports/figures/`: final thesis figures as PNG/PDF
  files with corresponding statistics and table CSV files
- `outputs/reporting/thesis_reports/figure_manifest.json`: manifest of the final
  reporting artifacts

These artifacts provide the structural-accuracy and interventional-fidelity
evidence used by the thesis analysis.
