#AI-Driven Music Success Modeling

An end-to-end machine learning project that predicts streaming performance and hit probability from Spotify + YouTube engagement signals.

This repository is structured as a practical AI workflow: data preparation, target engineering, multi-model benchmarking, threshold stress testing, and interpretable feature analysis.

## Why This Project Matters

- Builds a dual-objective prediction system:
	- Regression: estimate log-transformed stream counts.
	- Classification: identify high-impact tracks (hit vs non-hit).
- Benchmarks multiple algorithms under the same pipeline for fair comparison.
- Evaluates model behavior under stricter hit definitions using quantile-based target optimization.
- Emphasizes explainability through feature importance analysis and performance visualization.

## Core Workflow

1. Data loading and cleaning from Spotify/YouTube dataset.
2. Feature engineering and numeric preprocessing.
3. Two supervised learning tasks:
	 - `log_stream` regression
	 - `is_hit` classification
4. Model training and comparison across:
	 - Linear models
	 - Tree-based models
	 - Gradient boosting
	 - Neural networks
5. Quantile sensitivity analysis for hit-threshold robustness.
6. 5-fold cross-validation experiments at strict quantiles.
7. Model interpretation with feature importance plots.

## Models Benchmarked

- Regression:
	- Linear Regression
	- Ridge
	- Lasso
	- Decision Tree Regressor
	- Random Forest Regressor
	- Gradient Boosting Regressor
	- MLP Regressor
- Classification:
	- Logistic Regression
	- Decision Tree Classifier
	- Random Forest Classifier
	- Gradient Boosting Classifier
	- MLP Classifier

## Snapshot Results

From the saved notebook outputs, tree-based ensembles consistently perform strongest.

- Best regression model observed: Random Forest Regressor
	- R2 up to about 0.538
	- RMSE about 1.150
- Best classification model observed: Random Forest Classifier
	- Accuracy up to about 0.884
	- F1 up to about 0.675
	- ROC-AUC up to about 0.895

These results indicate that nonlinear ensemble methods capture interaction effects in music engagement features better than simple linear baselines.

## Repository Structure

- `Team_4.ipynb`: Full analysis pipeline, experiments, plots, and cross-validation runs.
- `README.md`: Project overview and usage guidance.
- `requirements.txt`: Python dependencies for reproducibility.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open `Team_4.ipynb` in Jupyter or VS Code.
4. Make sure the dataset file expected by the notebook is available locally.
5. Run all cells to reproduce training, metrics, and figures.

## Tech Stack

- Python
- NumPy, Pandas
- scikit-learn
- Matplotlib
- Jupyter Notebook

## Next Improvements

- Add notebook parameterization for data path and quantile threshold.
- Save trained models and metrics artifacts for repeatable evaluation.
- Add calibration and threshold tuning for imbalanced hit prediction.
- Convert notebook workflow into modular scripts for production-style pipelines.

## License

No license is currently defined in this repository. Add a LICENSE file to clarify reuse terms.
