# Counterfactual Explainer

This module provides tools to explore, visualize, and explain predictions from the Spatio-Temporal Object Tracking (STOT) model. It is organized into **five main parts**: a core explanation engine and four Streamlit-based apps for browsing logs, viewing explanations, editing scene graphs, and finding interesting scenarios.

---

## Overview

| Part | Purpose |
|------|--------|
| **Explainer** | Core explanation module (inference, movement tracking, counterfactual perturbation). |
| **Explanation Visualizer** | Browse scene graphs by day and get explanations for each predicted action. |
| **Distraction Adder** | Edit the scene graph and run the model to get new predictions and explanations. |
| **Proactive Action GT Summarizer** | Browse **ground truth** actions to find specific moments (e.g., “cooking pot moved to kitchen counter”). |
| **Proactive Action Summarizer** | Browse **predicted** actions to find moments of interest from the model’s output. |

---

## 1. Explainer (core)

**Location:** `explainer/`

The core of the counterfactual explainer. It runs the STOT model on routines, tracks object movements, and performs perturbation-based counterfactual analysis to explain why the model predicted certain actions.

**Key components:**

- **`explainer.py`** — Main `Explainer` class: loads data, runs inference, and runs the counterfactual analysis loop.
- **`stot_model.py`** — Wraps the STOT model for inference.
- **`movement_tracker.py`** — Tracks object movements across time steps.
- **`perturbation.py`** — Perturbation engine for counterfactual tests.
- **`data_utils.py`** — Dataset and routine loading.

The explanation visualizer and distraction adder use this module to generate explanations.

---

## 2. Explanation Visualizer

**Location:** `explantion_visualizer/`

The **main visualizer** for exploring already-computed logs. Use it to:

- View the **scene graph** for different days.
- **Scroll** through routines and time steps.
- Get **explanations** for each specific predicted action (mechanistic/counterfactual).

**Run (from project root):**

```bash
streamlit run counterfactual_explainer/explantion_visualizer/app.py
```

Alternative app (if available):

```bash
streamlit run counterfactual_explainer/explantion_visualizer/app_v2.py
```

**Requirements:** Logs produced by the core explainer (e.g. under `explantion_visualizer/../logs` or as configured in `log_handler.py`), and HOMER data under `data/HOMER/` for node labels/context.

---

## 3. Distraction Adder

**Location:** `distraction_adder/`

Similar to the explanation visualizer, but with **editing** and **re-prediction**:

- **Edit the scene graph** (e.g. move objects, add/remove nodes or edges).
- **Run the STOT model** on the edited graph to get new predictions.
- **Generate explanations** for those predictions.

Useful for “what-if” analysis and testing how the model reacts to manual changes.

**Run (from project root):**

```bash
streamlit run counterfactual_explainer/distraction_adder/app.py
```

**Requirements:** HOMER dataset (e.g. `data/HOMER/household<N>`) and a trained STOT model/checkpoint as expected by `stot_model_runner.py`.

---

## 4. Proactive Action GT Summarizer

**Location:** `proactive_action_gt_summarizer/`

For **finding interesting scenarios by ground truth actions**. It shows a list of **ground truth** actions that actually happened (e.g. “cooking pot moved to kitchen counter”). Use it when you care about **what really happened** and want to jump to that moment in the data.

- Browse CSVs of processed logs that summarize **ground truth** movements.
- Filter/sort by object, location, time, etc.
- Use the list to find the exact moment you care about, then inspect it in the explanation visualizer or distraction adder if needed.

**Run (from project root):**

```bash
streamlit run counterfactual_explainer/proactive_action_gt_summarizer/app.py
```

**Preprocessing:** The app reads from `counterfactual_explainer/processed_logs_gt/diff_steps/step_size_<N>/`. To generate these CSVs, run the log processor (e.g. from the `proactive_action_gt_summarizer` directory):

```bash
python counterfactual_explainer/proactive_action_gt_summarizer/main.py
```

Adjust `household_id` and range in `main.py` as needed.

---

## 5. Proactive Action Summarizer

**Location:** `proactive_action_summarizer/`

For **finding interesting scenarios by predicted actions**. It shows a list of **model-predicted** actions. Use it when you are interested in **what the model predicted** (e.g. a specific predicted movement) and want to find and inspect those moments.

- Browse CSVs of processed logs that summarize **predicted** movements.
- Filter/sort to find the prediction you care about.
- Then open that routine/day in the explanation visualizer or distraction adder for detailed explanations.

**Run (from project root):**

```bash
streamlit run counterfactual_explainer/proactive_action_summarizer/app.py
```

**Preprocessing:** The app reads from `counterfactual_explainer/processed_logs/`. Generate these CSVs using the log processor:

```bash
python counterfactual_explainer/proactive_action_summarizer/main.py
```

Adjust `household_id` and range in `main.py` as needed.

---

## Quick reference: which app when?

| Goal | App |
|------|-----|
| See scene graph and explanations for existing predictions | **Explanation Visualizer** |
| Edit scene graph and get new predictions + explanations | **Distraction Adder** |
| Find a moment where a **ground truth** action happened | **Proactive Action GT Summarizer** |
| Find a moment where a **predicted** action occurred | **Proactive Action Summarizer** |

---

## Setup

<!-- Use the same environment as the main STOT project (see root [README.md](../README.md)):

```bash
conda create --name <env-name> --file requirements_conda.txt
conda activate <env-name>
pip install -r requirements_pip.txt
pip install streamlit protobuf==3.20.* pyvis
``` -->
TODO: Write setup instructions and test.
TODO: Command run order.

Run all Streamlit commands from the **repository root** (`ExplainableSpatioTemporalObjectTracking/`) so that imports for `helpers`, `data`, and the STOT model resolve correctly.

---

## Directory layout

```
counterfactual_explainer/
├── explainer/                    # Core explanation module
├── explantion_visualizer/        # Scene graph + explanation browser
├── distraction_adder/            # Scene graph editor + predictions + explanations
├── proactive_action_gt_summarizer/   # Ground-truth action list (find GT moments)
├── proactive_action_summarizer/     # Predicted action list (find prediction moments)
├── util.py
├── main.py
└── README.md
```

---

## Citation

If you use this explainer with the STOT model, please cite the main paper:

```bibtex
@inproceedings{patel2022proactive,
  title={Proactive Robot Assistance via Spatio-Temporal Object Modeling},
  author={Patel, Maithili and Chernova, Sonia},
  booktitle={6th Annual Conference on Robot Learning},
  year={2022}
}
```

and 

```
explanation paper: coming soon ...
```
