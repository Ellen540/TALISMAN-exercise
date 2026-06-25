# TALISMAN: Take-home exercise

## Overview
This exercise gives you a small simulated laser system: a spot on a noisy camera image that drifts around, plus two actuators you can engage. We ask you to get to know it the way you would any new system: measure it, work out what it does, and tell us what you found. There is no single right answer and no hidden tricks. We are interested in how you approach a system you have not seen before: what you measure, what you conclude, and how you back it up.

**Time budget: about 3 hours.** This should be enough for the four core tasks plus a short investigation log and a few-sentence control proposal. Each core task asks for an evidence-backed result, not an exhaustive treatment. Rough plots with bullet-point reasoning are exactly right. The exercise is also part of the follow-up conversation, so think working notes rather than a polished report. There are optional extras at the end. We do not expect them, but feel free to investigate further if you have time left from your 3-h budget.

## The system

At each time step the test-bench gives you:

- **one camera image**: a 64×64 px frame with a bright spot and some noise;
- **five telemetry channels** (`temp_a`, `temp_b`, `flow`, `vib`, `pd_current`). They are all scalar readings from around the bench. The names are just labels, like unfamiliar tags on real hardware. Let the data, not the name, tell you what each measures and whether any relates to the laser spot position.

You can command **two actuator channels**, which move the spot in some way. Again, for you to determine how they do so. There is a fixed **target position** where the laser spot should sit. As you will see, left alone, the spot does not stay put over time.

**Note:** The simulator internals are not provided in source form, and the system parameters are randomised per instance. Your token builds *your* bench. Treat it like a real test bench: learn it through measurement, and show the evidence.

## Setup

You need **Python 3.12** (the simulator is compiled for this exact minor version). If your default `python` is a different version, get a 3.12 interpreter with `pyenv` or `conda`, or call `python3.12` explicitly below.

```bash
cd talisman-exercise

# On Linux / macOS
python3.12 -m venv .venv && source .venv/bin/activate

# Windows (PowerShell): py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
# Windows (CMD):        py -3.12 -m venv .venv && .venv\Scripts\activate.bat

pip install -e ".[notebook]"        # package + Jupyter. For no Jupyter, use: pip install -e .
jupyter lab getting_started.ipynb   # or open getting_started.ipynb in VS Code (no Jupyter needed)
```

**Paste your token** (from the invitation email) into the notebook in place of `YOUR_TOKEN`, and you're ready to go.

> ⚠️ **Use your own token.** Any string builds a valid bench, so if you leave the placeholder in, you would silently work on the wrong one.

## The bench in a nutshell

The interface follows the [Farama Gymnasium](https://gymnasium.farama.org/) convention (but the Gymnasium library is not needed):

```python
import numpy as np
from talisman import LaserEnv

env = LaserEnv(token="YOUR_TOKEN")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(np.zeros(2))

obs["image"]    # (64, 64) float array: the noisy camera frame
obs["sensors"]  # dict of five named floats: the five telemetry sensors
```

Worth knowing:

- **Actions** are length-2 arrays in [-1, 1], one per actuator channel (values outside are clipped). `np.zeros(2)` is always valid. Just watching is a fine experiment, and episodes never end, so record as long as you like, or think necessary.
- **An action command is a held setpoint**, i.e. *not* a one-shot nudge: the spot moves towards the action value you send and holds only while you keep sending that action; stop, and it eases back.
- **Ignore the reward** (always 0.0): this exercise is not reward-driven.
- **Positions** are (x, y) in pixels, but arrays index row-first, so the spot at (x, y) is `image[y, x]`. Plot frames with `origin="lower"` so y points up and markers you overlay at (x, y) land on the matching pixel (with matplotlib's default origin the image is drawn upside-down relative to your points). `env.render()` does this for you, and the notebook's first cells show the few lines of matplotlib.
- **Reproducibility:** a token always builds the same bench, and `reset()` replays the exact same session, so your notebook reruns identically. For an independent session on the same bench, you can pass a seed: `env.reset(seed=1)`.
- **Ground-truth flag:** `LaserEnv(token=..., oracle=True)` adds `info["true_centroid"]`. You can use this to validate your own estimates only. Since real hardware has no such signal, your conclusions should stand without it.
- The **getting-started notebook** has a **`log_run(env, n, action=...)`** cell: a small visible loop that steps the bench n times, sending one fixed `action` on every step (default zero = passive watching), and collects arrays. It's yours to edit (swap in your own estimator, keep the raw frames, or send a time-varying command).

## Your task

We have included **`getting_started.ipynb`** to get you going. Its opening cells walk through the whole workflow (build the bench, read a camera frame and the five sensors, send an actuator command, estimate the spot position, and log a run with a template for-loop), and you can copy from them freely. Everything below the horizontal divider is yours. Work in whatever form suits you: build straight in the notebook, start a fresh one, or use markdown plus Python scripts. Document as you go, and note dead ends, too. Depth on the core items beats shallow coverage of everything.

### Core

1. **Measure the laser spot position well:** The `crude_centroid` in the getting-started notebook is rough on purpose. Go beyond it, and put a number on *how precisely you know the position* in a single frame.
2. **Watch how the laser spot moves:** Leave the system alone (zero actions) for a while and describe what you observe about the spot's motion. Back what you claim with evidence.
3. **Find out what the actuators do:** Send deliberate test commands and quantify how each channel moves the laser spot: how far, which direction, how fast.
4. **Look at the telemetry:** How are the five channels related to the laser spot's motion, and to each other? Which carry real information about it, and how do you tell the rest apart?

### The investigation log

Keep a short chronological log: **5–10 bullets of "what I tried, in order", dead ends included.** A few lines, not a report. It will help us follow the path and serve as a basis for the follow-up.

### Control design proposal *(words, not code)*

Given what you found, how would you keep the spot on the target position? Provide a few sentences. A sketch is welcome, but an implementation is not required. Worth covering: what would you feed back on? Are any of the five measured telemetry signals worth adding and why / why not? How would you handle sudden laser spot displacements? What would you expect to go wrong on real hardware? Also tell us what you considered and decided *not* to do, and why.

### Optional *(only if time genuinely remains; not expected)*

A clean treatment of the four core items is the heart of the submission. We would much rather see a few things done well than everything done shallowly. If you are curious and have time left over, any of these are welcome: pick what interests you.

- **What effects limit the position reading on a single camera frame?** Which dominate, and how would you check?
- **What could *systematically* bias your actuator analysis in Task 3?** What effects could push the values consistently off?
- **Push the telemetry analysis further:** Task 4 was a first look at how the telemetry channels relate to the spot's motion and to each other. Beyond a static analysis, what would a more careful one need to account for?
- **What surprised you, and what would you look at next with more time?**
- **A first cut at a controller** that keeps the spot on target. Only if the core is completed and you are genuinely curious: a strong investigation without a controller beats a controller without the investigation. (This is the *implementation* part; the no-code, written control proposal above is the part we actually require.)

## How we read your work

Methods are open, you can use whatever you like. We look at:

1. **Investigation quality:** Hypotheses, deliberate experiments, evidence-backed conclusions, honest treatment of uncertainty and dead ends.
2. **Correctness and rigour**
3. **Right-sizing:** The simplest approach that answers the question. Choosing *not* to use a sophisticated method, and saying why, is a plus.
4. **Communication:** Can we follow your reasoning from your submission alone?
5. **Control proposal:** does it follow from your findings, with trade-offs and failure modes considered?
6. **Code quality:** Clear, simple analysis code. We do not expect production software.

## AI tools

You can use AI tools as you would in real work. It does not count against you. We ask out of interest in how you work:

1. add a sentence or two on how you used them, and
2. you must be able to explain and defend everything you submit.

## The follow-up

The take-home serves as a starting point for our conversation; it is not a final exam. It needn't be exhaustive or polished. After you submit we will have a ~30–40 minute Zoom call: mostly a two-way conversation about your background, your questions, and the project and team, with part of it spent on the exercise, where we walk through what you found and reason through a small new question or two about your system.

## Submitting

Send your submission to michael.schenk@cern.ch, as a zip or a repo link, including:

- your notebook(s) and/or scripts (make sure they run top-to-bottom),
- the investigation log,
- the control design proposal,
- the sentence or two on AI use.

**The installation has been tested on macOS, Linux, and Windows (PowerShell). Still, if anything is broken or unclear about the package, or you have trouble installing, please do not hesitate to ask: michael.schenk@cern.ch. We don't want you to lose time on setup.**
