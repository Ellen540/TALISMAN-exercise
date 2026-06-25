"""The TALISMAN laser-spot bench: a small simulated system to get to know.

This is the whole candidate-facing API — one class, three methods. The
simulator internals live in a separate module that ships without source;
treat the bench like real hardware and learn about it through measurements.
"""

from __future__ import annotations

import secrets
import sys
from typing import Any

import numpy as np

try:
    from . import _dynamics
except ImportError as err:  # pragma: no cover - Python version mismatch
    raise ImportError(
        "Could not load the simulator module (talisman/_dynamics). The handout "
        "is compiled for one specific Python minor version (see the README); "
        f"you are running Python {sys.version_info.major}.{sys.version_info.minor}."
    ) from err


class LaserEnv:
    """A 2D laser spot on a noisy camera, drifting over time.

    The interface follows the Gymnasium convention (without depending on it):

        >>> env = LaserEnv(token="your-personal-token")
        >>> obs, info = env.reset()
        >>> obs, reward, terminated, truncated, info = env.step(np.zeros(2))

    Observations
        ``obs["image"]``    a (64, 64) float array: one noisy camera frame.
                            Positions are (x, y) in pixels with
                            ``image[iy, ix]``; plot with ``origin="lower"``.
        ``obs["sensors"]``  a dict of five named telemetry readings. Whether
                            any of them relates to the spot is for you to
                            find out.

    Actions
        A length-2 array of steering commands, one per channel, each in
        [-1, 1] (values outside are clipped — the steering range is limited).
        ``np.zeros(2)`` is always valid: passive observation is a perfectly
        good experiment. How the two channels move the spot is for you to
        measure.

    The reward is always 0.0 and ``terminated``/``truncated`` are always
    False — this exercise is not reward-driven and episodes never end, so
    you can record time series for as long as you like. Every ``info``
    carries the step counter ``t`` and the fixed target position
    ``target``.

    Determinism: a given token always builds the same bench, and ``reset()``
    rewinds to the exact same session — same drift, same noise, step for
    step — so your notebook is exactly reproducible. If you want a fresh,
    statistically independent session on the same bench, pass any integer:
    ``env.reset(seed=1)``.

    The oracle flag: ``LaserEnv(token=..., oracle=True)`` adds
    ``info["true_centroid"]`` to every observation. It exists for validating
    your own estimates only. On real hardware this signal does not exist;
    your conclusions should stand without it, and the follow-up discussion
    will assume oracle-free analysis.
    """

    ACTION_DIM = 2
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    def __init__(
        self,
        token: str | None = None,
        *,
        seed: int | None = None,
        oracle: bool = False,
    ) -> None:
        """Create your instance of the bench.

        Args:
            token: your personal instance token (a short hex string from the
                exercise invitation). Every token is its own bench, with its
                own parameters and sensor wiring. If you pass nothing, a
                random throwaway bench is created (its token is in
                ``env.token``).
            seed: an integer alternative to ``token``, mainly for tests.
                Pass one or the other, not both.
            oracle: if True, ``info["true_centroid"]`` is included with every
                observation — for validating your own estimates only.
        """
        if token is not None and seed is not None:
            raise ValueError("Pass either token= or seed=, not both.")
        if token is None and seed is None:
            token = secrets.token_hex(5)
        if token is not None:
            master = _dynamics.master_seed_from_token(token)
        else:
            master = int(seed) & (2**63 - 1)
        self._token = token
        self._master = master
        self._oracle = bool(oracle)
        instance_ss, process_ss = _dynamics.seed_sequences(master)
        self._process_ss = process_ss
        self._params = _dynamics.InstanceParams(np.random.default_rng(instance_ss))
        self._model = _dynamics.DynamicsModel(self._params)
        self._state: _dynamics.BenchState | None = None
        self._rng: np.random.Generator | None = None
        self._last_image: np.ndarray | None = None

    # -- Gym-style API -----------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict, dict]:
        """Start (or rewind) a session on your bench.

        Returns ``(obs, info)``; ``info["target"]`` is the fixed setpoint the
        spot should ideally sit on (it is included in every step's ``info``
        too). Without arguments this rewinds to the exact same session every
        time (reproducible notebooks); with ``seed=<any int>`` you get an
        independent session on the same bench.
        """
        if seed is None:
            self._rng = np.random.default_rng(self._process_ss)
        else:
            session = np.random.SeedSequence([self._master, int(seed) & (2**63 - 1)])
            self._rng = np.random.default_rng(session)
        self._state = self._model.initial_state(self._rng)
        return self._observation()

    def step(self, action: Any) -> tuple[dict, float, bool, bool, dict]:
        """Apply one steering command and advance the bench by one time step.

        Args:
            action: array-like of shape (2,), e.g. ``np.array([0.1, -0.2])``
                or ``np.zeros(2)``. Values are clipped to [-1, 1].

        Returns:
            ``(obs, reward, terminated, truncated, info)``. The reward is
            always 0.0 (ignore it), and the episode never ends.
        """
        if self._state is None or self._rng is None:
            raise RuntimeError("Call env.reset() once before env.step().")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (2,):
            raise ValueError(
                f"action must have shape (2,), e.g. np.array([0.1, -0.2]); "
                f"got shape {a.shape}."
            )
        if not np.all(np.isfinite(a)):
            raise ValueError("action contains NaN or inf.")
        self._model.step(self._state, self._rng, a)
        obs, info = self._observation()
        return obs, 0.0, False, False, info

    def render(self) -> None:
        """Show the most recent camera frame (needs matplotlib)."""
        if self._last_image is None:
            raise RuntimeError("Nothing to render yet — call env.reset() first.")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 4.2))
        im = ax.imshow(self._last_image, origin="lower", cmap="magma")
        fig.colorbar(im, ax=ax, label="counts")
        ax.plot(
            *self._params.target,
            "c+",
            markersize=14,
            markeredgewidth=2,
            label="target",
        )
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.legend()
        plt.show()

    # -- conveniences --------------------------------------------------------

    @property
    def token(self) -> str | None:
        """The instance token this bench was built from (None if seeded)."""
        return self._token

    @property
    def target(self) -> np.ndarray:
        """The fixed setpoint (x, y) in pixels — same value as in reset info."""
        return self._params.target.copy()

    @property
    def sensor_names(self) -> tuple[str, ...]:
        """Names of the five telemetry channels, in observation order."""
        return tuple(_dynamics.SENSOR_NAMES)

    @property
    def image_shape(self) -> tuple[int, int]:
        """Shape of the camera frame, (rows, cols) = (y, x)."""
        return _dynamics.GRID_SHAPE

    def __repr__(self) -> str:
        src = f"token={self._token!r}" if self._token is not None else f"seed={self._master}"
        return f"LaserEnv({src}, oracle={self._oracle})"

    # -- internals -----------------------------------------------------------

    def _observation(self) -> tuple[dict, dict]:
        assert self._state is not None and self._rng is not None
        image, sensors = self._model.observe(self._state, self._rng)
        self._last_image = image
        obs = {"image": image, "sensors": sensors}
        info: dict[str, Any] = {
            "t": self._state.t,
            "target": self._params.target.copy(),
        }
        if self._oracle:
            info["true_centroid"] = self._state.truth["centroid"].copy()
        return obs, info
