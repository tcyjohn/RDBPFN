# Temporal vocabulary for SCM timestamp generation.
# Calendar-aligned Fourier seasonality + EventCalendar (ForecastPFN-inspired).
# DOW patterns via mixture of Dirichlets — structural templates, not ad-hoc bias.

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import random
import math

DEFAULT_NUM_POINTS = 20000
NUM_DAYS = 18262  # 50 years: 1970-01-01 to 2020-12-31

# Mixture of Dirichlet templates for day-of-week patterns.
# Each entry: (α vector, weight).  α controls concentration per DOW.
# Smaller α → that day suppressed.  Larger α → that day boosted.
DOW_TEMPLATES: List[Tuple[List[float], float]] = [
    ([3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0], 0.28),   # near-uniform
    ([3.0, 3.0, 3.0, 3.0, 3.0, 0.3, 0.3], 0.20),   # weekday-heavy (Mon-Fri)
    ([0.3, 0.3, 0.3, 0.3, 0.3, 3.0, 3.0], 0.13),   # weekend-heavy (Sat-Sun)
    ([3.0, 3.0, 3.0, 3.0, 0.3, 0.3, 0.3], 0.10),   # Mon-Thu heavy
    ([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 5.0], 0.07),   # Sun peak
    ([5.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], 0.05),   # Mon peak
    ([0.2, 0.2, 0.2, 0.2, 0.2, 5.0, 0.2], 0.05),   # Sat peak
    ([0.2, 0.2, 5.0, 0.2, 0.2, 0.2, 0.2], 0.04),   # Wed peak
    ([8.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 0.03),   # extreme Mon peak
    ([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 8.0], 0.03),   # extreme Sun peak
    ([0.1, 0.1, 0.1, 0.1, 0.1, 8.0, 0.1], 0.02),   # extreme Sat peak
]


def _sample_dow_pvec(seed: int, device: str = "cpu") -> torch.Tensor:
    """Sample a DOW probability vector from the Dirichlet mixture.

    Returns (7,) tensor, sum=1, representing p(Mon), ..., p(Sun).
    """
    rng = np.random.RandomState(seed)
    templates, weights = zip(*DOW_TEMPLATES)
    weights = np.array(weights) / sum(weights)
    idx = rng.choice(len(templates), p=weights)
    alpha_base = np.array(templates[idx])

    # Per-table concentration noise: multiply α by LogNormal to vary sharpness.
    # Higher std → more tables land at extreme entropy.
    concentration = float(np.exp(rng.normal(0.0, 0.7)))
    alpha = alpha_base * concentration

    p = rng.dirichlet(alpha)
    return torch.tensor(p, device=device, dtype=torch.float32)


class EventCalendar:
    """RDB-level shared event calendar.

    Each event h:
      base_doy_h   ~ U(0, 365)
      sigma_h      ~ U(1.0, 3.0) days
      importance_h ~ clip(LogNormal(-0.2, 0.5), 0.2, 2.5)

    Per-table sensitivity:
      active_r     ~ Bernoulli(0.4)
      local_sens_h ~ Beta(1, 4)
      sens_h       = active_r * local_sens_h
    """

    def __init__(self, seed: int, H: int = None, device: str = "cpu"):
        self.device = device
        rng = np.random.RandomState(seed)
        self.H = H if H is not None else rng.randint(3, 9)

        self.base_doys = torch.tensor(
            [rng.uniform(0, 365) for _ in range(self.H)],
            device=device, dtype=torch.float32,
        )
        self.sigmas = torch.tensor(
            [rng.uniform(1.0, 3.0) for _ in range(self.H)],
            device=device, dtype=torch.float32,
        )
        self.importances = torch.tensor(
            [float(np.clip(rng.lognormal(-0.2, 0.5), 0.2, 2.5)) for _ in range(self.H)],
            device=device, dtype=torch.float32,
        )

    @staticmethod
    def sample_table_sens(H: int, seed: int, device: str = "cpu") -> torch.Tensor:
        """Sample per-table sensitivity vector of shape (H,)."""
        rng = np.random.RandomState(seed)
        active = float(rng.random() < 0.4)
        local_sens = torch.tensor(
            [rng.beta(1, 4) for _ in range(H)],
            device=device, dtype=torch.float32,
        )
        return active * local_sens

    def evaluate(self, t: torch.Tensor, table_sens: torch.Tensor) -> torch.Tensor:
        """Evaluate event contribution at day indices t.

        Args:
            t: (N,) day indices since epoch.
            table_sens: (H,) per-table sensitivity.

        Returns:
            (N,) event contribution.
        """
        N = t.shape[0]
        H = self.H
        years = torch.floor(t / 365.25).unsqueeze(1)  # (N, 1)

        # Deterministic per-year jitter via sine
        jitter = self.sigmas.unsqueeze(0) * 0.3 * torch.sin(
            years * 7.123 + self.base_doys.unsqueeze(0) * 0.731
        )  # (N, H)

        t_center = years * 365.25 + self.base_doys.unsqueeze(0) + jitter  # (N, H)
        dist = (t.unsqueeze(1) - t_center) / self.sigmas.unsqueeze(0)  # (N, H)
        contributions = torch.exp(-0.5 * dist ** 2)  # (N, H)

        amp = self.importances * table_sens  # (H,)
        return (contributions * amp.unsqueeze(0)).sum(dim=1)  # (N,)

    def to_dict(self) -> dict:
        return {
            "H": self.H,
            "base_doys": self.base_doys.cpu().tolist(),
            "sigmas": self.sigmas.cpu().tolist(),
            "importances": self.importances.cpu().tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict, device: str = "cpu") -> "EventCalendar":
        obj = cls.__new__(cls)
        obj.device = device
        obj.H = d["H"]
        obj.base_doys = torch.tensor(d["base_doys"], device=device, dtype=torch.float32)
        obj.sigmas = torch.tensor(d["sigmas"], device=device, dtype=torch.float32)
        obj.importances = torch.tensor(d["importances"], device=device, dtype=torch.float32)
        return obj


class TemporalVocab:
    """Calendar-aligned temporal intensity generator.

    Composition:
        raw(t) = trend(t_norm) + seasonality(t) + events(t) + noise(t)
        intensity(t) = softplus(raw(t)) * dow_factor(t)

    where seasonality(t) uses Fourier series aligned to week/month/year periods,
    events come from a shared EventCalendar with per-table sensitivity,
    and dow_factor(t) = p_dow[dow(t)] * 7 is sampled from a Dirichlet mixture.
    """

    def __init__(self, device: str = "cpu", num_days: float = NUM_DAYS):
        self.device = device
        self.num_days = num_days

        self.trend_vocab = TrendVocab(device=device)
        self.seasonality_vocab = SeasonalityVocab(device=device, num_days=num_days)
        self.noise_vocab = NoiseVocab(device=device)

        self.event_calendar: Optional[EventCalendar] = None
        self.table_sens: Optional[torch.Tensor] = None
        self.intensity: Optional[torch.Tensor] = None

        self._sample_params()

    def _sample_params(self):
        """Sample per-table modulation weights, trend params, and DOW pattern."""
        self.m_week = random.uniform(0.0, 3.0)
        self.m_month = random.uniform(0.0, 0.2)
        self.m_year = random.uniform(0.0, 0.3)

        # DOW pattern: Dirichlet mixture → probability vector
        dow_seed = random.randint(0, 2**31 - 1)
        self.p_dow = _sample_dow_pvec(dow_seed, self.device)  # (7,), sum=1

        self.m_lin = random.gauss(0.0, 0.3)
        self.c_lin = random.gauss(0.0, 0.1)

        self.noise_std = 10.0 ** random.uniform(-3.0, -1.0)  # log-uniform [0.001, 0.1]

        self.seasonality_vocab.sample_params()

    def set_calendar(self, event_calendar: EventCalendar, table_sens: torch.Tensor):
        """Attach an RDB-level event calendar with this table's sensitivity."""
        self.event_calendar = event_calendar
        self.table_sens = table_sens.to(self.device)

    def generate(
        self,
        time_range: Tuple[float, float] = (0.0, float(NUM_DAYS)),
        num_points: int = DEFAULT_NUM_POINTS,
    ) -> torch.Tensor:
        """Generate temporal intensity distribution.

        Returns:
            (num_points, 2) tensor: [t (days), intensity].
        """
        t_start, t_end = time_range
        t = torch.linspace(t_start, t_end, num_points, device=self.device)

        # Trend (normalized)
        t_norm = t / self.num_days
        trend = self.m_lin * t_norm + self.c_lin

        # Fourier seasonality
        seasonal = self.seasonality_vocab.evaluate(t)
        seasonal = self.m_week * seasonal[0] + self.m_month * seasonal[1] + self.m_year * seasonal[2]

        # DOW multiplicative factor: p_dow[dow] * 7, mean≈1
        dow_idx = ((t.long() + 4) % 7).to(self.device)  # 1970-01-01 is Thursday (dow=4)
        dow_factor = self.p_dow[dow_idx] * 7.0  # (N,), mean ≈ 1

        # Events
        if self.event_calendar is not None and self.table_sens is not None:
            events = self.event_calendar.evaluate(t, self.table_sens)
        else:
            events = torch.zeros_like(t)

        # Noise
        noise = self.noise_std * torch.randn_like(t)

        # Compose: base intensity via softplus, then multiply by DOW factor
        raw = trend + seasonal + events + noise
        base_intensity = torch.nn.functional.softplus(raw)
        intensity = base_intensity * dow_factor

        self.intensity = intensity
        return torch.stack([t, intensity], dim=1)

    def sample_time(
        self,
        num_samples: int,
        time_range: Tuple[float, float] = (0.0, float(NUM_DAYS)),
        num_points: int = DEFAULT_NUM_POINTS,
        distribution: Optional[torch.Tensor] = None,
        t_min: Optional[torch.Tensor] = None,
        gamma: float = 0.0,
    ) -> torch.Tensor:
        """Sample event times with optional per-row lower bounds.

        Args:
            num_samples: Number of samples (rows).
            time_range: (min_t, max_t) in days.
            num_points: Discretization grid size.
            distribution: Pre-generated (t, intensity) tensor.
            t_min: Optional (num_samples,) per-row lower bounds in days.
            gamma: Lifecycle decay strength.

        Returns:
            (num_samples,) sampled day indices, unsorted.
        """
        if distribution is None:
            distribution = self.generate(time_range, num_points=num_points)

        t_points = distribution[:, 0]
        intensity = distribution[:, 1]
        K = t_points.shape[0]
        device = t_points.device

        if t_min is not None:
            t_min_clamped = t_min.to(device).clamp(min=time_range[0], max=time_range[1])
            mask = t_points.unsqueeze(0) >= t_min_clamped.unsqueeze(1)
            intensity_bc = intensity.unsqueeze(0) * mask.float()
        else:
            intensity_bc = intensity.unsqueeze(0).expand(num_samples, -1)

        if gamma > 0.0 and t_min is not None:
            tau = (t_points.unsqueeze(0) - t_min_clamped.unsqueeze(1)) / (
                time_range[1] - t_min_clamped.unsqueeze(1)
            ).clamp(min=1e-8)
            tau = tau.clamp(min=0.0)
            decay = torch.exp(-gamma * tau)
            intensity_bc = intensity_bc * decay

        row_sums = intensity_bc.sum(dim=1, keepdim=True).clamp(min=1e-12)
        probs = intensity_bc / row_sums

        sample_indices = torch.multinomial(probs, 1, replacement=True).squeeze(-1)
        sampled_times = t_points[sample_indices]
        return sampled_times

    def evaluate_basis(self, t: torch.Tensor) -> torch.Tensor:
        """Evaluate basis features at day indices t.

        Args:
            t: (N,) day indices.

        Returns:
            (N, 8): [trend_level, trend_slope, week_sin1, week_cos1,
                      month_sin1, dow_factor, year_sin1, year_cos1].
        """
        N = t.shape[0]

        # Trend basis
        t_norm = t / self.num_days
        trend_level = self.m_lin * t_norm + self.c_lin
        trend_slope = torch.full_like(t, self.m_lin / self.num_days)

        # Fourier basis: first harmonic of each period
        w_sin1 = torch.sin(2 * np.pi * t / 7.0)
        w_cos1 = torch.cos(2 * np.pi * t / 7.0)
        m_sin1 = torch.sin(2 * np.pi * t / 30.4375)
        y_sin1 = torch.sin(2 * np.pi * t / 365.25)
        y_cos1 = torch.cos(2 * np.pi * t / 365.25)

        # DOW multiplicative factor at each time point
        dow_idx = ((t.long() + 4) % 7).to(self.device)
        dow_factor = self.p_dow[dow_idx] * 7.0

        return torch.stack([
            trend_level, trend_slope,
            w_sin1, w_cos1,
            m_sin1, dow_factor,
            y_sin1, y_cos1,
        ], dim=-1)

    def build_gate_vector(self) -> torch.Tensor:
        """Return (3,) float: [trend_active, seasonal_active, events_active]."""
        events_active = 1.0 if (self.event_calendar is not None and
                                self.table_sens is not None and
                                self.table_sens.sum() > 0) else 0.0
        return torch.tensor(
            [1.0, 1.0, events_active],
            device=self.device, dtype=torch.float32,
        )

    @property
    def get_intensity(self) -> torch.Tensor:
        return self.intensity

    def __len__(self):
        return 3  # trend, seasonality, events


class TrendVocab:
    """Simple linear trend component."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def evaluate(self, t_norm: torch.Tensor, m_lin: float, c_lin: float) -> torch.Tensor:
        """Evaluate trend at normalized time points."""
        return m_lin * t_norm + c_lin


class SeasonalityVocab:
    """Calendar-aligned Fourier seasonality.

    Three periods, each with multiple harmonics:
      week:  3 harmonics, period = 7 days
      month: 6 harmonics, period = 30.4375 days
      year:  6 harmonics, period = 365.25 days

    Coefficients c_f, d_f ~ N(0, 1/f), rescaled to unit norm.
    """

    def __init__(self, device: str = "cpu", num_days: float = NUM_DAYS):
        self.device = device
        self.num_days = num_days
        self.week_nharm = 3
        self.month_nharm = 6
        self.year_nharm = 6

    def sample_params(self):
        """Sample per-table Fourier coefficients."""
        self.week_coeffs = self._sample_fourier_coeffs(self.week_nharm)
        self.month_coeffs = self._sample_fourier_coeffs(self.month_nharm)
        self.year_coeffs = self._sample_fourier_coeffs(self.year_nharm)

    def _sample_fourier_coeffs(self, n_harmonics: int) -> torch.Tensor:
        """Sample c_f, d_f ~ N(0, 1/f), rescale to unit norm.

        Returns:
            (n_harmonics * 2,) tensor: [c_1, ..., c_F, d_1, ..., d_F].
        """
        coeffs = []
        for f in range(1, n_harmonics + 1):
            std = 1.0 / f
            coeffs.append(random.gauss(0, std))  # c_f
            coeffs.append(random.gauss(0, std))  # d_f
        t = torch.tensor(coeffs, device=self.device, dtype=torch.float32)
        norm = t.norm()
        if norm > 1e-8:
            t = t / norm
        return t

    def _eval_fourier_series(
        self, t: torch.Tensor, period: float, coeffs: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate Fourier series at day indices t.

        Args:
            t: (N,) day indices.
            period: Period in days.
            coeffs: (2F,) tensor [c_1..c_F, d_1..d_F].

        Returns:
            (N,) series values.
        """
        F = len(coeffs) // 2
        omega = 2 * np.pi / period
        result = torch.zeros_like(t)
        for f in range(F):
            c = coeffs[f]
            d = coeffs[F + f]
            freq = (f + 1) * omega
            result += c * torch.sin(freq * t) + d * torch.cos(freq * t)
        return result

    def evaluate(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate all three Fourier series at day indices t.

        Returns:
            (week_vals, month_vals, year_vals): each (N,).
        """
        week = self._eval_fourier_series(t, 7.0, self.week_coeffs)
        month = self._eval_fourier_series(t, 30.4375, self.month_coeffs)
        year = self._eval_fourier_series(t, 365.25, self.year_coeffs)
        return week, month, year

    def __len__(self):
        return (self.week_nharm + self.month_nharm + self.year_nharm) * 2


class NoiseVocab:
    """Small Gaussian noise for temporal intensity."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def generate(self, t: torch.Tensor, std: float = 0.01) -> torch.Tensor:
        return std * torch.randn_like(t, device=self.device)

    def __len__(self):
        return 1
