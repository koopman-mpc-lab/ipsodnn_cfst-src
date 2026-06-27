from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class IPSOConfig:
    population_size: int = 50
    max_iterations: int = 200
    c1: float = 2.0
    c2: float = 2.0
    w_max: float = 0.9
    w_min: float = 0.4
    alpha: float = 0.5
    stagnation_threshold: int = 5
    levy_lambda: float = 1.5
    levy_beta0: float = 0.01


def adaptive_inertia_weight(
    iteration: int,
    max_iterations: int,
    fitness_values: np.ndarray,
    config: IPSOConfig,
) -> float:
    sigma_f = float(np.std(fitness_values))
    normalized_sigma = sigma_f / (np.mean(fitness_values) + 1e-8)
    progress = iteration / max(max_iterations, 1)
    return config.w_max - (config.w_max - config.w_min) * np.sin(0.5 * np.pi * progress) * np.exp(
        -config.alpha * normalized_sigma
    )


def levy_step(config: IPSOConfig, rng: np.random.Generator) -> float:
    sigma_u = (
        (
            math.gamma(1 + config.levy_lambda)
            * math.sin(math.pi * config.levy_lambda / 2)
        )
        / (
            math.gamma((1 + config.levy_lambda) / 2)
            * config.levy_lambda
            * 2 ** ((config.levy_lambda - 1) / 2)
        )
    ) ** (1 / config.levy_lambda)
    u = rng.normal(0.0, sigma_u)
    v = rng.normal(0.0, 1.0)
    return float(u / (abs(v) ** (1 / config.levy_lambda)))


@dataclass
class IPSOResult:
    best_position: np.ndarray
    best_fitness: float
    history: list[float]


class IPSOOptimizer:
    def __init__(
        self,
        bounds_low: np.ndarray,
        bounds_high: np.ndarray,
        fitness_fn,
        config: IPSOConfig | None = None,
        random_seed: int = 42,
    ) -> None:
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.fitness_fn = fitness_fn
        self.config = config or IPSOConfig()
        self.rng = np.random.default_rng(random_seed)
        self.dimension = bounds_low.size

    def _clip(self, position: np.ndarray) -> np.ndarray:
        return np.clip(position, self.bounds_low, self.bounds_high)

    def optimize(self) -> IPSOResult:
        cfg = self.config
        positions = self.rng.uniform(self.bounds_low, self.bounds_high, size=(cfg.population_size, self.dimension))
        velocities = self.rng.uniform(-0.1, 0.1, size=(cfg.population_size, self.dimension))
        personal_best = positions.copy()
        personal_fitness = np.array([self.fitness_fn(row) for row in positions], dtype=np.float64)
        global_best_idx = int(np.argmin(personal_fitness))
        global_best = personal_best[global_best_idx].copy()
        global_fitness = float(personal_fitness[global_best_idx])
        stagnation = np.zeros(cfg.population_size, dtype=int)
        history = [global_fitness]

        for iteration in range(cfg.max_iterations):
            inertia = adaptive_inertia_weight(iteration, cfg.max_iterations, personal_fitness, cfg)
            r1 = self.rng.random((cfg.population_size, self.dimension))
            r2 = self.rng.random((cfg.population_size, self.dimension))
            velocities = (
                inertia * velocities
                + cfg.c1 * r1 * (personal_best - positions)
                + cfg.c2 * r2 * (global_best - positions)
            )
            positions = self._clip(positions + velocities)

            for particle_idx in range(cfg.population_size):
                fitness = float(self.fitness_fn(positions[particle_idx]))
                if fitness < personal_fitness[particle_idx]:
                    personal_best[particle_idx] = positions[particle_idx].copy()
                    personal_fitness[particle_idx] = fitness
                    stagnation[particle_idx] = 0
                else:
                    stagnation[particle_idx] += 1

                if stagnation[particle_idx] >= cfg.stagnation_threshold:
                    step = levy_step(cfg, self.rng)
                    positions[particle_idx] = self._clip(
                        positions[particle_idx]
                        + cfg.levy_beta0 * step * (positions[particle_idx] - global_best)
                    )
                    stagnation[particle_idx] = 0
                    fitness = float(self.fitness_fn(positions[particle_idx]))
                    if fitness < personal_fitness[particle_idx]:
                        personal_best[particle_idx] = positions[particle_idx].copy()
                        personal_fitness[particle_idx] = fitness

            global_best_idx = int(np.argmin(personal_fitness))
            if personal_fitness[global_best_idx] < global_fitness:
                global_best = personal_best[global_best_idx].copy()
                global_fitness = float(personal_fitness[global_best_idx])
            history.append(global_fitness)

        return IPSOResult(best_position=global_best, best_fitness=global_fitness, history=history)
