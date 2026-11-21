"""Семплирование: практическое сравнение методов отбора подвыборки.

Сравниваем на реальном датасете Kaggle, как способ отбора подвыборки влияет
на смещение оценки: простое случайное, стратифицированное, взвешенное
(importance sampling), систематическое и reservoir sampling (для потока).

Запуск: python src/sampling_demo.py
Результат: reports/sampling_demo.md + reports/sampling_demo.json
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from config import PARAMS, PROCESSED_DIR, REPORTS_DIR

TARGET = PARAMS["dataset"]["target"]
SEED = 42
N = 300  # размер подвыборки


def reservoir_sample(stream, k: int, seed: int = SEED) -> list:
    """Reservoir sampling (алгоритм R): равномерная выборка k элементов
    из потока неизвестной длины за один проход и O(k) памяти.
    Именно так семплируют логи production-трафика, которые не влезают в память."""
    rng = random.Random(seed)
    res: list = []
    for i, item in enumerate(stream):
        if i < k:
            res.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                res[j] = item
    return res


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "train.csv")
    pop_rate = df[TARGET].mean()
    pop_charges = df["MonthlyCharges"].mean()
    rng = np.random.default_rng(SEED)
    results = {}

    # 1. Простое случайное (SRS): несмещённое в среднем, но на малом n
    #    доля редкого класса "гуляет" от прогона к прогону.
    srs_rates = [df.sample(N, random_state=s)[TARGET].mean() for s in range(50)]
    s1 = df.sample(N, random_state=SEED)
    results["simple_random"] = {
        "churn_rate": round(float(s1[TARGET].mean()), 4),
        "churn_rate_std_over_50_runs": round(float(np.std(srs_rates)), 4),
        "mean_monthly_charges": round(float(s1["MonthlyCharges"].mean()), 2),
    }

    # 2. Стратифицированное: доля классов фиксирована по построению ->
    #    дисперсия оценки ниже. Это то, что используется в prepare_data.py.
    strat = (df.groupby(TARGET, group_keys=False)
               .apply(lambda g: g.sample(int(round(N * len(g) / len(df))),
                                         random_state=SEED), include_groups=True))
    strat_rates = [
        df.groupby(TARGET, group_keys=False)
          .apply(lambda g: g.sample(int(round(N * len(g) / len(df))), random_state=s),
                 include_groups=True)[TARGET].mean()
        for s in range(50)
    ]
    results["stratified"] = {
        "churn_rate": round(float(strat[TARGET].mean()), 4),
        "churn_rate_std_over_50_runs": round(float(np.std(strat_rates)), 4),
        "mean_monthly_charges": round(float(strat["MonthlyCharges"].mean()), 2),
    }

    # 3. Oversampling редкого класса 50/50: удобно для обучения на
    #    несбалансированных данных, но ОЦЕНКУ так делать нельзя --
    #    churn_rate уезжает к 0.5 и метрики становятся несопоставимыми.
    half = N // 2
    bal = pd.concat([
        df[df[TARGET] == 0].sample(half, random_state=SEED),
        df[df[TARGET] == 1].sample(half, random_state=SEED),
    ])
    results["balanced_oversample"] = {
        "churn_rate": round(float(bal[TARGET].mean()), 4),
        "mean_monthly_charges": round(float(bal["MonthlyCharges"].mean()), 2),
        "note": "смещает baseline: годится для обучения, не для оценки качества",
    }

    # 4. Взвешенное (importance sampling) по MonthlyCharges: намеренно
    #    перекашиваем выборку к дорогим клиентам -- так собирают выборку
    #    под бизнес-приоритет, но затем метрики нужно перевешивать обратно.
    w = df["MonthlyCharges"] / df["MonthlyCharges"].sum()
    idx = rng.choice(df.index, size=N, replace=False, p=w)
    weighted = df.loc[idx]
    results["weighted_by_revenue"] = {
        "churn_rate": round(float(weighted[TARGET].mean()), 4),
        "mean_monthly_charges": round(float(weighted["MonthlyCharges"].mean()), 2),
        "note": "выборка смещена к дорогим клиентам -- нужна коррекция веса при оценке",
    }

    # 5. Систематическое (каждый k-й): дешёвое, но ломается, если в данных
    #    есть периодичность, совпадающая с шагом.
    step = len(df) // N
    systematic = df.iloc[::step].head(N)
    results["systematic"] = {
        "churn_rate": round(float(systematic[TARGET].mean()), 4),
        "mean_monthly_charges": round(float(systematic["MonthlyCharges"].mean()), 2),
        "note": "риск смещения при периодичности в порядке строк",
    }

    # 6. Reservoir: один проход по потоку, память O(k).
    res_idx = reservoir_sample(iter(df.index), N)
    reservoir = df.loc[res_idx]
    results["reservoir_stream"] = {
        "churn_rate": round(float(reservoir[TARGET].mean()), 4),
        "mean_monthly_charges": round(float(reservoir["MonthlyCharges"].mean()), 2),
        "note": "для потоков неизвестной длины: один проход, O(k) памяти",
    }

    out = {
        "population": {
            "n_rows": int(len(df)),
            "churn_rate": round(float(pop_rate), 4),
            "mean_monthly_charges": round(float(pop_charges), 2),
        },
        "sample_size": N,
        "methods": results,
    }
    (REPORTS_DIR / "sampling_demo.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Семплирование: сравнение методов на Kaggle Telco Customer Churn",
        "",
        f"Генеральная совокупность (train): **{len(df)} строк**, "
        f"доля churn = **{pop_rate:.4f}**, средний MonthlyCharges = **{pop_charges:.2f}**.",
        f"Размер подвыборки n = **{N}**.",
        "",
        "| Метод | churn_rate | отклонение от популяции | std по 50 прогонам | mean MonthlyCharges |",
        "|---|---|---|---|---|",
    ]
    for name, m in results.items():
        std = m.get("churn_rate_std_over_50_runs", "—")
        lines.append(
            f"| {name} | {m['churn_rate']:.4f} | "
            f"{m['churn_rate'] - pop_rate:+.4f} | {std} | {m['mean_monthly_charges']:.2f} |"
        )
    lines += [
        "",
        "## Вывод",
        "",
        "Стратифицированное семплирование даёт ту же среднюю оценку, что простое "
        "случайное, но с заметно меньшей дисперсией между прогонами — поэтому "
        "именно оно используется в `src/prepare_data.py` для train/test split.",
        "Oversampling и взвешенное семплирование намеренно смещают выборку: "
        "они допустимы на этапе обучения, но метрики на такой выборке нельзя "
        "сравнивать с production без обратного перевешивания.",
    ]
    (REPORTS_DIR / "sampling_demo.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
