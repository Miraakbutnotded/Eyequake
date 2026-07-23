"""Keşifsel veri analizi (EDA) + katalog tamlık teşhisi.

Üretir:
  - reports/figures/*.png  (yıllık sayım, FMD, uzaysal, derinlik, kümülatif)
  - konsola Mc, b-değeri ve tamlık teşhisi

Kullanım:
    ./.venv/bin/python scripts/02_eda.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # başlıksız (non-interactive) backend
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from eyequake.analysis.seismology import (  # noqa: E402
    b_value_aki,
    frequency_magnitude_distribution,
    magnitude_of_completeness,
)
from eyequake.config import FIGURES_DIR  # noqa: E402
from eyequake.data.clean import load_catalog  # noqa: E402

THRESHOLDS = [2.5, 3.5, 4.0, 4.5]


def plot_yearly_counts(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for thr in THRESHOLDS:
        counts = df[df["mag"] >= thr].groupby("year").size()
        ax.plot(counts.index, counts.values, marker=".", label=f"M≥{thr}")
    ax.set(xlabel="Yıl", ylabel="Olay sayısı", title="Yıllık deprem sayısı (USGS, Türkiye)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_yearly_counts.png", dpi=130)
    plt.close(fig)


def plot_fmd(df: pd.DataFrame, mc: float) -> None:
    fmd = frequency_magnitude_distribution(df["mag"].to_numpy())
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(fmd.bin_centers, fmd.cumulative, "s-", label="Kümülatif (M≥)", ms=4)
    ax.semilogy(fmd.bin_centers, np.maximum(fmd.incremental, 0.1), ".", alpha=0.5,
                label="Artımlı")
    ax.axvline(mc, color="red", ls="--", label=f"Mc ≈ {mc}")
    ax.set(xlabel="Büyüklük (M)", ylabel="Olay sayısı (log)",
           title="Frequency-Magnitude Distribution (Gutenberg-Richter)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_fmd_gutenberg_richter.png", dpi=130)
    plt.close(fig)


def plot_spatial(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    big = df[df["mag"] >= 4.0]
    sc = ax.scatter(df["longitude"], df["latitude"], c=df["mag"], s=3,
                    cmap="YlOrRd", alpha=0.4)
    ax.scatter(big["longitude"], big["latitude"], s=20, facecolors="none",
               edgecolors="black", linewidths=0.4, label="M≥4.0")
    fig.colorbar(sc, ax=ax, label="Büyüklük")
    ax.set(xlabel="Boylam", ylabel="Enlem", title="Episantr dağılımı (1990–2026)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_spatial_epicenters.png", dpi=130)
    plt.close(fig)


def plot_depth(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["depth"].clip(upper=200), bins=60, color="steelblue")
    ax.set(xlabel="Derinlik (km)", ylabel="Olay sayısı", title="Derinlik dağılımı")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_depth_hist.png", dpi=130)
    plt.close(fig)


def plot_cumulative(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    strong = df[df["mag"] >= 4.5].sort_values("time")
    ax.plot(strong["time"], np.arange(1, len(strong) + 1), color="darkred")
    ax.set(xlabel="Tarih", ylabel="Kümülatif olay (M≥4.5)",
           title="Kümülatif güçlü deprem sayısı — eğim = sismisite oranı")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_cumulative_strong.png", dpi=130)
    plt.close(fig)


def main() -> int:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_catalog()
    mags = df["mag"].to_numpy()

    mc = magnitude_of_completeness(mags)
    bv = b_value_aki(mags, mc)

    plot_yearly_counts(df)
    plot_fmd(df, mc)
    plot_spatial(df)
    plot_depth(df)
    plot_cumulative(df)

    print("=== TAMLIK TEŞHİSİ ===")
    print(f"Toplam olay: {len(df)}")
    print(f"Magnitude of completeness (Mc): {mc}")
    print(f"Gutenberg-Richter b-değeri: {bv.b} ± {bv.sigma} (Mc üstü n={bv.n_above_mc})")
    print(f"GR a-değeri: {bv.a}")
    print("\n=== EŞİĞE GÖRE OLAY SAYISI ===")
    for thr in THRESHOLDS:
        n = int((df["mag"] >= thr).sum())
        print(f"  M≥{thr}: {n}")

    # Tamlık yanlılığını sayısal göster: ilk vs son 10 yıl, sabit yüksek eşikte.
    print("\n=== ZAMAN-DEĞİŞKEN TAMLIK (yıllık ort. olay) ===")
    for thr in (2.5, 4.0):
        early = df[(df["mag"] >= thr) & (df["year"].between(1994, 2003))]
        late = df[(df["mag"] >= thr) & (df["year"].between(2014, 2023))]
        print(f"  M≥{thr}: 1994-2003 ≈ {len(early) / 10:.0f}/yıl | "
              f"2014-2023 ≈ {len(late) / 10:.0f}/yıl")

    print(f"\nFigürler: {FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
