"""Uzaysal öngörü-beceri metriği (alan-beceri skoru + yoğunlaşma kazancı).

TANIMLAYICI bir beceri ölçüsüdür — deprem TAHMİNİ DEĞİLDİR. Sorduğu soru:
bir EĞİTİM penceresinden kurulan göreli risk yüzeyi, GELECEKTEKİ (TEST)
depremleri alan-uniform (Poisson) bir baseline'a göre daha iyi YOĞUNLAŞTIRIYOR
mu? Pozitif skor, yüksek-riskli hücrelerin gelecekteki olayları baseline'dan
fazla topladığını; sıfır, alan-uniform baseline ile aynı performansı; negatif,
baseline'dan kötü olduğunu gösterir.

Bu, projenin "naive baseline'ı yenebiliyor mu?" çekirdek sorusunun hücre-bazlı
uzaysal cevabıdır; bir olasılık ya da deprem öngörüsü iddiası değildir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Region
from ..web.export import _to_utc, build_cell_index, cell_bins

try:  # numpy 2.4+ : np.trapz tamamen kaldırıldı, np.trapezoid ile değiştirildi
    from numpy import trapezoid as _trapz
except ImportError:  # daha eski numpy sürümleri
    from numpy import trapz as _trapz


def _coerce_counts(counts_sorted_by_risk: np.ndarray) -> np.ndarray:
    """float diziye çevirir; NaN/inf ve negatif değerleri reddeder.

    `total <= 0` koruması NaN'i kaçırır (NaN karşılaştırmaları False döner) ve
    negatifleri yakalamaz; bu yüzden sayım vektörü önce burada doğrulanır.
    """
    counts = np.asarray(counts_sorted_by_risk, dtype=float)
    if not np.all(np.isfinite(counts)):
        raise ValueError("Sonlu olmayan (NaN/inf) olay sayısı geçersiz.")
    if np.any(counts < 0):
        raise ValueError("Negatif olay sayısı geçersiz.")
    return counts


def area_skill_score(counts_sorted_by_risk: np.ndarray) -> float:
    """Alan-beceri skoru (Area Skill Score): ASS = 2*AUC - 1.

    Girdi: hücreler AZALAN riske göre sıralı; her değer o hücredeki GELECEK olay
    sayısı. Eşit-alan varsayımı (her hücre = 1/n alan). AUC, kümülatif yakalanan
    olay eğrisinin (y; 1'e normalize, başı 0; n+1 nokta) kümülatif hücre oranına
    (x = linspace(0, 1, n+1)) karşı trapez alanıdır. Bu, simetrik Lorenz/Gini
    inşasıdır:
        +1'e yakın → güçlü yoğunlaşma (yüksek-risk hücreler olayları topluyor),
         0          → alan-uniform baseline ile aynı,
        -1'e yakın → ters yoğunlaşma (baseline'dan kötü).

    Önemli: sonlu n için skorun TAVANI tam 1.0 değil, (n-1)/n'dir (Gini sınırı) —
    tüm olaylar tek hücrede toplansa bile. Örn. n=8 için tavan 7/8 = 0.875.

    Bu, simetrik Lorenz/Gini ölçüsüdür; Zechar-Jordan (2008) Molchan Area Skill
    Score DEĞİLDİR ve istatistiksel anlamlılık iddia ETMEZ — anlamlılık ayrı bir
    bootstrap testi gerektirir.

    Sıralama varsayımı: risk_index 0–100 arası tamsayıya yuvarlanır, dolayısıyla
    hücreler eşitlenebilir (tie); eşitlik içindeki sıralama keyfîdir ve skoru hafif
    kaydırabilir — belgelenmiş bir varsayım, garanti değil.

    NaN/inf, negatif ya da toplamı pozitif olmayan girdide ValueError yükseltir.
    Tanımlayıcı beceri ölçüsü; deprem tahmini değildir.
    """
    counts = _coerce_counts(counts_sorted_by_risk)
    total = counts.sum()
    if total <= 0:
        raise ValueError("area_skill_score: toplam olay sayısı pozitif olmalı")
    n = counts.size
    captured = np.concatenate([[0.0], np.cumsum(counts) / total])  # n+1 nokta, başı 0
    cell_fraction = np.linspace(0.0, 1.0, n + 1)
    auc = float(_trapz(captured, cell_fraction))
    return round(2.0 * auc - 1.0, 4)


def spatial_concentration_gain(
    counts_sorted_by_risk: np.ndarray, area_fraction: float = 0.25
) -> float:
    """En riskli `area_fraction` alanının yoğunlaşma kazancı (concentration gain).

    En yüksek riskli k = max(1, round(area_fraction * n)) hücrenin yakaladığı olay
    oranının, bu hücrelerin GERÇEKLEŞEN alan oranına (k / n) bölümü. 1.0 = alan-
    uniform baseline (yakalanan oran = alan oranı); >1 baseline-üstü yoğunlaşma;
    <1 baseline-altı.

    Not: tabana `area_fraction` değil k/n yazılır — yuvarlama nedeniyle k/n ≠
    area_fraction olabilir; istenen oranla bölmek sıfır-beceri uniform alanı yanlış
    şekilde 1.0'dan saptırırdı.

    NaN/inf, negatif, toplamı <= 0 ya da area_fraction (0, 1] dışındaysa ValueError.
    Tanımlayıcı ölçü; deprem tahmini değildir.
    """
    counts = _coerce_counts(counts_sorted_by_risk)
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("spatial_concentration_gain: toplam olay sayısı pozitif olmalı")
    if not 0.0 < area_fraction <= 1.0:
        raise ValueError("spatial_concentration_gain: area_fraction (0, 1] aralığında olmalı")
    n = counts.size
    k = max(1, round(area_fraction * n))
    captured = float(counts[:k].sum()) / total
    return round(captured / (k / n), 4)


def evaluate_surface_skill(
    df: pd.DataFrame,
    region: Region,
    *,
    min_mag: float = 4.0,
    cell_deg: float = 0.25,
    test_frac: float = 0.25,
) -> dict:
    """Zaman-dışı (out-of-time) uzaysal beceri değerlendirmesi.

    Katalog, (1 - test_frac) zaman-kuantilinde EĞİTİM/TEST olarak ayrılır. Risk
    yüzeyi YALNIZCA eğitimden kurulur (`build_cell_index` yeniden kullanılır);
    sonra TEST olayları (mag >= min_mag) eğitim hücre kutularına atanır ve her
    hücredeki gelecek-olay sayısı, kurulan hücre DataFrame'iyle AYNI satır
    sırasında sayılır. Bu sayı vektörü `area_skill_score`/`spatial_concentration_
    gain`'e verilir.

    Döner: {"area_skill_score", "gain_top25", "n_train_cells", "n_test_events"}.
    `n_test_events`, eğitim hücrelerine DÜŞEN test olaylarının sayısıdır; hiç test
    olayı düşmezse skill ve gain None döner. Eğitim veya test penceresi boşsa
    ValueError yükseltir.

    Sıralama varsayımı: hücreler tamsayı risk_index'e (0–100) göre sıralanır,
    eşitlikler (tie) mümkündür ve eşitlik içi sıra keyfîdir; skoru hafif
    kaydırabilir — belgelenmiş varsayım, garanti değil.

    Tanımlayıcı beceri ölçüsü — deprem tahmini değildir.
    """
    if df.empty:
        raise ValueError("evaluate_surface_skill: katalog boş")
    times = _to_utc(df["time"])
    split = times.quantile(1.0 - test_frac)
    train = df[times <= split]
    test = df[times > split]
    if train.empty:
        raise ValueError("evaluate_surface_skill: eğitim penceresi boş")
    if test.empty:
        raise ValueError("evaluate_surface_skill: test penceresi boş")

    cells = build_cell_index(train, region, cell_deg=cell_deg, min_mag=min_mag)
    n_train_cells = int(len(cells))

    test_sub = test[test["mag"] >= min_mag]
    test_lat_bin, test_lon_bin = cell_bins(
        test_sub["latitude"], test_sub["longitude"], region, cell_deg
    )
    test_counts = (
        pd.DataFrame({"lat_bin": test_lat_bin, "lon_bin": test_lon_bin})
        .groupby(["lat_bin", "lon_bin"])
        .size()
    )
    # Test olaylarını kurulan hücrelerin AYNI satır sırasında say.
    counts = np.array(
        [
            int(test_counts.get((int(row.lat_bin), int(row.lon_bin)), 0))
            for row in cells.itertuples(index=False)
        ],
        dtype=float,
    )
    n_test_events = int(counts.sum())

    if n_test_events <= 0:
        return {
            "area_skill_score": None,
            "gain_top25": None,
            "n_train_cells": n_train_cells,
            "n_test_events": 0,
        }
    return {
        "area_skill_score": area_skill_score(counts),
        "gain_top25": spatial_concentration_gain(counts, area_fraction=0.25),
        "n_train_cells": n_train_cells,
        "n_test_events": n_test_events,
    }
