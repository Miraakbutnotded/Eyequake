"""baselines.py sözleşme testleri: dürüst baseline + sızıntısızlık + fail-closed.

Bu modül projenin değişmez kuralının referansıdır ("her model naive baseline'a karşı
ölçülür"). Baseline sessizce bozulursa, üstündeki TÜM beceri iddiaları anlamsızlaşır —
o yüzden burada sınır davranışı da matematik kadar önemli.
"""

from __future__ import annotations

import numpy as np
import pytest

from eyequake.forecast.baselines import climatology_forecast, persistence_forecast
from eyequake.forecast.evaluate import score, skill_score


def test_climatology_predicts_training_mean_as_constant():
    y_train = np.array([2.0, 4.0, 6.0])
    pred = climatology_forecast(y_train, n_test=4)
    assert pred.shape == (4,)
    assert np.allclose(pred, 4.0)  # sabit tahmin = eğitim ortalaması


def test_climatology_ignores_test_length_in_the_mean():
    # n_test tahminin DEĞERİNİ değil yalnızca uzunluğunu belirler.
    y_train = np.array([1.0, 3.0])
    assert climatology_forecast(y_train, 1)[0] == pytest.approx(2.0)
    assert climatology_forecast(y_train, 50)[0] == pytest.approx(2.0)


def test_climatology_all_zero_history_predicts_zero():
    pred = climatology_forecast(np.zeros(5), n_test=3)
    assert np.allclose(pred, 0.0)  # sismisite yoksa baseline de sıfır


def test_persistence_is_previous_value_starting_from_last_train():
    y_train = np.array([1.0, 2.0, 9.0])
    y_test = np.array([10.0, 20.0, 30.0])
    pred = persistence_forecast(y_train, y_test)
    assert pred.tolist() == [9.0, 10.0, 20.0]  # ilk tahmin son eğitim değeri


def test_persistence_never_uses_the_concurrent_test_value():
    """Sızıntı (leakage) koruması: y_hat[t] asla y_test[t] olamaz.

    Persistence'in t anındaki tahmini t-1'in GERÇEĞİdir. Eğer y_test[t] kullanılsaydı
    baseline mükemmel görünür ve hiçbir model onu yenemezdi — kural işlevsizleşirdi.
    """
    y_train = np.array([0.0])
    y_test = np.array([5.0, 6.0, 7.0])
    pred = persistence_forecast(y_train, y_test)
    assert not np.any(pred == y_test)  # hiçbir konumda eşzamanlı değer kullanılmadı
    assert pred[1:].tolist() == y_test[:-1].tolist()


def test_persistence_length_matches_test_length():
    for n in (1, 2, 7):
        pred = persistence_forecast(np.array([3.0]), np.arange(n, dtype=float))
        assert pred.shape == (n,)


def test_persistence_casts_integer_counts_to_float():
    pred = persistence_forecast(np.array([2]), np.array([4, 6]))
    assert pred.dtype == np.float64  # sayım verisi gelse de metrikler float bekler


@pytest.mark.parametrize(
    "fn",
    [
        lambda: climatology_forecast(np.array([]), 3),
        lambda: persistence_forecast(np.array([]), np.array([1.0, 2.0])),
    ],
)
def test_empty_history_raises_instead_of_producing_nan_baseline(fn):
    """Boş geçmiş SESSİZ NaN değil, açık hata vermeli (fail-closed).

    np.mean([]) = NaN idi; NaN baseline skill_score üzerinden NaN beceriye yayılır ve
    NaN her `<=` karşılaştırmasını False yaptığı için dürüstlük kapılarından
    "baseline yenildi" gibi geçerdi.
    """
    with pytest.raises(ValueError, match="Boş eğitim geçmişi"):
        fn()


def test_nan_baseline_would_have_faked_a_positive_skill_claim():
    """Regresyon bekçisi: NaN baseline'ın neden tehlikeli olduğunu sabitler.

    Bu test baseline'ı ÇAĞIRMAZ; NaN bir metriğin kapı mantığında nasıl davrandığını
    gösterir — `> 0` testi de `<= 0` testi de False döner, yani skor ne kabul ne de
    reddedilir. Kapıların NaN'ı ayrıca elemesi bu yüzden şart.
    """
    nan_skill = skill_score(model_metric=1.0, baseline_metric=float("nan"))
    assert np.isnan(nan_skill)
    assert not (nan_skill > 0.0)   # "yendi" denemez
    assert not (nan_skill <= 0.0)  # ama saf eşik testi de reddetmez → fail-open riski


def test_climatology_is_beatable_but_honest_on_a_trending_series():
    """Dürüstlük duyarlılığı: sabit baseline, trendli seride yenilebilir olmalı."""
    y_train = np.array([1.0, 1.0, 1.0, 1.0])
    y_test = np.array([5.0, 6.0, 7.0])
    base = climatology_forecast(y_train, len(y_test))
    better = np.array([5.0, 6.0, 7.0])  # kusursuz model
    gain = skill_score(score(y_test, better)["mae"], score(y_test, base)["mae"])
    assert gain == pytest.approx(1.0)  # baseline yenilebilir → kural işliyor
