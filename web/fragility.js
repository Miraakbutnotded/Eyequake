/* EyeQuake — BIS Faz-1 bina kırılganlık katmanı (tarayıcı portu).
 *
 * src/eyequake/analysis/fragility.py'nin birebir JS karşılığı. Tüm parametreler ve
 * metinler web/data/fragility_params.json'dan gelir (Python'dan export edilir) — bu
 * dosyada parametre sayısı YOKTUR (tests/test_web_honesty.py bunu korur).
 * selfCheck() JSON'daki altın vektörleri yeniden hesaplar; sapma → form kapanır.
 * Göreli yapısal kırılganlık göstergesi; hasar/çökme tahmini değildir.
 */

(function (root) {
  "use strict";

  // Python round() eşleniği: tam ondalık değere göre en yakın, eşitlikte çift hane.
  // (Math.round yarımı yukarı yuvarlar; toFixed eşitlikte büyüğü seçer — ikisi de sapar.)
  function pyRound(x, d) {
    const exact = x.toFixed(Math.min(100, d + 70));
    const dot = exact.indexOf(".");
    const tail = exact.slice(dot + 1 + d);
    if (/^50*$/.test(tail)) {
      const kept = d === 0 ? exact.slice(0, dot) : exact.slice(0, dot + 1 + d);
      const last = Number(kept[kept.length - 1]);
      if (last % 2 === 0) return Number(kept);
      const step = Math.pow(10, -d) * Math.sign(x);
      return Number((Number(kept) + step).toFixed(d));
    }
    return Number(x.toFixed(d));
  }

  const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));

  const CARD_DEFAULTS = {
    structure_type: "unknown", year_built: null, n_floors: null, soft_story: null,
    irregularity: null, pounding_risk: null, prior_unrepaired_damage: null,
    amnesty_imar_barisi: null, retrofit: null, retrofit_year: null,
    source: "self_reported", evidence_ref: null,
  };

  function create(P) {
    function eraFromYear(year) {
      return P.eras[P.era_breaks.filter((b) => year >= b).length];
    }

    function validate(card, currentYear) {
      const errors = [];
      const now = currentYear;
      const { year_min, floors_min, floors_max } = P.validation;
      if (!P.structure_types.includes(card.structure_type)) {
        errors.push(["structure_type", `geçersiz değer: ${card.structure_type}`]);
      }
      if (!P.sources.includes(card.source)) {
        errors.push(["source", `geçersiz değer: ${card.source}`]);
      }
      if (card.year_built !== null && !(card.year_built >= year_min && card.year_built <= now)) {
        errors.push(["year_built", `[${year_min}, ${now}] aralığında olmalı`]);
      }
      if (card.n_floors !== null && !(card.n_floors >= floors_min && card.n_floors <= floors_max)) {
        errors.push(["n_floors", `[${floors_min}, ${floors_max}] aralığında olmalı`]);
      }
      if (card.retrofit_year !== null) {
        if (!card.retrofit) errors.push(["retrofit_year", "retrofit=True olmadan verilemez"]);
        const yearOk = card.year_built !== null &&
          card.year_built >= year_min && card.year_built <= now;
        const lo = yearOk ? card.year_built : year_min;
        if (!(card.retrofit_year >= lo && card.retrofit_year <= now)) {
          errors.push(["retrofit_year", `[${lo}, ${now}] aralığında olmalı`]);
        }
      }
      return errors;
    }

    function baseVulnerability(type, era) {
      const types = type in P.base ? [type] : Object.keys(P.base);
      const eraIdx = era !== null ? [P.eras.indexOf(era)] : P.eras.map((_, i) => i);
      const vals = [];
      for (const t of types) for (const i of eraIdx) vals.push(P.base[t][i]);
      let s = 0;
      for (const v of vals) s += v;
      return s / vals.length;
    }

    function completeness(card) {
      let total = 0, got = 0;
      for (const [name, w] of P.completeness_weights) {
        total += w;
        const provided = name === "structure_type"
          ? card.structure_type !== "unknown"
          : card[name] !== null;
        if (provided) got += w;
      }
      return got / total;
    }

    function vulnerabilityClass(v) {
      for (const [lower, cls] of P.class_bins) if (v >= lower) return cls;
      return P.class_fallback;
    }

    function assess(rawCard, currentYear) {
      const card = { ...CARD_DEFAULTS, ...rawCard };
      const errors = validate(card, currentYear);
      if (errors.length) return { errors };

      const era = card.year_built !== null ? eraFromYear(card.year_built) : null;
      const imputed = [];
      if (card.structure_type === "unknown") imputed.push("structure_type");
      if (era === null) imputed.push("year_built");

      let v = baseVulnerability(card.structure_type, era);
      if (card.n_floors !== null) {
        if (P.masonry.includes(card.structure_type)) {
          v += card.n_floors > P.masonry_floor.limit ? P.masonry_floor.penalty : 0;
        } else if (P.floor_penalty_types.includes(card.structure_type)) {
          v += Math.min(P.floor.cap, Math.max(0, P.floor.step * (card.n_floors - P.floor.base)));
        }
      }
      for (const [name, delta] of P.modifiers) if (card[name]) v += delta;
      if (card.retrofit) v -= P.retrofit_credit * P.retrofit_trust[card.source];
      v = pyRound(clip(v, P.v_bounds[0], P.v_bounds[1]), 3);

      const M = P.multiplier;
      const mRaw = pyRound(clip(M.intercept + M.slope * v, M.min, M.max), 2);
      const comp = pyRound(completeness(card), 2);
      const sourceTrust = P.source_trust[card.source];
      const kappa = pyRound(clip(comp * sourceTrust, 0, 1), 2);
      const K = P.kappa_thresholds;
      const resolution = kappa >= K.full ? "building_full"
        : kappa >= K.partial ? "building_partial" : "site_only";

      const warnings = [];
      const pl = P.plausibility;
      const fl = card.n_floors;
      if (P.masonry.includes(card.structure_type) && fl !== null && fl > pl.masonry_tall_floors) {
        warnings.push(pl.warn_masonry_tall);
      }
      if (era === P.eras[0] && fl !== null && fl >= pl.pre1975_tall_floors) {
        warnings.push(pl.warn_pre1975_tall);
      }

      return {
        errors: [],
        vulnerability_index: v,
        vulnerability_class: vulnerabilityClass(v),
        multiplier_raw: mRaw,
        completeness: comp,
        source_trust: sourceTrust,
        kappa,
        resolution,
        trust_flag: card.source,
        imputed_fields: imputed,
        plausibility_warnings: warnings,
        fragility_model_version: P.fragility_model_version,
      };
    }

    function overlay(siteRisk, rawCard, currentYear) {
      const siteOnly = (reason, assessment) => ({
        errors: [], parcel_risk: siteRisk, site_risk: siteRisk, applied: false,
        resolution: "site_only", multiplier_effective: 1, downward_capped: false,
        reason, assessment, notes: [P.site_only_note],
      });
      if (rawCard === null) return siteOnly("no_card", null);
      const a = assess(rawCard, currentYear);
      if (a.errors.length) return { errors: a.errors };
      if (a.resolution === "site_only") return siteOnly("low_confidence", a);

      let mEff = 1 + (a.multiplier_raw - 1) * a.kappa;
      let capped = false;
      if (a.trust_flag === "self_reported" && mEff < P.self_report_floor) {
        mEff = P.self_report_floor;
        capped = true;
      }
      mEff = pyRound(mEff, 3);
      const notes = [P.class_adjacency];
      if (a.trust_flag === "self_reported") notes.push(P.self_report_badge);
      return {
        errors: [],
        parcel_risk: clip(pyRound(siteRisk * mEff, 0), 0, 100),
        site_risk: siteRisk, applied: true, resolution: a.resolution,
        multiplier_effective: mEff, downward_capped: capped, reason: null,
        assessment: a, notes,
      };
    }

    // Altın vektörler: Python'un hesapladığı sonuçlar birebir tutmalı.
    function selfCheck() {
      const mismatches = [];
      P.golden.forEach((g, i) => {
        const o = overlay(g.site_risk, g.card, P.golden_year);
        if (o.errors.length) {
          mismatches.push(`#${i} doğrulama hatası: ${o.errors.map((e) => e[0]).join(",")}`);
          return;
        }
        const got = {
          parcel_risk: o.parcel_risk, applied: o.applied, resolution: o.resolution,
          multiplier_effective: o.multiplier_effective, downward_capped: o.downward_capped,
          vulnerability_index: o.assessment.vulnerability_index,
          vulnerability_class: o.assessment.vulnerability_class,
          multiplier_raw: o.assessment.multiplier_raw,
          completeness: o.assessment.completeness, kappa: o.assessment.kappa,
        };
        for (const [k, want] of Object.entries(g.expected)) {
          if (got[k] !== want) mismatches.push(`#${i} ${k}: ${got[k]} ≠ ${want}`);
        }
      });
      return mismatches;
    }

    return { assess, overlay, validate, selfCheck, eraFromYear, params: P };
  }

  const api = { create, pyRound };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EyeQuakeFragility = api;
})(typeof window !== "undefined" ? window : globalThis);
