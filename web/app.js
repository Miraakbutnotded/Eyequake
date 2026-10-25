/* EyeQuake — sismik risk haritası ön yüzü.
 * Statik GeoJSON (web/data/) + USGS canlı feed tüketir; backend yok. */

const TURKEY_CENTER = [39.0, 35.0];
const RISK_STOPS = [
  { min: 85, color: "#d10000", band: "Çok yüksek" },
  { min: 70, color: "#ff3b00", band: "Yüksek" },
  { min: 55, color: "#ff8c00", band: "Orta-yüksek" },
  { min: 40, color: "#ffd500", band: "Orta" },
  { min: 0,  color: "#6b6b1a", band: "Düşük" },
];

const USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query";
const TR = { minlat: 35.5, maxlat: 42.5, minlon: 25.5, maxlon: 45.0 };

// Reasenberg-Jones aftershock model (generic Turkey defaults)
// N(M≥m, t1→t2) = 10^(a + b*(M-m)) · ∫(t+c)^(-p)dt
// etas_params.json'dan yüklenir (provenance'lı tek doğruluk kaynağı); fetch başarısızsa fallback.
let RJ = { a: -1.67, b: 1.0, p: 1.08, c: 0.05 };

// Generic R-J artçı sayıları yüksek belirsizlik taşır → sahte-kesinlik veren tek sayı
// yerine kaba ARALIK göster (~0.4×–2.3×).
function etasBand(n) {
  if (n < 0.5) return "~0";
  return `${Math.round(n * 0.4)}–${Math.round(Math.max(1, n * 2.3))}`;
}

let recentLayer = null;
let recentVisible = false;

// --- Hücre renklendirme metrikleri (BIS zemin katmanı) ---
let hazardLayer = null;
let currentMetric = "risk"; // risk | amp | liq | site_risk

function ampColor(a) {
  if (a >= 2.0) return "#54278f";
  if (a >= 1.6) return "#756bb1";
  if (a >= 1.3) return "#9e9ac8";
  if (a >= 1.1) return "#cbc9e2";
  return "#f2f0f7";
}
const LIQ_COLOR = { "yüksek": "#d10000", "orta": "#ff8c00", "düşük": "#ffd500", "çok düşük": "#6b6b1a" };

function cellColor(p) {
  if (currentMetric === "amp") return ampColor(p.amp ?? 1);
  if (currentMetric === "liq") return LIQ_COLOR[p.liq_class] || "#333";
  if (currentMetric === "site_risk") return riskStyle(p.site_risk ?? 0).color;
  return riskStyle(p.risk_index).color;
}

function cellStyle(f) {
  return { fillColor: cellColor(f.properties), fillOpacity: 0.55, weight: 0.3, color: "#000" };
}

// --- Helpers ---
function riskStyle(idx) {
  return RISK_STOPS.find((s) => idx >= s.min) || RISK_STOPS[RISK_STOPS.length - 1];
}

function fmt(n, d = 2) {
  return Number(n).toLocaleString("tr-TR", { maximumFractionDigits: d });
}

function hasValue(v) {
  return v !== null && v !== undefined && Number.isFinite(Number(v));
}

function etasExpected(mainMag, minMag, t1Days, t2Days) {
  const { a, b, p, c } = RJ;
  const prod     = Math.pow(10, a + b * (mainMag - minMag));
  const integral = (Math.pow(t2Days + c, 1 - p) - Math.pow(t1Days + c, 1 - p)) / (1 - p);
  return Math.max(0, prod * integral);
}

function elapsedDays(tsMs) {
  return (Date.now() - tsMs) / 86_400_000;
}

function formatElapsed(days) {
  if (days < 1 / 60)  return "az önce";
  if (days < 1 / 24)  return `${Math.round(days * 1440)} dakika önce`;
  if (days < 1)       return `${Math.round(days * 24)} saat önce`;
  if (days < 2)       return "1 gün önce";
  return `${Math.floor(days)} gün önce`;
}

// Distinct blue/violet palette so recent-quake dots never blend into the
// red/orange/yellow/green hazard grid underneath them. White stroke gives
// contrast against any cell color.
function recentMagStyle(mag) {
  if (mag >= 5.5) return { color: "#ff2dd4", radius: 9 + (mag - 5.5) * 4 };
  if (mag >= 4.5) return { color: "#a855f7", radius: 7.5 };
  if (mag >= 3.5) return { color: "#3b82f6", radius: 6 };
  return           { color: "#60a5fa",  radius: 4 };
}

// --- Panel state ---
function setMode(mode) {
  document.getElementById("panelEmpty").classList.toggle("hidden",   mode !== "empty");
  document.getElementById("panelContent").classList.toggle("hidden", mode !== "cell");
  document.getElementById("panelEvent").classList.toggle("hidden",   mode !== "event");
}

function resetPanel() { setMode("empty"); }

// --- Map init ---
const map = L.map("map", { zoomControl: true }).setView(TURKEY_CENTER, 6);

L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
  attribution:
    '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: "abcd",
  maxZoom: 19,
}).addTo(map);

// --- Hazard cell panel ---
function showCell(props, latlng) {
  setMode("cell");
  const stop = riskStyle(props.risk_index);
  document.getElementById("scoreValue").textContent        = props.risk_index;
  document.getElementById("scoreValue").style.color        = stop.color;
  document.getElementById("scoreFill").style.width         = props.risk_index + "%";
  document.getElementById("scoreFill").style.background    = stop.color;
  document.getElementById("scoreBand").textContent         = stop.band + " risk";
  document.getElementById("scoreBand").style.color         = stop.color;
  document.getElementById("mLoc").textContent    = `${fmt(latlng.lat, 2)}°K, ${fmt(latlng.lng, 2)}°D`;
  document.getElementById("mRate").textContent   = `${fmt(props.annual_rate)} /yıl`;
  document.getElementById("mMax").textContent    = `M ${fmt(props.max_mag, 1)}`;
  document.getElementById("mN").textContent      = props.n_events;
  document.getElementById("mRecent").textContent = `${props.n_recent} olay`;
  document.getElementById("mEnergy").textContent =
    hasValue(props.energy_index) ? `log10 ${fmt(props.energy_index, 2)}` : "—";
  document.getElementById("mRecency").textContent =
    hasValue(props.recency_days) ? `${fmt(props.recency_days, 0)} gün önce` : "—";
  document.getElementById("mShallow").textContent =
    hasValue(props.shallow_fraction) ? `%${fmt(props.shallow_fraction * 100, 0)}` : "—";
  document.getElementById("mRiskModel").textContent =
    props.risk_model === "v2_afad_multi_signal" ? "AFAD çok-sinyalli v2" : (props.risk_model || "—");
  document.getElementById("mSiteClass").textContent =
    props.site_class ? `${props.site_class} (Vs30 ${props.vs30})` : "—";
  document.getElementById("mAmp").textContent = props.amp ? `${fmt(props.amp, 2)}×` : "—";
  document.getElementById("mLiq").textContent = props.liq_class || "—";
  document.getElementById("mSiteRisk").textContent = props.site_risk ?? "—";
  document.getElementById("note").textContent    =
    "Bu v2 indeks AFAD geçmişinden türetilen oran, büyüklük, enerji, güncellik, " +
    "sığlık ve saha sinyallerinin Türkiye genelindeki yüzdelik sıralamasıdır. " +
    "Gelecekteki bir depremin zamanını/büyüklüğünü tahmin etmez.";
}

// --- Recent event panel ---
function showRecentEvent(props, coords) {
  setMode("event");
  const [lon, lat, depthRaw] = coords;
  const depth   = depthRaw ?? 0;
  const elapsed = elapsedDays(props.time);
  const mag     = props.mag ?? 0;
  const st      = recentMagStyle(mag);

  document.getElementById("evMag").textContent    = `M ${fmt(mag, 1)}`;
  document.getElementById("evMag").style.color    = st.color;
  document.getElementById("evPlace").textContent  = props.place || "Bilinmiyor";
  document.getElementById("evTime").textContent   = formatElapsed(elapsed);
  document.getElementById("evDepth").textContent  = `${fmt(depth, 1)} km`;
  document.getElementById("evCoord").textContent  = `${fmt(lat, 2)}°K, ${fmt(lon, 2)}°D`;
  document.getElementById("evElapsed").textContent = `${fmt(elapsed, 1)} gün`;

  const etasBlock = document.getElementById("etasBlock");
  if (mag >= 4.5) {
    etasBlock.classList.remove("hidden");
    const n3_7  = etasExpected(mag, 3.0, 0, 7);
    const n3_30 = etasExpected(mag, 3.0, 0, 30);
    const n4_7  = etasExpected(mag, 4.0, 0, 7);
    const n4_30 = etasExpected(mag, 4.0, 0, 30);
    const hasRemain = elapsed < 30;
    const remainDays = Math.max(0, 30 - elapsed);
    const n3_rem = hasRemain ? etasExpected(mag, 3.0, elapsed, 30) : 0;
    const n4_rem = hasRemain ? etasExpected(mag, 4.0, elapsed, 30) : 0;
    const pct = Math.min(100, (elapsed / 30) * 100).toFixed(1);

    document.getElementById("etasGrid").innerHTML = `
      <div class="etas-table">
        <div class="etas-row etas-head">
          <span></span><span>7g</span><span>30g</span><span>Kalan</span>
        </div>
        <div class="etas-row">
          <span class="etas-dim">M≥3</span>
          <span>${etasBand(n3_7)}</span>
          <span>${etasBand(n3_30)}</span>
          <span class="${hasRemain ? "etas-rem" : "etas-dim"}">${hasRemain ? etasBand(n3_rem) : "—"}</span>
        </div>
        <div class="etas-row">
          <span class="etas-dim">M≥4</span>
          <span>${etasBand(n4_7)}</span>
          <span>${etasBand(n4_30)}</span>
          <span class="${hasRemain ? "etas-rem" : "etas-dim"}">${hasRemain ? etasBand(n4_rem) : "—"}</span>
        </div>
      </div>
      ${hasRemain ? `
      <div class="etas-prog-wrap">
        <div class="etas-prog-fill" style="width:${pct}%"></div>
      </div>
      <div class="etas-prog-label">${fmt(elapsed, 1)} / 30 gün geçti · kalan ${fmt(remainDays, 1)} gün</div>
      ` : ""}
      <div class="etas-caption">≈ kaba beklenti aralığı — <strong>kesin sayı değildir</strong>;
        gerçek değer birkaç kat değişebilir.</div>
      <details class="etas-details"><summary>nasıl hesaplanır + dürüst sınırlar</summary>
        Generic Reasenberg-Jones artçı modeli — <strong>deprem tahmini değildir</strong>.
        Projenin fitted ETAS'ı süper-kritik (n≫1, fiziksel-olmayan) olduğundan panelde
        kullanılmaz. <a href="./methodology.html" target="_blank">Tüm metodoloji + sınırlar →</a>
      </details>
    `;
  } else {
    etasBlock.classList.remove("hidden");
    document.getElementById("etasGrid").innerHTML =
      `<p class="note">M&lt;4.5 — anlamlı artçı dizisi beklenmez, tahmin gösterilmiyor.
       <strong>"Düşük gözlenen aktivite" güvenli demek DEĞİLDİR</strong> — bu konum yine
       yüksek sismik risk taşıyabilir (soldaki risk katmanına bak).
       <a href="./methodology.html" target="_blank">Metodoloji →</a></p>`;
  }
}

// --- Live recent earthquakes (USGS FDSN) ---
async function loadRecentQuakes() {
  const btn = document.getElementById("btnRecent");
  btn.disabled = true;
  btn.innerHTML = "⏳ Yükleniyor…";

  const start = new Date(Date.now() - 30 * 86_400_000).toISOString().split("T")[0];
  const url =
    `${USGS_API}?format=geojson&starttime=${start}` +
    `&minlatitude=${TR.minlat}&maxlatitude=${TR.maxlat}` +
    `&minlongitude=${TR.minlon}&maxlongitude=${TR.maxlon}` +
    `&minmagnitude=3.0&orderby=time`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (recentLayer) recentLayer.remove();

    recentLayer = L.geoJSON(data, {
      pointToLayer: (f, latlng) => {
        const st    = recentMagStyle(f.properties.mag);
        const fresh = elapsedDays(f.properties.time) < 1;
        return L.circleMarker(latlng, {
          radius:      st.radius,
          color:       "#ffffff",
          weight:      fresh ? 2.5 : 1.5,
          fillColor:   st.color,
          fillOpacity: fresh ? 0.95 : 0.85,
        });
      },
      onEachFeature: (f, layer) => {
        layer.on("click", (e) => {
          L.DomEvent.stopPropagation(e);
          showRecentEvent(f.properties, f.geometry.coordinates);
        });
        layer.on("mouseover", (e) => e.target.setStyle({ weight: 3, fillOpacity: 1 }));
        layer.on("mouseout",  (e) => {
          const fresh = elapsedDays(f.properties.time) < 1;
          e.target.setStyle({ weight: fresh ? 2.5 : 1.5, fillOpacity: fresh ? 0.95 : 0.85 });
        });
      },
    }).addTo(map);

    recentVisible = true;
    btn.innerHTML = `<span class="live-dot"></span> Son 30 Gün (${data.features.length})`;
    btn.classList.add("active");
    btn.disabled = false;
  } catch (err) {
    btn.textContent = "⚠ Yüklenemedi";
    btn.disabled = false;
    console.error("[EyeQuake] recent quakes fetch error:", err);
  }
}

function toggleRecentLayer() {
  if (!recentLayer) { loadRecentQuakes(); return; }
  const btn = document.getElementById("btnRecent");
  if (recentVisible) {
    recentLayer.remove();
    recentVisible = false;
    btn.classList.remove("active");
  } else {
    recentLayer.addTo(map);
    recentVisible = true;
    btn.classList.add("active");
  }
}

// --- Static layers + boot ---
async function load() {
  try {
    const [meta, hazard, events, etasParams] = await Promise.all([
      fetch("./data/meta.json").then((r) => r.json()),
      fetch("./data/hazard_cells.geojson").then((r) => r.json()),
      fetch("./data/big_events.geojson").then((r) => r.json()),
      fetch("./data/etas_params.json").then((r) => r.json()).catch(() => null),
    ]);
    if (etasParams?.rj_aftershock) RJ = etasParams.rj_aftershock; // hardcoded değil, provenance'lı

    document.getElementById("meta").textContent =
      `${hazard.features.length} hücre · ${events.features.length} büyük deprem (M≥6) · güncellendi ${meta.generated}`;
    const srcEl = document.getElementById("src");
    if (srcEl) srcEl.textContent = meta.source || "AFAD";

    hazardLayer = L.geoJSON(hazard, {
      filter: (f) => f.properties.risk_index >= 40, // hide "Düşük" cells for a cleaner map
      style: cellStyle,
      onEachFeature: (feature, layer) => {
        layer.on({
          click:     (e) => showCell(feature.properties, e.latlng),
          mouseover: (e) => e.target.setStyle({ weight: 1.5, color: "#fff", fillOpacity: 0.8 }),
          mouseout:  (e) => e.target.setStyle({ weight: 0.3, color: "#000", fillOpacity: 0.55 }),
        });
      },
    }).addTo(map);

    L.geoJSON(events, {
      pointToLayer: (f, latlng) =>
        L.circleMarker(latlng, {
          radius:      4 + (f.properties.mag - 6) * 3,
          color:       "#00e5ff",
          weight:      1.5,
          fillColor:   "#00e5ff",
          fillOpacity: 0.15,
        }).bindPopup(`<b>M ${f.properties.mag}</b> — ${f.properties.date}<br>${f.properties.place}`),
    }).addTo(map);

    addLegend();
  } catch (err) {
    document.getElementById("meta").textContent =
      "Veri yüklenemedi — yerel sunucu üzerinden açıldığından emin ol (file:// çalışmaz).";
    console.error("[EyeQuake] veri yükleme hatası:", err);
  }

  loadRecentQuakes();
}

let hazardLegendDiv = null;

const RISK_LEGEND = () =>
  RISK_STOPS.filter((s) => s.min > 0)
    .map((s) => `<i style="background:${s.color}"></i>${s.band} (${s.min}+)`)
    .join("<br>");

const METRIC_LEGENDS = {
  risk: () => "<b>Göreli risk indeksi</b>" + RISK_LEGEND(),
  site_risk: () => "<b>Saha-düzeltilmiş risk</b>" + RISK_LEGEND(),
  amp: () =>
    "<b>Zemin büyütmesi (×)</b>" +
    [["#54278f", "≥2.0 (E yumuşak)"], ["#756bb1", "1.6–2.0"], ["#9e9ac8", "1.3–1.6 (D)"],
     ["#cbc9e2", "1.1–1.3 (C)"], ["#f2f0f7", "~1.0 (B kaya)"]]
      .map(([c, l]) => `<i style="background:${c}"></i>${l}`).join("<br>"),
  liq: () =>
    "<b>Sıvılaşma yatkınlığı</b>" +
    [["#d10000", "yüksek"], ["#ff8c00", "orta"], ["#ffd500", "düşük"], ["#6b6b1a", "çok düşük"]]
      .map(([c, l]) => `<i style="background:${c}"></i>${l}`).join("<br>") +
    '<br><span class="legend-note">*tarama göstergesi, jeoteknik değil</span>',
};

function updateLegend() {
  if (!hazardLegendDiv) return;
  hazardLegendDiv.innerHTML =
    METRIC_LEGENDS[currentMetric]() +
    '<br><hr style="border-color:#333;margin:6px 0">' +
    '<i style="background:#00e5ff;border-radius:50%"></i>M≥6 tarihsel';
}

function setMetric(m) {
  currentMetric = m;
  if (hazardLayer) hazardLayer.setStyle(cellStyle);
  document.querySelectorAll(".metric-btn")
    .forEach((b) => b.classList.toggle("active", b.dataset.metric === m));
  updateLegend();
}

function addLegend() {
  const hazard = L.control({ position: "bottomright" });
  hazard.onAdd = () => {
    hazardLegendDiv = L.DomUtil.create("div", "legend");
    updateLegend();
    return hazardLegendDiv;
  };
  hazard.addTo(map);

  const recent = L.control({ position: "bottomleft" });
  recent.onAdd = () => {
    const div = L.DomUtil.create("div", "legend legend-recent");
    div.innerHTML =
      "<b>Son 30 gün (canlı)</b>" +
      '<i style="background:#ff2dd4;border-radius:50%;border:1px solid #fff"></i>M≥5.5<br>' +
      '<i style="background:#a855f7;border-radius:50%;border:1px solid #fff"></i>M 4.5–5.5<br>' +
      '<i style="background:#3b82f6;border-radius:50%;border:1px solid #fff"></i>M 3.5–4.5<br>' +
      '<i style="background:#60a5fa;border-radius:50%;border:1px solid #fff"></i>M &lt;3.5';
    return div;
  };
  recent.addTo(map);
}

load();
