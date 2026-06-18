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
const RJ = { a: -1.67, b: 1.0, p: 1.08, c: 0.05 };

let recentLayer = null;
let recentVisible = false;

// --- Helpers ---
function riskStyle(idx) {
  return RISK_STOPS.find((s) => idx >= s.min) || RISK_STOPS[RISK_STOPS.length - 1];
}

function fmt(n, d = 2) {
  return Number(n).toLocaleString("tr-TR", { maximumFractionDigits: d });
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
  document.getElementById("note").textContent    =
    "Bu indeks geçmiş (1990+) deprem aktivitesinin Türkiye genelindeki yüzdelik " +
    "sıralamasıdır. Gelecekteki bir depremin zamanını/büyüklüğünü tahmin etmez.";
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
          <span>${Math.round(n3_7)}</span>
          <span>${Math.round(n3_30)}</span>
          <span class="${hasRemain ? "etas-rem" : "etas-dim"}">${hasRemain ? Math.round(n3_rem) : "—"}</span>
        </div>
        <div class="etas-row">
          <span class="etas-dim">M≥4</span>
          <span>${Math.round(n4_7)}</span>
          <span>${Math.round(n4_30)}</span>
          <span class="${hasRemain ? "etas-rem" : "etas-dim"}">${hasRemain ? Math.round(n4_rem) : "—"}</span>
        </div>
      </div>
      ${hasRemain ? `
      <div class="etas-prog-wrap">
        <div class="etas-prog-fill" style="width:${pct}%"></div>
      </div>
      <div class="etas-prog-label">${fmt(elapsed, 1)} / 30 gün geçti · kalan ${fmt(remainDays, 1)} gün</div>
      ` : ""}
    `;
  } else {
    etasBlock.classList.add("hidden");
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
    const [meta, hazard, events] = await Promise.all([
      fetch("./data/meta.json").then((r) => r.json()),
      fetch("./data/hazard_cells.geojson").then((r) => r.json()),
      fetch("./data/big_events.geojson").then((r) => r.json()),
    ]);

    document.getElementById("meta").textContent =
      `${hazard.features.length} hücre · ${events.features.length} büyük deprem (M≥6) · güncellendi ${meta.generated}`;
    const srcEl = document.getElementById("src");
    if (srcEl) srcEl.textContent = meta.source || "AFAD";

    L.geoJSON(hazard, {
      filter: (f) => f.properties.risk_index >= 40, // hide "Düşük" cells for a cleaner map
      style: (f) => ({
        fillColor:   riskStyle(f.properties.risk_index).color,
        fillOpacity: 0.55,
        weight:      0.3,
        color:       "#000",
      }),
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

function addLegend() {
  const hazard = L.control({ position: "bottomright" });
  hazard.onAdd = () => {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML =
      "<b>Göreli risk indeksi (hücreler)</b>" +
      RISK_STOPS.filter((s) => s.min > 0)
        .map((s) => `<i style="background:${s.color}"></i>${s.band} (${s.min}+)`)
        .join("<br>") +
      '<br><hr style="border-color:#333;margin:6px 0">' +
      '<i style="background:#00e5ff;border-radius:50%"></i>M≥6 tarihsel';
    return div;
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
