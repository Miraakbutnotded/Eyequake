/* EyeQuake — sismik risk haritası ön yüzü.
 * Statik GeoJSON (web/data/) tüketir; backend yok. */

const TURKEY_CENTER = [39.0, 35.0];
const RISK_STOPS = [
  { min: 85, color: "#d10000", band: "Çok yüksek" },
  { min: 70, color: "#ff3b00", band: "Yüksek" },
  { min: 55, color: "#ff8c00", band: "Orta-yüksek" },
  { min: 40, color: "#ffd500", band: "Orta" },
  { min: 0, color: "#6b6b1a", band: "Düşük" },
];

function riskStyle(idx) {
  const stop = RISK_STOPS.find((s) => idx >= s.min) || RISK_STOPS[RISK_STOPS.length - 1];
  return stop;
}

function fmt(n, d = 2) {
  return Number(n).toLocaleString("tr-TR", { maximumFractionDigits: d });
}

const map = L.map("map", { zoomControl: true }).setView(TURKEY_CENTER, 6);

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  {
    attribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
    subdomains: "abcd",
    maxZoom: 19,
  }
).addTo(map);

// --- Panel güncelleme ---
const panelEmpty = document.getElementById("panelEmpty");
const panelContent = document.getElementById("panelContent");

function showCell(props, latlng) {
  panelEmpty.classList.add("hidden");
  panelContent.classList.remove("hidden");

  const stop = riskStyle(props.risk_index);
  document.getElementById("scoreValue").textContent = props.risk_index;
  document.getElementById("scoreValue").style.color = stop.color;
  document.getElementById("scoreFill").style.width = props.risk_index + "%";
  document.getElementById("scoreFill").style.background = stop.color;
  document.getElementById("scoreBand").textContent = stop.band + " risk";
  document.getElementById("scoreBand").style.color = stop.color;

  document.getElementById("mLoc").textContent =
    `${fmt(latlng.lat, 2)}°K, ${fmt(latlng.lng, 2)}°D`;
  document.getElementById("mRate").textContent = `${fmt(props.annual_rate)} /yıl`;
  document.getElementById("mMax").textContent = `M ${fmt(props.max_mag, 1)}`;
  document.getElementById("mN").textContent = props.n_events;
  document.getElementById("mRecent").textContent = `${props.n_recent} olay`;

  document.getElementById("note").textContent =
    "Bu indeks geçmiş (1990+) deprem aktivitesinin Türkiye genelindeki yüzdelik " +
    "sıralamasıdır. Gelecekteki bir depremin zamanını/büyüklüğünü tahmin etmez.";
}

// --- Hazard katmanı ---
async function load() {
  try {
    const [meta, hazard, events] = await Promise.all([
      fetch("./data/meta.json").then((r) => r.json()),
      fetch("./data/hazard_cells.geojson").then((r) => r.json()),
      fetch("./data/big_events.geojson").then((r) => r.json()),
    ]);

    document.getElementById("meta").textContent =
      `${hazard.features.length} hücre · ${events.features.length} büyük deprem (M≥6) · güncellendi ${meta.generated}`;

    L.geoJSON(hazard, {
      style: (f) => ({
        fillColor: riskStyle(f.properties.risk_index).color,
        fillOpacity: 0.55,
        weight: 0.3,
        color: "#000",
      }),
      onEachFeature: (feature, layer) => {
        layer.on({
          click: (e) => showCell(feature.properties, e.latlng),
          mouseover: (e) => e.target.setStyle({ weight: 1.5, color: "#fff", fillOpacity: 0.8 }),
          mouseout: (e) =>
            e.target.setStyle({ weight: 0.3, color: "#000", fillOpacity: 0.55 }),
        });
      },
    }).addTo(map);

    L.geoJSON(events, {
      pointToLayer: (f, latlng) =>
        L.circleMarker(latlng, {
          radius: 4 + (f.properties.mag - 6) * 3,
          color: "#00e5ff",
          weight: 1.5,
          fillColor: "#00e5ff",
          fillOpacity: 0.15,
        }).bindPopup(
          `<b>M ${f.properties.mag}</b> — ${f.properties.date}<br>${f.properties.place}`
        ),
    }).addTo(map);

    addLegend();
  } catch (err) {
    document.getElementById("meta").textContent =
      "Veri yüklenemedi — yerel sunucu üzerinden açıldığından emin ol (file:// çalışmaz).";
    console.error("[EyeQuake] veri yükleme hatası:", err);
  }
}

function addLegend() {
  const legend = L.control({ position: "bottomright" });
  legend.onAdd = () => {
    const div = L.DomUtil.create("div", "legend");
    div.innerHTML =
      "<b>Göreli risk indeksi</b>" +
      RISK_STOPS.map((s) => `<i style="background:${s.color}"></i>${s.band} (${s.min}+)`).join("<br>") +
      '<br><i style="background:#00e5ff;border-radius:50%"></i>M≥6 tarihsel';
    return div;
  };
  legend.addTo(map);
}

load();
