/**
 * map.js — Leaflet map with automatic geocoding for substation names.
 *
 * Priority order for coordinates:
 *   1. SUBSTATION_COORDS lookup below  (add your real lat/lng here)
 *   2. Nominatim (OpenStreetMap) geocoder — fires automatically for unknowns
 *   3. Deterministic fallback inside India if geocoding fails
 */

let _map          = null;
let _markerLayer  = null;
const _geoCache   = {};   // name → [lat, lng]

// ── 1. Known coordinates ─────────────────────────────────────────────────────
// Add entries here matching your exact substation names (case-sensitive).
// Example:
//   "Shahdara 220kV":  [28.6692, 77.2942],
//   "Mundka 66kV":     [28.6815, 76.9986],
const SUBSTATION_COORDS = {
  // "MY_SUB": [lat, lng],
};

// ── 2. Geocode via Nominatim ─────────────────────────────────────────────────
async function _geocode(name) {
  if (_geoCache[name]) return _geoCache[name];

  // Try the raw name first, then append "substation India" for better hits
  const queries = [
    `${name} substation India`,
    `${name} India`,
    name,
  ];

  for (const q of queries) {
    try {
      const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`;
      const res = await fetch(url, { headers: { "Accept-Language": "en" } });
      const data = await res.json();
      if (data && data.length > 0) {
        const coords = [parseFloat(data[0].lat), parseFloat(data[0].lon)];
        _geoCache[name] = coords;
        return coords;
      }
    } catch (_) { /* network error — try next query */ }
  }
  return null;
}

// ── 3. Deterministic fallback inside India ───────────────────────────────────
function _fallbackCoords(name) {
  const hash = [...(name || "X")].reduce((a, c) => a + c.charCodeAt(0), 0);
  return [
    18.0 + (hash % 14) + (hash % 7) * 0.1,    // lat  18–32 N  (India range)
    73.0 + ((hash * 13) % 24) + (hash % 5) * 0.1, // lng 73–97 E
  ];
}

// ── Map initialisation ───────────────────────────────────────────────────────
function initMap() {
  if (_map) return;
  _map = L.map("map", { zoomControl: true }).setView([22.5, 82.3], 5);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: "© OpenStreetMap © CARTO",
    maxZoom: 19,
  }).addTo(_map);
  _markerLayer = L.layerGroup().addTo(_map);
}

function _makeIcon(color = "#00e5d4") {
  return L.divIcon({
    className: "",
    html: `<div style="
      width:14px;height:14px;
      background:${color};
      border:2px solid #fff;
      border-radius:50%;
      box-shadow:0 0 10px ${color};
    "></div>`,
    iconSize: [14, 14],
    iconAnchor: [7, 7],
  });
}

// ── Public: update map for a given substation ────────────────────────────────
async function updateMap(substation) {
  if (!_map) initMap();
  _markerLayer.clearLayers();

  const label = substation || "All Substations";

  // Resolve coordinates
  let coords = SUBSTATION_COORDS[substation] || null;

  if (!coords && substation) {
    // Show a loading marker while geocoding
    const provisional = _fallbackCoords(substation);
    const loadingIcon = _makeIcon("#888");
    L.marker(provisional, { icon: loadingIcon })
      .bindPopup(`<b style="color:#aaa">${label}</b><br><i>Locating…</i>`)
      .addTo(_markerLayer);
    _map.setView(provisional, 6);

    coords = await _geocode(substation);

    // Remove provisional marker
    _markerLayer.clearLayers();
  }

  if (!coords) {
    coords = substation ? _fallbackCoords(substation) : [22.5, 82.3];
  }

  _geoCache[substation || "DEFAULT"] = coords;

  L.marker(coords, { icon: _makeIcon() })
    .bindPopup(`<b style="color:#00e5d4">${label}</b>`)
    .addTo(_markerLayer)
    .openPopup();

  _map.setView(coords, substation ? 10 : 5);
}

// ── Show all substations at once ─────────────────────────────────────────────
async function updateMapAll(substations) {
  if (!_map) initMap();
  _markerLayer.clearLayers();
  if (!substations || substations.length === 0) return;

  const bounds = [];

  await Promise.all(substations.map(async (name) => {
    let coords = SUBSTATION_COORDS[name] || _geoCache[name] || null;
    if (!coords) {
      coords = await _geocode(name);
      if (!coords) coords = _fallbackCoords(name);
      _geoCache[name] = coords;
    }
    L.marker(coords, { icon: _makeIcon() })
      .bindPopup(`<b style="color:#00e5d4">${name}</b>`)
      .addTo(_markerLayer);
    bounds.push(coords);
  }));

  if (bounds.length === 1) {
    _map.setView(bounds[0], 10);
  } else if (bounds.length > 1) {
    _map.fitBounds(bounds, { padding: [40, 40] });
  }
}

document.addEventListener("DOMContentLoaded", initMap);