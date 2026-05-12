let map;
let marker;

// Substation coordinates mapping
const SUBSTATION_COORDINATES = {
  "Sub1": { lat: 28.6139, lng: 77.2090, label: "Sub1 - Delhi" },
  "Sub2": { lat: 19.0760, lng: 72.8777, label: "Sub2 - Mumbai" },
  "Sub3": { lat: 12.9716, lng: 77.5946, label: "Sub3 - Bangalore" },
  "Sub4": { lat: 17.3850, lng: 78.4867, label: "Sub4 - Hyderabad" },
  "Sub5": { lat: 22.5726, lng: 88.3639, label: "Sub5 - Kolkata" }
};

function initializeMap(lat, lng, label) {
  if (map) {
    map.remove();
  }

  map = L.map('map', {
    layers: [],
    zoomControl: true,
    attributionControl: true
  }).setView([lat, lng], 4);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
  // Add a styled background
  const mapPane = map.getPane('mapPane');
  if (mapPane) {
    mapPane.style.background = '#0e0b20';
  }

  // Draw grid using rectangles
  const gridGroup = L.featureGroup().addTo(map);
  
  // Add semi-transparent background rectangles to represent regions
  const locations = [
    { lat: 28.6139, lng: 77.2090, size: 3, color: '#00e5d4', opacity: 0.3, label: 'Sub1' },
    { lat: 19.0760, lng: 72.8777, size: 3, color: '#9c6fff', opacity: 0.3, label: 'Sub2' },
    { lat: 12.9716, lng: 77.5946, size: 3, color: '#e040fb', opacity: 0.3, label: 'Sub3' },
    { lat: 17.3850, lng: 78.4867, size: 3, color: '#b2ff59', opacity: 0.3, label: 'Sub4' },
    { lat: 22.5726, lng: 88.3639, size: 3, color: '#00e5d4', opacity: 0.3, label: 'Sub5' }
  ];

  locations.forEach(loc => {
    L.circle([loc.lat, loc.lng], {
      radius: 200000,
      color: loc.color,
      weight: 1,
      opacity: 0.3,
      fill: true,
      fillColor: loc.color,
      fillOpacity: 0.1
    }).addTo(gridGroup);
  });

  // Add grid lines
  for (let gridLat = 0; gridLat <= 40; gridLat += 5) {
    L.polyline(
      [[gridLat, 60], [gridLat, 100]],
      { color: 'rgba(0, 229, 212, 0.1)', weight: 1 }
    ).addTo(gridGroup);
  }

  for (let gridLng = 60; gridLng <= 100; gridLng += 5) {
    L.polyline(
      [[0, gridLng], [40, gridLng]],
      { color: 'rgba(0, 229, 212, 0.1)', weight: 1 }
    ).addTo(gridGroup);
  }

  marker = L.marker([lat, lng], {
    title: label,
    icon: L.icon({
      iconUrl: 'data:image/svg+xml;charset=UTF-8,%3Csvg width="32" height="41" viewBox="0 0 32 41" fill="none" xmlns="http://www.w3.org/2000/svg"%3E%3Cpath d="M32 16C32 26.954 16 41 16 41S0 26.954 0 16C0 7.163 7.164 0 16 0C24.837 0 32 7.163 32 16Z" fill="%2300E5D422" stroke="%2300E5D4" stroke-width="2"/%3E%3Ccircle cx="16" cy="13" r="4" fill="%23080612" stroke="%2300E5D4" stroke-width="1"/%3E%3C/svg%3E',
      iconSize: [32, 41],
      iconAnchor: [16, 41],
      popupAnchor: [0, -41],
      className: 'leaflet-marker-icon'
    })
  })
    .addTo(map)
    .bindPopup(label, { autoClose: false, closeButton: true })
    .openPopup();

  setTimeout(() => {
    map.invalidateSize();
  }, 100);
}

function updateMap(substation) {
  const coords = SUBSTATION_COORDINATES[substation];
  
  if (coords) {
    initializeMap(coords.lat, coords.lng, coords.label);
  } else {
    // Fallback to Sub1 if substation not found
    const defaultCoords = SUBSTATION_COORDINATES["Sub1"];
    initializeMap(defaultCoords.lat, defaultCoords.lng, defaultCoords.label);
  }
}