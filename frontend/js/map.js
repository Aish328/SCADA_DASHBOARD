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

  map = L.map('map').setView([lat, lng], 11);

  L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
      attribution: 'SCADA Dashboard © OpenStreetMap contributors'
    }
  ).addTo(map);

  marker = L.marker([lat, lng])
    .addTo(map)
    .bindPopup(label)
    .openPopup();
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