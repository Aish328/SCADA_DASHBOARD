let map;
let marker;

function initializeMap(lat,lng,label){

  if(map){

    map.remove();

  }

  map = L.map('map').setView([lat,lng],11);

  L.tileLayer(
    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    {
      attribution:'SCADA Dashboard'
    }
  ).addTo(map);

  marker = L.marker([lat,lng])
    .addTo(map)
    .bindPopup(label)
    .openPopup();

}

function updateMap(city){

  const cityData = SCADA_DATA[city];

  initializeMap(
    cityData.coordinates.lat,
    cityData.coordinates.lng,
    city + " Substation"
  );

}