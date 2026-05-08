document.addEventListener("DOMContentLoaded", async () => {

  const substationSelect = document.getElementById("substationSelect");
  const feederPills = document.getElementById("feederPills");
  let currentFilters = {
    substation: null,
    feeder: null
  };

  /* ========================
     LOAD FILTERS FROM BACKEND
  ======================== */
  async function loadFilters() {
    const filters = await ApiService.getFilters();
    
    // Populate substations dropdown
    substationSelect.innerHTML = '';
    filters.substations.forEach(substation => {
      const option = document.createElement("option");
      option.value = substation;
      option.textContent = substation;
      substationSelect.appendChild(option);
    });

    if (filters.substations.length > 0) {
      substationSelect.value = filters.substations[0];
      currentFilters.substation = filters.substations[0];
    }
  }

  /* ========================
     LOAD AND UPDATE KPIs
  ======================== */
  async function updateKpis(substation, feeder) {
    const kpis = await ApiService.getKpis(substation, feeder);
    
    Object.keys(kpis).forEach(key => {
      const el = document.getElementById(key);
      if (el && typeof kpis[key] === 'number') {
        el.textContent = kpis[key].toFixed(2);
      }
    });
  }

  /* ========================
     RENDER FEEDERS
  ======================== */
  async function renderFeeders(substation) {
    feederPills.innerHTML = "";
    
    const filters = await ApiService.getFilters();
    const feeders = filters.feeders_by_substation[substation] || [];
    
    feeders.forEach((feeder, index) => {
      const pill = document.createElement("div");
      pill.className = index === 0 ? "pill active" : "pill";
      pill.textContent = feeder;

      pill.addEventListener("click", () => {
        document.querySelectorAll(".pill").forEach(p => {
          p.classList.remove("active");
        });
        pill.classList.add("active");
        currentFilters.feeder = feeder;
        updateDashboard(substation, feeder);
      });

      feederPills.appendChild(pill);
    });

    if (feeders.length > 0) {
      currentFilters.feeder = feeders[0];
    }
  }

  /* ========================
     UPDATE DASHBOARD
  ======================== */
  async function updateDashboard(substation, feeder) {
    await updateKpis(substation, feeder);
    await createCharts(substation, feeder);
    updateMap(substation);
  }

  /* ========================
     INITIAL LOAD
  ======================== */
  try {
    await loadFilters();
    const initialSubstation = substationSelect.value;
    await renderFeeders(initialSubstation);
    await updateDashboard(initialSubstation, null);
  } catch (error) {
    console.error('Error loading initial data:', error);
  }

  /* ========================
     SUBSTATION CHANGE
  ======================== */
  substationSelect.addEventListener("change", async (e) => {
    const selectedSubstation = e.target.value;
    currentFilters.substation = selectedSubstation;
    currentFilters.feeder = null;
    await renderFeeders(selectedSubstation);
    await updateDashboard(selectedSubstation, null);
  });

  /* ========================
     REFRESH
  ======================== */
  document.getElementById("refreshBtn").addEventListener("click", async () => {
    const selectedSubstation = substationSelect.value;
    await updateDashboard(selectedSubstation, currentFilters.feeder);
  });

  /* ========================
     UPDATE CLOCK
  ======================== */
  function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString();
    const dateString = now.toLocaleDateString();
    document.getElementById("clock").textContent = `${dateString} ${timeString}`;
  }
  
  updateClock();
  setInterval(updateClock, 1000);

});