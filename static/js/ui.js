document.addEventListener("DOMContentLoaded", async () => {

  const substationSelect = document.getElementById("substationSelect");
  const feederPills      = document.getElementById("feederPills");
  let currentFilters = { substation: null, feeder: null };

  /* ── Helpers ─────────────────────────────────────────────────────────── */
  function setKpiLoading() {
    document.querySelectorAll(".tile-value").forEach(el => {
      el.textContent = "…";
      el.classList.add("loading");
    });
  }

  function setVal(id, value, suffix = "") {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.remove("loading");
    if (value == null || value === "") { el.textContent = "—"; return; }
    if (typeof value === "number") {
      el.textContent = (Number.isInteger(value) ? value : value.toFixed(1)) + suffix;
    } else {
      el.textContent = value + suffix;
    }
  }

  /* ── Load filters ─────────────────────────────────────────────────────── */
  async function loadFilters() {
    const filters = await ApiService.getFilters();
    substationSelect.innerHTML = "";
    filters.substations.forEach(sub => {
      const opt = document.createElement("option");
      opt.value = sub; opt.textContent = sub;
      substationSelect.appendChild(opt);
    });
    if (filters.substations.length > 0) {
      substationSelect.value    = filters.substations[0];
      currentFilters.substation = filters.substations[0];
    }
    return filters;
  }

  /* ── Feeder pills ─────────────────────────────────────────────────────── */
  async function renderFeeders(substation) {
    feederPills.innerHTML = "";
    let feeders = [];
    try {
      const f = await ApiService.getFilters();
      feeders = f.feeders_by_substation[substation] || [];
    } catch(e) { console.warn(e); }

    const allPill = _pill("All", true, async () => {
      _activatePill(allPill);
      currentFilters.feeder = null;
      await updateDashboard(substation, null);
    });
    feederPills.appendChild(allPill);
    feeders.forEach(feeder => {
      const pill = _pill(feeder, false, async () => {
        _activatePill(pill);
        currentFilters.feeder = feeder;
        await updateDashboard(substation, feeder);
      });
      feederPills.appendChild(pill);
    });
    currentFilters.feeder = null;
  }

  function _pill(label, active, onClick) {
    const el = document.createElement("div");
    el.className = active ? "pill active" : "pill";
    el.textContent = label;
    el.addEventListener("click", onClick);
    return el;
  }
  function _activatePill(target) {
    document.querySelectorAll(".pill").forEach(p => p.classList.remove("active"));
    target.classList.add("active");
  }

  /* ── KPI update ───────────────────────────────────────────────────────── */
  async function updateKpis(substation, feeder) {
    setKpiLoading();
    const k = await ApiService.getKpis(substation, feeder);

    // Voltage section
    setVal("fvhi",  k.fvhi_count,  " surges");
    setVal("fvhd",  k.fvhd_total,  " min");
    setVal("fvli",  k.fvli_count,  " dips");
    setVal("fvld",  k.fvld_total,  " min");
    setVal("fvsm",  k.avg_voltage, " V");

    // Current section
    setVal("fchi",  k.fchi_count,  " surges");
    setVal("fchd",  k.fchd_total,  " min");
    setVal("fcli",  k.fcli_count,  " dips");
    setVal("fcld",  k.fcld_total,  " min");
    setVal("fcsm",  k.avg_current, " A");

    // Aggregated
    setVal("avg_feeder_voltage", k.avg_voltage, " V");
    setVal("avg_feeder_current", k.avg_current, " A");
    setVal("max_voltage",        k.max_voltage, " V");
    setVal("max_current",        k.max_current, " A");
    setVal("total_records",      k.total_records);
  }

  /* ── Full refresh ─────────────────────────────────────────────────────── */
  async function updateDashboard(substation, feeder) {
    await Promise.all([
      updateKpis(substation, feeder),
      createCharts(substation, feeder),
    ]);
    updateMap(substation);
  }

  /* ── Clock ────────────────────────────────────────────────────────────── */
  function updateClock() {
    const now = new Date();
    document.getElementById("clock").textContent =
      now.toLocaleDateString() + "  " + now.toLocaleTimeString();
  }
  updateClock();
  setInterval(updateClock, 1000);

  /* ── Init ─────────────────────────────────────────────────────────────── */
  try {
    await loadFilters();
    const sub = substationSelect.value;
    await renderFeeders(sub);
    await updateDashboard(sub, null);
  } catch(err) {
    console.error("Init error:", err);
    const banner = document.createElement("div");
    banner.style.cssText = "color:#ff5252;padding:16px;font-family:monospace;font-size:13px";
    banner.textContent = `⚠ API error: ${err.message}. Make sure FastAPI server is running.`;
    document.querySelector(".main").prepend(banner);
  }

  /* ── Events ───────────────────────────────────────────────────────────── */
  substationSelect.addEventListener("change", async e => {
    const sub = e.target.value;
    currentFilters.substation = sub;
    currentFilters.feeder     = null;
    await renderFeeders(sub);
    await updateDashboard(sub, null);
  });

  document.getElementById("refreshBtn").addEventListener("click", async () => {
    await updateDashboard(substationSelect.value, currentFilters.feeder);
  });

});