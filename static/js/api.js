/**
 * ApiService — all backend calls for the SCADA dashboard.
 * Assumes the FastAPI server is running on the same origin.
 */
console.log("API JS LOADED");

async function fetchKPIData() {
    console.log("Fetching KPI data...");
    try{
        const response = await fetch("/kpi/");   
        console.log("reposnse received");
        const data = await response.json();
        console.log("Data received:", data);
    } catch (error) {
        console.error("Error fetching KPI data:", error);
        const data = null;
    }
    

    // console.log(data);
}

fetchKPIData();
const ApiService = (() => {

  const BASE = "";   // same-origin; change to "http://localhost:8000" if running separately

  async function _get(path) {
    const res = await fetch(BASE + path);
    if (!res.ok) throw new Error(`API error ${res.status} on ${path}`);
    return res.json();
  }

  return {
    /** /filters/ */
    getFilters() {
      return _get("/filters/");
    },

    /** /kpi/?substation=...&feeder=... */
    getKpis(substation, feeder, limit = 96) {
      const p = new URLSearchParams({ limit });
      if (substation) p.set("substation", substation);
      if (feeder)     p.set("feeder", feeder);
      return _get(`/kpi/?${p}`);
      
    },

    /** /data/series?substation=...&feeder=...&limit=24 */
    getChartSeries(substation, feeder, limit = 96) {
      const p = new URLSearchParams({ limit });
      if (substation) p.set("substation", substation);
      if (feeder)     p.set("feeder", feeder);
      return _get(`/data/series?${p}`);
    },

    /** /spikes/?spike_type=all&substation=...&feeder=... */
    getSpikes(substation, feeder, spikeType = "all") {
      const p = new URLSearchParams({ spike_type: spikeType });
      if (substation) p.set("substation", substation);
      if (feeder)     p.set("feeder", feeder);
      return _get(`/spikes/?${p}`);
    },
  };

})();