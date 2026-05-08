// API Configuration
const API_BASE_URL = "http://localhost:8000";

// API Service Object
const ApiService = {
  
  /**
   * Fetch all filters (substations and feeders)
   */
  async getFilters() {
    try {
      const response = await fetch(`${API_BASE_URL}/filters/`);
      if (!response.ok) throw new Error('Failed to fetch filters');
      return await response.json();
    } catch (error) {
      console.error('Error fetching filters:', error);
      return { substations: [], feeders: [], feeders_by_substation: {} };
    }
  },

  /**
   * Fetch KPI data
   */
  async getKpis(substation = null, feeder = null) {
    try {
      let url = `${API_BASE_URL}/kpi/`;
      const params = new URLSearchParams();
      
      if (substation) params.append('substation', substation);
      if (feeder) params.append('feeder', feeder);
      
      if (params.toString()) url += `?${params.toString()}`;
      
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch KPIs');
      return await response.json();
    } catch (error) {
      console.error('Error fetching KPIs:', error);
      return {
        avg_feeder_voltage: 0,
        avg_feeder_current: 0,
        max_voltage: 0,
        max_current: 0,
        min_voltage: 0,
        min_current: 0,
        total_records: 0,
        substations: [],
        feeders: []
      };
    }
  },

  /**
   * Fetch spikes data
   */
  async getSpikes(spike_type = 'all', substation = null, feeder = null) {
    try {
      let url = `${API_BASE_URL}/spikes/`;
      const params = new URLSearchParams();
      
      params.append('spike_type', spike_type);
      if (substation) params.append('substation', substation);
      if (feeder) params.append('feeder', feeder);
      
      url += `?${params.toString()}`;
      
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch spikes');
      return await response.json();
    } catch (error) {
      console.error('Error fetching spikes:', error);
      return { spike_type: spike_type, total_spikes: 0, spikes: [] };
    }
  },

  /**
   * Fetch chart series data
   */
  async getChartData(substation = null, feeder = null, limit = 12) {
    try {
      let url = `${API_BASE_URL}/data/series`;
      const params = new URLSearchParams();
      if (substation) params.append('substation', substation);
      if (feeder) params.append('feeder', feeder);
      params.append('limit', limit);
      url += `?${params.toString()}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error('Failed to fetch chart data');
      return await response.json();
    } catch (error) {
      console.error('Error fetching chart data:', error);
      return {
        categories: [],
        voltage1: [],
        voltage2: [],
        voltage3: [],
        voltage4: [],
        current1: [],
        current2: [],
        current3: [],
        current4: [],
        trend1: [],
        trend2: [],
        trend3: []
      };
    }
  },

  /**
   * Fetch all data records
   */
  async getAllData(substation = null, feeder = null) {
    try {
      return await this.getKpis(substation, feeder);
    } catch (error) {
      console.error('Error fetching data:', error);
      return null;
    }
  }
};

// Mock data fallback (for development without backend running)
const MOCK_SCADA_DATA = {
  "Sub1": {
    kpis: {
      fvhi: 328,
      fvhd: 178,
      fvli: 118,
      fvld: 67,
      fvsm: 21,
      fchi: 316,
      fchd: 144,
      fcli: 56,
      fcld: 85,
      fcsm: 17
    },
    feeders: ["F1", "F2", "F3"],
    charts: {
      voltage1: [328, 315, 311, 311, 320, 325, 330, 335, 340, 335, 330, 325],
      voltage2: [178, 164, 167, 131, 140, 150, 160, 165, 170, 165, 160, 155],
      voltage3: [118, 109, 109, 86, 95, 105, 115, 125, 130, 125, 120, 115],
      voltage4: [67, 40, 81, 71, 60, 65, 70, 75, 80, 75, 70, 65],
      current1: [316, 344, 253, 345, 330, 340, 350, 360, 370, 365, 360, 355],
      current2: [144, 125, 142, 120, 130, 140, 150, 160, 170, 165, 160, 155],
      current3: [56, 133, 122, 100, 110, 120, 130, 140, 150, 145, 140, 135],
      current4: [85, 57, 28, 31, 40, 45, 50, 55, 60, 55, 50, 45],
      trend1: [300, 310, 320, 330, 340, 350, 360, 365, 370, 365, 360, 355],
      trend2: [150, 160, 170, 180, 190, 200, 210, 215, 220, 215, 210, 205],
      trend3: [100, 110, 120, 130, 140, 150, 160, 165, 170, 165, 160, 155]
    }
  },
  "Sub2": {
    kpis: {
      fvhi: 410,
      fvhd: 220,
      fvli: 140,
      fvld: 70,
      fvsm: 110,
      fchi: 370,
      fchd: 180,
      fcli: 120,
      fcld: 60,
      fcsm: 95
    },
    feeders: ["F1", "F2"],
    charts: {
      voltage1: [410, 405, 400, 395, 390, 385, 380, 375, 370, 365, 360, 355],
      voltage2: [220, 215, 210, 205, 200, 195, 190, 185, 180, 175, 170, 165],
      voltage3: [140, 135, 130, 125, 120, 115, 110, 105, 100, 95, 90, 85],
      voltage4: [70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15],
      current1: [370, 375, 380, 385, 390, 395, 400, 405, 410, 405, 400, 395],
      current2: [180, 185, 190, 195, 200, 205, 210, 215, 220, 215, 210, 205],
      current3: [120, 125, 130, 135, 140, 145, 150, 155, 160, 155, 150, 145],
      current4: [60, 65, 70, 75, 80, 85, 90, 95, 100, 95, 90, 85],
      trend1: [350, 360, 370, 380, 390, 400, 410, 420, 430, 425, 420, 415],
      trend2: [180, 190, 200, 210, 220, 230, 240, 250, 260, 255, 250, 245],
      trend3: [120, 130, 140, 150, 160, 170, 180, 190, 200, 195, 190, 185]
    }
  }
};