let chartRegistry = [];

function destroyCharts() {
  chartRegistry.forEach(chart => {
    chart.destroy();
  });
  chartRegistry = [];
}

async function createCharts(substation, feeder) {
  destroyCharts();

  let chartData = await ApiService.getChartData(substation, feeder, 12);

  const labels = chartData.categories.length > 0
    ? chartData.categories
    : [
        "00:00", "01:00", "02:00", "03:00",
        "04:00", "05:00", "06:00", "07:00",
        "08:00", "09:00", "10:00", "11:00"
      ];

  if (chartData.categories.length === 0) {
    if (MOCK_SCADA_DATA[substation]) {
      chartData = MOCK_SCADA_DATA[substation].charts;
    } else {
      chartData = MOCK_SCADA_DATA["Sub1"].charts;
    }
  }

  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      }
    },
    scales: {
      x: {
        ticks: {
          color: "#7a6fa0"
        },
        grid: {
          color: "rgba(100,80,200,.08)"
        }
      },
      y: {
        ticks: {
          color: "#7a6fa0"
        },
        grid: {
          color: "rgba(100,80,200,.08)"
        }
      }
    }
  };

  function buildBarChart(canvasId, data, color) {
    const ctx = document.getElementById(canvasId);

    if (!ctx) {
      console.error(canvasId + " not found");
      return;
    }

    const chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels,
        datasets: [
          {
            data: data,
            backgroundColor: color,
            borderRadius: 3,
            hoverBackgroundColor: color
          }
        ]
      },
      options: commonOptions
    });

    chartRegistry.push(chart);
  }

  function buildLineChart(canvasId) {
    const ctx = document.getElementById(canvasId);

    if (!ctx) {
      console.error(canvasId + " not found");
      return;
    }

    const chart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            data: chartData.trend1,
            borderColor: "#00e5d4",
            tension: 0.4,
            fill: false,
            borderWidth: 2
          },
          {
            data: chartData.trend2,
            borderColor: "#9c6fff",
            tension: 0.4,
            fill: false,
            borderWidth: 2
          },
          {
            data: chartData.trend3,
            borderColor: "#e040fb",
            tension: 0.4,
            fill: false,
            borderWidth: 2
          }
        ]
      },
      options: commonOptions
    });

    chartRegistry.push(chart);
  }

  // Create voltage charts
  buildBarChart("chart1", chartData.voltage1, "#00e5d4");
  buildBarChart("chart2", chartData.voltage2, "#9c6fff");
  buildBarChart("chart3", chartData.voltage3, "#e040fb");
  buildBarChart("chart4", chartData.voltage4, "#b2ff59");
  buildLineChart("chart5");

  // Create current charts
  buildBarChart("chart6", chartData.current1, "#00e5d4");
  buildBarChart("chart7", chartData.current2, "#9c6fff");
  buildBarChart("chart8", chartData.current3, "#e040fb");
  buildBarChart("chart9", chartData.current4, "#b2ff59");
  buildLineChart("chart10");
}