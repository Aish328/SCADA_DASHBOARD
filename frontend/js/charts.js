let chartRegistry = [];

function destroyCharts(){

  chartRegistry.forEach(chart=>{

    chart.destroy();

  });

  chartRegistry = [];

}

function createCharts(city){

  destroyCharts();

  const cityData = SCADA_DATA[city];

  const labels = [
    "2012","2013","2014","2015",
    "2016","2017","2018","2019",
    "2020","2021","2022","2023"
  ];

  const commonOptions = {

    responsive:true,

    maintainAspectRatio:false,

    plugins:{
      legend:{
        display:false
      }
    },

    scales:{

      x:{
        ticks:{
          color:"#7a6fa0"
        },

        grid:{
          color:"rgba(100,80,200,.08)"
        }
      },

      y:{
        ticks:{
          color:"#7a6fa0"
        },

        grid:{
          color:"rgba(100,80,200,.08)"
        }
      }

    }

  };

  function buildBarChart(canvasId,data,color){

    const ctx =
      document.getElementById(canvasId);

    if(!ctx){

      console.error(canvasId + " not found");

      return;

    }

    const chart = new Chart(ctx,{

      type:"bar",

      data:{

        labels,

        datasets:[{

          data,

          backgroundColor:color,

          borderRadius:4

        }]

      },

      options:commonOptions

    });

    chartRegistry.push(chart);

  }

  function buildLineChart(canvasId){

    const ctx =
      document.getElementById(canvasId);

    if(!ctx){

      console.error(canvasId + " not found");

      return;

    }

    const chart = new Chart(ctx,{

      type:"line",

      data:{

        labels,

        datasets:[

          {
            data:cityData.charts.trend1,
            borderColor:"#00e5d4",
            tension:.4
          },

          {
            data:cityData.charts.trend2,
            borderColor:"#9c6fff",
            tension:.4
          },

          {
            data:cityData.charts.trend3,
            borderColor:"#e040fb",
            tension:.4
          }

        ]

      },

      options:commonOptions

    });

    chartRegistry.push(chart);

  }

  /* VOLTAGE */

  buildBarChart(
    "chart1",
    cityData.charts.voltage1,
    "#00e5d4"
  );

  buildBarChart(
    "chart2",
    cityData.charts.voltage2,
    "#9c6fff"
  );

  buildBarChart(
    "chart3",
    cityData.charts.voltage3,
    "#e040fb"
  );

  buildBarChart(
    "chart4",
    cityData.charts.voltage4,
    "#b2ff59"
  );

  buildLineChart("chart5");

  /* CURRENT */

  buildBarChart(
    "chart6",
    cityData.charts.current1,
    "#00e5d4"
  );

  buildBarChart(
    "chart7",
    cityData.charts.current2,
    "#9c6fff"
  );

  buildBarChart(
    "chart8",
    cityData.charts.current3,
    "#e040fb"
  );

  buildBarChart(
    "chart9",
    cityData.charts.current4,
    "#b2ff59"
  );

  buildLineChart("chart10");

}