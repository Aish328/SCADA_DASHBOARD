document.addEventListener("DOMContentLoaded",()=>{

  const substationSelect =
    document.getElementById("substationSelect");

  const feederPills =
    document.getElementById("feederPills");

  /* ------------------------
     LOAD SUBSTATIONS
  ------------------------ */

  Object.keys(SCADA_DATA)
    .forEach(city=>{

      const option =
        document.createElement("option");

      option.value = city;
      option.textContent = city;

      substationSelect.appendChild(option);

    });

  /* ------------------------
     UPDATE KPIs
  ------------------------ */

  function updateKpis(city){

    const kpis =
      SCADA_DATA[city].kpis;

    Object.keys(kpis)
      .forEach(key=>{

        const el =
          document.getElementById(key);

        if(el){

          el.textContent =
            kpis[key];

        }

      });

  }

  /* ------------------------
     RENDER FEEDERS
  ------------------------ */

  function renderFeeders(city){

    feederPills.innerHTML = "";

    SCADA_DATA[city]
      .feeders
      .forEach((feeder,index)=>{

        const pill =
          document.createElement("div");

        pill.className =
          index===0
            ? "pill active"
            : "pill";

        pill.textContent = feeder;

        pill.addEventListener("click",()=>{

          document.querySelectorAll(".pill")
            .forEach(p=>{

              p.classList.remove("active");

            });

          pill.classList.add("active");

        });

        feederPills.appendChild(pill);

      });

  }

  /* ------------------------
     UPDATE DASHBOARD
  ------------------------ */

  function updateDashboard(city){

    updateKpis(city);

    renderFeeders(city);

    createCharts(city);

    updateMap(city);

  }

  /* ------------------------
     INITIAL LOAD
  ------------------------ */

  updateDashboard("Delhi");

  /* ------------------------
     SUBSTATION CHANGE
  ------------------------ */

  substationSelect
    .addEventListener("change",e=>{

      updateDashboard(
        e.target.value
      );

    });

  /* ------------------------
     REFRESH
  ------------------------ */

  document
    .getElementById("refreshBtn")
    .addEventListener("click",()=>{

      const selectedCity =
        substationSelect.value;

      updateDashboard(selectedCity);

    });

});