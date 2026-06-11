/**
 * charts.js — Real V/I time-series charts with:
 *   • Actual phase values on Y axis, time on X axis
 *   • Static threshold reference lines (HIGH / LOW)
 *   • Anomaly regions shaded via Chart.js plugin
 *   • Tooltip shows value + whether anomaly + episode duration
 */

const _charts = {};

// ── Colour palette ────────────────────────────────────────────────────────────
const C = {
  vry:    "#00e5d4",
  vyb:    "#e040fb",
  vbr:    "#ffab40",
  v_avg:  "#b2ff59",
  ir:     "#40c4ff",
  iy:     "#ff6e40",
  ib:     "#ea80fc",
  i_avg:  "#ccff90",
  high:   "rgba(255,80,80,0.18)",
  low:    "rgba(80,120,255,0.18)",
  thresh_high: "#ff5252",
  thresh_low:  "#448aff",
  grid:   "rgba(100,80,200,0.12)",
  tick:   "#7a6fa0",
};

// ── Anomaly shading plugin ────────────────────────────────────────────────────
// Draws coloured bands behind the line for every anomaly row.
const anomalyShadingPlugin = {
  id: "anomalyShading",
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea) return;
    const xScale = scales.x;
    const { anomalyFlags, anomalyColor } = chart.options._anomaly || {};
    if (!anomalyFlags || !anomalyFlags.length) return;

    ctx.save();
    ctx.fillStyle = anomalyColor || C.high;

    let inRun = false;
    let runStart = null;

    const drawRect = (startPx, endPx) => {
      ctx.fillRect(
        startPx, chartArea.top,
        endPx - startPx, chartArea.bottom - chartArea.top
      );
    };

    for (let i = 0; i < anomalyFlags.length; i++) {
      const px = xScale.getPixelForValue(i);
      const halfStep = i === 0
        ? 0
        : (px - xScale.getPixelForValue(i - 1)) / 2;

      if (anomalyFlags[i]) {
        if (!inRun) { runStart = px - halfStep; inRun = true; }
      } else {
        if (inRun) { drawRect(runStart, px - halfStep); inRun = false; }
      }
    }
    if (inRun) drawRect(runStart, chartArea.right);
    ctx.restore();
  },
};

Chart.register(anomalyShadingPlugin);
// console.log(labels);
// console.log(values);
// ── Base chart options ────────────────────────────────────────────────────────
function _baseOpts(yLabel, anomalyFlags, anomalyColor, extraAnnotations) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    interaction: { mode: "index", intersect: false },
    _anomaly: { anomalyFlags, anomalyColor },
    plugins: {
      legend: {
        labels: { color: "#e8e0ff", font: { family: "Share Tech Mono", size: 10 }, boxWidth: 12 },
      },
      tooltip: {
        callbacks: {
          afterBody(items) {
            const idx = items[0]?.dataIndex;
            if (idx == null) return [];
            const flags  = anomalyFlags || [];
            const dur    = extraAnnotations?.durations || [];
            const lines  = [];
            if (flags[idx]) {
              lines.push(`⚠ ANOMALY`);
              if (dur[idx]) lines.push(`Episode duration: ${dur[idx]} min`);
            }
            return lines;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: C.tick,
          font: { size: 9, family: "Share Tech Mono" },
          maxRotation: 45,
          autoSkip: true,
          maxTicksLimit: 10,
        },
        grid: { color: C.grid },
      },
      y: {
        title: { display: true, text: yLabel, color: C.tick, font: { size: 10 } },
        ticks: { color: C.tick, font: { size: 9 } },
        grid: { color: C.grid },
      },
    },
  };
}

// ── Dataset builder ───────────────────────────────────────────────────────────
function _ds(label, data, color, { fill = false, dashed = false, pointRadius = 1.5 } = {}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color + "22",
    borderWidth: dashed ? 1 : 1.8,
    borderDash: dashed ? [6, 3] : [],
    pointRadius,
    pointHoverRadius: 4,
    tension: 0.25,
    fill,
  };
}

// Flat threshold line dataset
function _threshLine(label, value, count, color) {
  return {
    label,
    data: Array(count).fill(value),
    borderColor: color,
    backgroundColor: "transparent",
    borderWidth: 1.2,
    borderDash: [4, 4],
    pointRadius: 0,
    tension: 0,
  };
}

// ── Destroy + create ──────────────────────────────────────────────────────────
function _make(id, datasets, labels, opts) {
  if (_charts[id]) _charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  _charts[id] = new Chart(ctx, {
    type: "line",
    data: { labels, datasets },
    options: opts,
  });
}

// ── Main entry point ──────────────────────────────────────────────────────────
async function createCharts(substation, feeder) {
  let s;
  try {
    s = await ApiService.getChartSeries(substation, feeder,96);
  } catch (e) {
    console.error("Chart series error:", e);
    return;
  }

  const lbl = s.categories || [];
  const n   = lbl.length;
  // CANGED THRESHOLD
  const VH  = s.voltage_high ?? 10.98;
  const VL  = s.voltage_low  ?? 10;
  const IH  = s.current_high ?? 10.98;
  const IL  = s.current_low  ??  10.5;

  // ── Chart 1: Voltage — FVHI (high surge) ─────────────────────────────────
  _make("chart1",
    [
      _ds("VRY", s.vry, C.vry),
      _ds("VYB", s.vyb, C.vyb),
      _ds("VBR", s.vbr, C.vbr),
      _ds("V avg", s.v_avg, C.v_avg, { dashed: true }),
      _threshLine(`High (${VH}V)`, VH, n, C.thresh_high),
    ],
    lbl,
    _baseOpts("Voltage (V)", s.fvhi_flag, C.high, { durations: s.fvhd })
  );

  // ── Chart 2: FVHD — duration bar of each high-voltage episode ────────────
  _makeBar("chart2", lbl, s.fvhd, "FVHD (min)", C.thresh_high,
    "High-voltage duration shown per row — non-zero rows = part of a surge episode");

  // ── Chart 3: Voltage — FVLI (low dip) ────────────────────────────────────
  _make("chart3",
    [
      _ds("VRY", s.vry, C.vry),
      _ds("VYB", s.vyb, C.vyb),
      _ds("VBR", s.vbr, C.vbr),
      _ds("V avg", s.v_avg, C.v_avg, { dashed: true }),
      _threshLine(`Low (${VL}V)`, VL, n, C.thresh_low),
    ],
    lbl,
    _baseOpts("Voltage (V)", s.fvli_flag, C.low, { durations: s.fvld })
  );

  // ── Chart 4: FVLD ─────────────────────────────────────────────────────────
  _makeBar("chart4", lbl, s.fvld, "FVLD (min)", C.thresh_low,
    "Low-voltage duration shown per row");

  // ── Chart 5: Voltage combined ─────────────────────────────────────────────
  {
    // Combine both flag arrays for combined shading
    const both = (s.fvhi_flag || []).map((v, i) => v || (s.fvli_flag||[])[i] || 0);
    _make("chart5",
      [
        _ds("VRY", s.vry, C.vry),
        _ds("VYB", s.vyb, C.vyb),
        _ds("VBR", s.vbr, C.vbr),
        _ds("V avg", s.v_avg, C.v_avg, { dashed: true }),
        _threshLine(`High (${VH}V)`, VH, n, C.thresh_high),
        _threshLine(`Low (${VL}V)`,  VL, n, C.thresh_low),
      ],
      lbl,
      _baseOpts("Voltage (V)", both, C.high)
    );
  }

  // ── Chart 6: Current — FCHI (high surge) ─────────────────────────────────
  _make("chart6",
    [
      _ds("IR", s.ir, C.ir),
      _ds("IY", s.iy, C.iy),
      _ds("IB", s.ib, C.ib),
      _ds("I avg", s.i_avg, C.i_avg, { dashed: true }),
      _threshLine(`High (${IH}A)`, IH, n, C.thresh_high),
    ],
    lbl,
    _baseOpts("Current (A)", s.fchi_flag, C.high, { durations: s.fchd })
  );

  // ── Chart 7: FCHD ─────────────────────────────────────────────────────────
  _makeBar("chart7", lbl, s.fchd, "FCHD (min)", C.thresh_high,
    "High-current duration shown per row");

  // ── Chart 8: Current — FCLI (low dip) ────────────────────────────────────
  _make("chart8",
    [
      _ds("IR", s.ir, C.ir),
      _ds("IY", s.iy, C.iy),
      _ds("IB", s.ib, C.ib),
      _ds("I avg", s.i_avg, C.i_avg, { dashed: true }),
      _threshLine(`Low (${IL}A)`, IL, n, C.thresh_low),
    ],
    lbl,
    _baseOpts("Current (A)", s.fcli_flag, C.low, { durations: s.fcld })
  );

  // ── Chart 9: FCLD ─────────────────────────────────────────────────────────
  _makeBar("chart9", lbl, s.fcld, "FCLD (min)", C.thresh_low,
    "Low-current duration shown per row");

  // ── Chart 10: Current combined ────────────────────────────────────────────
  {
    const both = (s.fchi_flag || []).map((v, i) => v || (s.fcli_flag||[])[i] || 0);
    _make("chart10",
      [
        _ds("IR", s.ir, C.ir),
        _ds("IY", s.iy, C.iy),
        _ds("IB", s.ib, C.ib),
        _ds("I avg", s.i_avg, C.i_avg, { dashed: true }),
        _threshLine(`High (${IH}A)`, IH, n, C.thresh_high),
        _threshLine(`Low (${IL}A)`,  IL, n, C.thresh_low),
      ],
      lbl,
      _baseOpts("Current (A)", both, C.high)
    );
  }
}

// ── Duration bar chart helper ─────────────────────────────────────────────────
function _makeBar(id, labels, data, label, color, title) {
  const maxVal = Math.max(...data, 10);
  if (_charts[id]) _charts[id].destroy();
  const ctx = document.getElementById(id);
  if (!ctx) return;
  _charts[id] = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label,
        data,
        backgroundColor: color + "88",
        borderColor: color,
        borderWidth: 1,
        borderRadius: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300 },
      plugins: {
        legend: { labels: { color: "#e8e0ff", font: { size: 10 } } },
        tooltip: {
          callbacks: {
            label: (item) => {
              const v = item.raw;
              return v > 0 ? `${label}: ${v} min (part of ${v}-min episode)` : "No anomaly";
            },
          },
        },
      },
      scales: {
        x: {
          ticks: { color: C.tick, font: { size: 9 }, maxRotation: 45, autoSkip: true, maxTicksLimit: 10 },
          grid: { color: C.grid },
        },
        y: {
          title: { display: true, text: "Duration (min)", color: C.tick, font: { size: 10 } },
          ticks: { color: C.tick, font: { size: 9 } },
          grid: { color: C.grid },
          beginAtZero: true,
          suggestedMax: maxVal * 1.15
        },
      },
    },
  });
}