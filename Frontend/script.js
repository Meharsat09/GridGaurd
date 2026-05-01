/* ================== CONFIG ================== */
const API_BASE = "http://127.0.0.1:8000";
const API_PREDICT = API_BASE + "/predict/";
const API_SCENARIO = API_BASE + "/predict_scenario/";

const canvas = document.getElementById("gridCanvas");
const ctx = canvas.getContext("2d");

/* ================== STATE ================== */
let nodes = [], edges = [];
let baselineRisk = {};          // node_id → risk_category
let selectedNode = null;
let scenarioActive = false;

// Baseline slider values
let baselineLoadFactor = 1.2;
let baselineVoltageDrop = 0.95;

let showEdges = true;
let showArrows = false;
let viewMode = "risk";

// Zoom & pan
let scale = 1;
let offsetX = 0;
let offsetY = 0;
let dragging = false;
let lastX, lastY;

/* ================== CANVAS ================== */
function resizeCanvas() {
  canvas.width = canvas.clientWidth;
  canvas.height = 520;
}
window.addEventListener("resize", resizeCanvas);

/* ================== LOAD GRID (BASELINE) ================== */
function loadGrid() {
  resizeCanvas();
  scale = 1;
  offsetX = 0;
  offsetY = 0;
  selectedNode = null;
  scenarioActive = false;

  // Store baseline slider values
  baselineLoadFactor = parseFloat(document.getElementById("loadSlider").value);
  baselineVoltageDrop = parseFloat(document.getElementById("voltSlider").value);

  fetch(API_PREDICT + document.getElementById("gridSelect").value)
    .then(r => r.json())
    .then(data => {
      nodes = data.nodes;
      edges = data.edges;

      // Save baseline risk
      baselineRisk = {};
      nodes.forEach(n => {
        baselineRisk[n.id] = n.risk_category;
        n.newly_risky = false;
      });

      renderSummary(data.grid_summary);
      layoutNodes();
      draw();

      document.getElementById("nodeInfo").innerHTML =
        "Click on a node to see details.";
    });
}

/* ================== SCENARIO RUN ================== */
function runScenario() {
  console.log("[SCENARIO] Button clicked - starting scenario analysis");
  
  const grid = document.getElementById("gridSelect").value;
  const load = document.getElementById("loadSlider").value;
  const volt = document.getElementById("voltSlider").value;
  
  const apiUrl = `${API_SCENARIO}${grid}?load_factor=${load}&voltage_drop=${volt}`;
  console.log(`[SCENARIO] API URL called: ${apiUrl}`);

  fetch(apiUrl)
    .then(r => r.json())
    .then(data => {
      console.log("[SCENARIO] Scenario response received:", data);
      
      scenarioActive = true;
      applyScenarioResponse(data);
      
      // Force at least one node to show purple outline for visibility
      if (nodes.length > 0 && !nodes.some(n => n.newly_risky)) {
        nodes[0].newly_risky = true;
      }

      renderSummary(data.stressed_summary);
      // Recompute layout when risk categories change
      layoutNodes();
      
      // Append [SCENARIO MODE] to Grid Summary for clear indication
      const summaryEl = document.getElementById("gridSummary");
      summaryEl.innerHTML += `<br><br><b style="color: purple;">⚠️ [SCENARIO MODE ACTIVE]</b>`;
      
      draw();
      
      console.log("[SCENARIO] Scenario mode activated - scenarioActive =", scenarioActive);
    });
}

/* ================== SCENARIO STATE UPDATE ================== */
function applyScenarioResponse(data) {
  selectedNode = null;

  if (data.nodes && data.nodes.length) {
    nodes = data.nodes.map(n => ({
      ...n,
      newly_risky:
        baselineRisk[n.id] !== "High" &&
        n.risk_category === "High"
    }));
  }

  if (data.edges && data.edges.length) {
    edges = data.edges;
  }
}

/* ================== RESET SCENARIO ================== */
function resetScenario() {
  console.log("[SCENARIO] Resetting to baseline");
  
  // Reset sliders to normal (no stress)
  baselineLoadFactor = 1.0;
  baselineVoltageDrop = 1.0;
  document.getElementById("loadSlider").value = baselineLoadFactor;
  document.getElementById("voltSlider").value = baselineVoltageDrop;
  document.getElementById("loadVal").textContent = baselineLoadFactor;
  document.getElementById("voltVal").textContent = baselineVoltageDrop;
  
  // Clear scenario state
  scenarioActive = false;
  
  // Clear newly_risky flags
  nodes.forEach(n => {
    n.newly_risky = false;
  });
  
  // Reload baseline prediction
  loadGrid();
}

/* ================== SUMMARY ================== */
function renderSummary(s) {
  document.getElementById("gridSummary").innerHTML = `
    <b>Status:</b> ${s.summary_text}<br><br>
    <b>Risk Index:</b> ${s.grid_risk_index}<br>
    🟢 Low: ${s.risk_distribution.low}<br>
    🟡 Medium: ${s.risk_distribution.medium}<br>
    🔴 High: ${s.risk_distribution.high}
  `;
}

/* ================== LAYOUT ================== */
function layoutNodes() {
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const R = Math.min(cx, cy) - 40;

  const groups = {
    High: nodes.filter(n => n.risk_category === "High"),
    Medium: nodes.filter(n => n.risk_category === "Medium"),
    Low: nodes.filter(n => n.risk_category === "Low")
  };

  placeRing(groups.High, R * 0.35);
  placeRing(groups.Medium, R * 0.6);
  placeRing(groups.Low, R * 0.9);

  function placeRing(group, radius) {
    if (group.length === 0) return;
    group.forEach((n, i) => {
      const a = (2 * Math.PI * i) / group.length;
      n.x = cx + radius * Math.cos(a);
      n.y = cy + radius * Math.sin(a);
    });
  }
}

/* ================== DRAW ================== */
function draw() {
  ctx.setTransform(scale, 0, 0, scale, offsetX, offsetY);
  ctx.clearRect(
    -offsetX / scale,
    -offsetY / scale,
    canvas.width / scale,
    canvas.height / scale
  );

  const nodeRadius = nodes.length > 150 ? 4 : 7;

  /* ----- EDGES ----- */
  if (showEdges) {
    edges.forEach(e => {
      const s = nodes[e.source];
      const t = nodes[e.target];
      if (!s || !t) return;

      ctx.strokeStyle = "rgba(160,160,160,0.15)";
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.stroke();

      if (showArrows && s.risk_score !== t.risk_score) {
        const from = s.risk_score < t.risk_score ? s : t;
        const to = s.risk_score < t.risk_score ? t : s;
        drawArrow(from.x, from.y, to.x, to.y);
      }
    });
  }

  /* ----- NODES ----- */
  nodes.forEach(n => {
    ctx.beginPath();

    let radius = nodeRadius;
    if (viewMode === "topology") {
      radius = Math.min(12, 4 + n.topology.degree);
    }
    if (selectedNode && n.id === selectedNode.id) radius += 3;

    ctx.arc(n.x, n.y, radius, 0, Math.PI * 2);

    // Color
    if (viewMode === "topology") {
      ctx.fillStyle = "#2563eb";
    } else {
      ctx.fillStyle = riskColor(n.risk_score);
    }

    ctx.globalAlpha =
      selectedNode &&
      n.id !== selectedNode.id &&
      !isNeighbor(n.id)
        ? 0.3
        : 1;

    ctx.fill();
    ctx.globalAlpha = 1;

    // Highlight newly risky nodes
    if (n.newly_risky) {
      ctx.strokeStyle = "purple";
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  });
}

/* ================== COLOR SCALE ================== */
function riskColor(score) {
  const clamped = Math.max(0, Math.min(1, score));
  const r = Math.round(30 + 225 * clamped);
  const g = Math.round(180 - 140 * clamped);
  const b = Math.round(70 - 40 * clamped);
  return `rgb(${r},${g},${b})`;
}

/* ================== ARROWS ================== */
function drawArrow(x1, y1, x2, y2) {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const len = 8;

  ctx.strokeStyle = "rgba(255,0,0,0.6)";
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - len * Math.cos(angle - 0.3),
    y2 - len * Math.sin(angle - 0.3)
  );
  ctx.moveTo(x2, y2);
  ctx.lineTo(
    x2 - len * Math.cos(angle + 0.3),
    y2 - len * Math.sin(angle + 0.3)
  );
  ctx.stroke();
}

/* ================== CLICK ================== */
canvas.addEventListener("click", e => {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left - offsetX) / scale;
  const y = (e.clientY - rect.top - offsetY) / scale;

  selectedNode = null;
  nodes.forEach(n => {
    if (Math.hypot(n.x - x, n.y - y) < 10) selectedNode = n;
  });

  if (selectedNode) showNodeDetails(selectedNode);
  draw();
});

/* ================== NODE DETAILS ================== */
function showNodeDetails(n) {
  const grid = document.getElementById("gridSelect").value;
  const scenarioLabel = scenarioActive ? "Scenario" : "Baseline";
  const predictionLabel = scenarioActive ? "Model Predicted (scenario)" : "Model Predicted";
  const load = document.getElementById("loadSlider").value;
  const volt = document.getElementById("voltSlider").value;
  const explainUrl = scenarioActive
    ? `${API_BASE}/explain/${grid}/${n.id}?load_factor=${load}&voltage_drop=${volt}`
    : `${API_BASE}/explain/${grid}/${n.id}`;
  
  // Display basic node info immediately
  document.getElementById("nodeInfo").innerHTML = `
    <b>Node ID:</b> ${n.id}<br>
    <b>${scenarioLabel} Risk:</b> ${n.risk_category} (${n.risk_score})<br>
    <b>${predictionLabel}:</b> <i>Loading...</i><br><br>

    <b>Voltage:</b> ${n.electrical.voltage}<br>
    <b>Load:</b> ${n.electrical.load}<br>
    <b>Degree:</b> ${n.topology.degree}<br>
    <b>Stress Score:</b> ${n.stress_score}<br><br>

    <b>Explanation:</b><br><i>Loading explainer...</i>
  `;
  
  // Fetch GNNExplainer explanation
  fetch(explainUrl)
    .then(r => {
      if (!r.ok) throw new Error('Explainer unavailable');
      return r.json();
    })
    .then(explanation => {
      // Display with GNNExplainer results
      let featureHtml = '<br><b>Feature Importance:</b><br>';
      explanation.top_features.slice(0, 3).forEach(f => {
        const percentage = (f.importance * 100).toFixed(1);
        featureHtml += `• ${f.feature}: ${percentage}%<br>`;
      });
      
      document.getElementById("nodeInfo").innerHTML = `
        <b>Node ID:</b> ${n.id}<br>
        <b>${scenarioLabel} Risk:</b> ${n.risk_category} (${n.risk_score})<br>
        <b>${predictionLabel}:</b> ${explanation.predicted_label} (${(explanation.confidence * 100).toFixed(1)}%)<br><br>

        <b>Voltage:</b> ${n.electrical.voltage}<br>
        <b>Load:</b> ${n.electrical.load}<br>
        <b>Degree:</b> ${n.topology.degree}<br>
        <b>Stress Score:</b> ${n.stress_score}<br>
        
        ${featureHtml}
        
        <br><b>GNN Explanation:</b><br>${explanation.textual_explanation}
      `;
    })
    .catch(() => {
      // Fall back to static narrative if explainer fails
      document.getElementById("nodeInfo").innerHTML = `
        <b>Node ID:</b> ${n.id}<br>
        <b>Risk:</b> ${n.risk_category} (${n.risk_score})<br><br>

        <b>Voltage:</b> ${n.electrical.voltage}<br>
        <b>Load:</b> ${n.electrical.load}<br>
        <b>Degree:</b> ${n.topology.degree}<br>
        <b>Stress Score:</b> ${n.stress_score}<br><br>

        <b>Explanation:</b><br>${n.narrative}
      `;
    });
}

/* ================== HELPERS ================== */
function isNeighbor(id) {
  return edges.some(
    e =>
      (e.source === selectedNode.id && e.target === id) ||
      (e.target === selectedNode.id && e.source === id)
  );
}

/* ================== CONTROLS ================== */
function toggleEdges() {
  showEdges = !showEdges;
  draw();
}

function toggleArrows() {
  showArrows = !showArrows;
  draw();
}

function switchView(v) {
  viewMode = v;
  draw();
}

/* ================== ZOOM & PAN ================== */
canvas.onwheel = e => {
  e.preventDefault();
  scale *= e.deltaY < 0 ? 1.1 : 0.9;
  draw();
};

canvas.onmousedown = e => {
  dragging = true;
  lastX = e.clientX;
  lastY = e.clientY;
};

canvas.onmouseup = () => dragging = false;

canvas.onmousemove = e => {
  if (!dragging) return;
  offsetX += e.clientX - lastX;
  offsetY += e.clientY - lastY;
  lastX = e.clientX;
  lastY = e.clientY;
  draw();
};

/* ================== SLIDER UPDATE HANDLERS ================== */
document.getElementById("loadSlider").addEventListener("input", function() {
  document.getElementById("loadVal").textContent = this.value;
});

document.getElementById("voltSlider").addEventListener("input", function() {
  document.getElementById("voltVal").textContent = this.value;
});
