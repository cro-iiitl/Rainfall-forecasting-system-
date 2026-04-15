/* ── main.js — UP Rainfall Prediction frontend ── */
'use strict';

const API = '';   // same origin

/* ─── Rain background ────────────────────────── */
function spawnRain() {
  const bg = document.getElementById('rainBg');
  const N = 60;
  for (let i = 0; i < N; i++) {
    const d = document.createElement('div');
    d.className = 'rain-drop';
    d.style.left    = Math.random() * 100 + 'vw';
    d.style.height  = (60 + Math.random() * 80) + 'px';
    d.style.opacity = 0.2 + Math.random() * 0.5;
    d.style.animationDuration  = (1.2 + Math.random() * 1.8) + 's';
    d.style.animationDelay     = (-Math.random() * 3) + 's';
    bg.appendChild(d);
  }
}

/* ─── District loader ────────────────────────── */
async function loadDistricts() {
  try {
    const res  = await fetch(`${API}/api/districts`);
    const data = await res.json();
    const sel  = document.getElementById('districtSel');
    data.districts.forEach(d => {
      const opt = document.createElement('option');
      opt.value = d;
      opt.textContent = d;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.warn('District load failed', e);
  }
}

/* ─── Helpers ────────────────────────────────── */
function rainIcon(mm) {
  if (mm < 0.5)  return 'sun';
  if (mm < 2.5)  return 'cloud-drizzle';
  if (mm < 10)   return 'cloud-rain';
  if (mm < 35)   return 'cloud-lightning';
  return 'waves';
}
function rainClass(mm) {
  if (mm < 0.5)  return 'rain-none';
  if (mm < 2.5)  return 'rain-light';
  if (mm < 10)   return 'rain-mod';
  if (mm < 35)   return 'rain-heavy';
  return 'rain-storm';
}
function rainLabel(mm) {
  if (mm < 0.5)  return { text: 'Trace',      cls: 'pill-dry'   };
  if (mm < 2.5)  return { text: 'Very Light', cls: 'pill-rain'  };
  if (mm < 10)   return { text: 'Light',      cls: 'pill-rain'  };
  if (mm < 35)   return { text: 'Moderate',   cls: 'pill-warn'  };
  return           { text: 'Heavy',      cls: 'pill-storm'  };
}
function fmtDate(iso) {
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString('en-IN', { weekday:'short', month:'short', day:'numeric' });
}
function fmtDateShort(iso) {
  const d = new Date(iso + 'T00:00:00');
  return d.toLocaleDateString('en-IN', { month:'short', day:'numeric' });
}
function dayName(iso) {
  const d = new Date(iso + 'T00:00:00');
  const days = ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'];
  return days[d.getDay()];
}
function prettyToken(value) {
  return value ? String(value).replace(/_/g, ' ') : '—';
}
function refreshIcons() {
  if (window.lucide) {
    window.lucide.createIcons();
  }
}

/* ─── Render ─────────────────────────────────── */
function showError(msg) {
  const el = document.getElementById('errorBox');
  el.textContent = msg;
  el.classList.add('active');
}
function clearError() {
  const el = document.getElementById('errorBox');
  el.textContent = '';
  el.classList.remove('active');
}

let chart = null;

function renderResults(forecast, label) {
  clearError();
  document.getElementById('resultSection').classList.remove('hidden');

  /* Location header */
  const first = forecast[0];
  document.getElementById('locName').textContent = label;
  document.getElementById('locCoords').textContent =
    `${first.lat}°N, ${first.lon}°E  ·  Model: ${first.model_district}`;
  document.getElementById('modelBadge').textContent = first.model_used
    ? prettyToken(first.model_used)
    : 'XGBoost Day-wise';
  document.getElementById('sourceBadge').textContent = first.lag_rainfall_source
    ? prettyToken(first.lag_rainfall_source)
    : 'live weather';

  /* Stats */
  const rains    = forecast.map(f => f.predicted_rainfall_mm);
  const total    = rains.reduce((a, b) => a + b, 0);
  const peak     = Math.max(...rains);
  const peakIndex = rains.indexOf(peak);
  const peakDay = forecast[peakIndex] ? `${fmtDateShort(forecast[peakIndex].date)} · ${peak.toFixed(1)} mm` : '—';
  const rainyDays = rains.filter(v => v >= 0.5).length;
  const avgCloud  = forecast.reduce((a, f) => a + (f.cloud_cover_pct || 0), 0) / forecast.length;
  document.getElementById('statTotal').textContent    = total.toFixed(1) + ' mm';
  document.getElementById('statPeak').textContent     = peakDay;
  document.getElementById('statRainy').textContent    = rainyDays + ' / 7';
  document.getElementById('statCloud').textContent    = avgCloud.toFixed(0) + '%';

  /* Forecast cards */
  const grid = document.getElementById('cardsGrid');
  grid.innerHTML = '';
  forecast.forEach((f, i) => {
    const prob    = f.rain_probability_pct != null ? f.rain_probability_pct : null;
    const lbl     = rainLabel(f.predicted_rainfall_mm);
    const card = document.createElement('div');
    card.className = `forecast-card glass ${rainClass(f.predicted_rainfall_mm)}`;
    card.style.animationDelay = (i * 0.07) + 's';
    card.innerHTML = `
      <div class="fc-day">${dayName(f.date)}</div>
      <div class="fc-date">${fmtDateShort(f.date)}</div>
      <span class="fc-icon" aria-hidden="true"><i data-lucide="${rainIcon(f.predicted_rainfall_mm)}"></i></span>
      <div class="fc-rain">${f.predicted_rainfall_mm.toFixed(1)}</div>
      <div class="fc-unit">mm rainfall</div>
      ${prob != null ? `<div class="fc-prob" style="color:var(--accent-rain)">${prob.toFixed(0)}% chance</div>` : ''}
      <div class="fc-bar-wrap"><div class="fc-bar" style="width:${Math.min(prob ?? (f.predicted_rainfall_mm > 0 ? 60 : 5), 100)}%"></div></div>
      <div style="margin-top:10px"><span class="pill ${lbl.cls}">${lbl.text}</span></div>
    `;
    grid.appendChild(card);
  });

  /* Bar chart */
  const ctx = document.getElementById('rainfallChart').getContext('2d');
  const labels  = forecast.map(f => `${dayName(f.date)} ${fmtDateShort(f.date)}`);
  const values  = rains;

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Predicted Rainfall (mm)',
        data: values,
        backgroundColor: values.map(v =>
          v < 0.5  ? 'rgba(255,255,255,0.08)' :
          v < 2.5  ? 'rgba(56,189,248,0.45)' :
          v < 10   ? 'rgba(79,156,249,0.65)' :
          v < 35   ? 'rgba(251,191,36,0.65)' :
                     'rgba(248,113,113,0.7)'
        ),
        borderColor: values.map(v =>
          v < 0.5  ? 'rgba(255,255,255,0.15)' :
          v < 2.5  ? '#38bdf8' :
          v < 10   ? '#4f9cf9' :
          v < 35   ? '#fbbf24' :
                     '#f87171'
        ),
        borderWidth: 2,
        borderRadius: 8,
        borderSkipped: false,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: { duration: 900, easing: 'easeOutQuart' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(10,22,45,0.95)',
          borderColor: 'rgba(79,156,249,0.3)',
          borderWidth: 1,
          titleColor: '#e8f4ff',
          bodyColor: '#7ba3cc',
          padding: 12,
          callbacks: {
            label: ctx => ` ${ctx.parsed.y.toFixed(2)} mm`
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)' },
          ticks: { color: '#7ba3cc', font: { family: 'Inter', size: 11 } }
        },
        y: {
          grid: { color: 'rgba(255,255,255,0.05)' },
          ticks: {
            color: '#7ba3cc', font: { family: 'Inter', size: 11 },
            callback: v => v + ' mm'
          },
          beginAtZero: true
        }
      }
    }
  });

  /* Detail table */
  const tbody = document.querySelector('#detailTable tbody');
  tbody.innerHTML = '';
  forecast.forEach(f => {
    const lbl = rainLabel(f.predicted_rainfall_mm);
    const prob = f.rain_probability_pct != null ? f.rain_probability_pct.toFixed(1) + '%' : '—';
    const tempRange = f.t2m_min_c != null && f.t2m_max_c != null
      ? `${f.t2m_min_c.toFixed(1)}–${f.t2m_max_c.toFixed(1)}°C`
      : '—';
    const row = document.createElement('tr');
    row.innerHTML = `
      <td>${fmtDate(f.date)}</td>
      <td>${f.predicted_rainfall_mm.toFixed(2)} mm</td>
      <td>${tempRange}</td>
      <td>${f.t2m_mean_c != null ? f.t2m_mean_c.toFixed(1) + '°C' : '—'}</td>
      <td>${f.cloud_cover_pct != null ? f.cloud_cover_pct.toFixed(0) + '%' : '—'}</td>
      <td>${f.api_precipitation_mm != null ? f.api_precipitation_mm.toFixed(2) + ' mm' : '—'}</td>
      <td>${prob}</td>
      <td><span class="pill ${lbl.cls}">${lbl.text}</span></td>
      <td><span style="font-size: 0.85em; opacity: 0.9; color: #64ffda;">${f.model_reasoning || 'N/A'}</span></td>
    `;
    tbody.appendChild(row);
  });

  refreshIcons();
  document.getElementById('resultSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/* ─── Prediction call ────────────────────────── */
async function runPrediction() {
  clearError();
  const district = document.getElementById('districtSel').value;
  const lat      = parseFloat(document.getElementById('latInput').value);
  const lon      = parseFloat(document.getElementById('lonInput').value);

  let body = {};
  let label = '';

  if (district) {
    body = { district };
    label = district + ' District';
  } else if (!isNaN(lat) && !isNaN(lon)) {
    body = { lat, lon };
    label = `${lat.toFixed(4)}°N, ${lon.toFixed(4)}°E`;
  } else {
    showError('Please select a district OR enter both latitude and longitude.');
    return;
  }

  /* show loader, hide results */
  document.getElementById('resultSection').classList.add('hidden');
  document.getElementById('loaderWrap').classList.add('active');
  document.getElementById('predictBtn').disabled = true;

  try {
    const res  = await fetch(`${API}/api/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    });
    const data = await res.json();

    if (!res.ok || data.error) {
      showError(data.error || `Server error ${res.status}`);
      return;
    }
    renderResults(data.forecast, label);
  } catch (e) {
    showError('Network error — make sure the server is running.');
    console.error(e);
  } finally {
    document.getElementById('loaderWrap').classList.remove('active');
    document.getElementById('predictBtn').disabled = false;
  }
}

/* ─── Init ───────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  spawnRain();
  loadDistricts();
  refreshIcons();

  document.getElementById('predictBtn').addEventListener('click', runPrediction);

  /* clear lat/lon when district chosen and vice versa */
  document.getElementById('districtSel').addEventListener('change', () => {
    document.getElementById('latInput').value = '';
    document.getElementById('lonInput').value = '';
  });
  document.getElementById('latInput').addEventListener('input', () => {
    document.getElementById('districtSel').value = '';
  });
  document.getElementById('lonInput').addEventListener('input', () => {
    document.getElementById('districtSel').value = '';
  });
});
