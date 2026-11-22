const NS = "http://www.w3.org/2000/svg";
const BAND_COLOR = ["var(--good)", "var(--warning)", "var(--serious)", "var(--critical)", "var(--critical)"];
const BAND_ICON = [
  '<circle cx="8" cy="8" r="7" fill="C"/><path d="M4.6 8.3l2.2 2.2 4.6-4.9" fill="none" stroke="#fff" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>',
  '<circle cx="8" cy="8" r="7" fill="C"/><path d="M4.5 8h7" stroke="#0b0b0b" stroke-width="1.8" stroke-linecap="round"/>',
  '<path d="M8 1.2l7 12.6H1z" fill="C"/><path d="M8 5.6v4" stroke="#0b0b0b" stroke-width="1.7" stroke-linecap="round"/><circle cx="8" cy="11.6" r="0.95" fill="#0b0b0b"/>',
  '<path d="M5 1h6l4 4v6l-4 4H5l-4-4V5z" fill="C"/><path d="M8 4.4v4.4" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/><circle cx="8" cy="11.4" r="1" fill="#fff"/>',
  '<path d="M5 1h6l4 4v6l-4 4H5l-4-4V5z" fill="C"/><path d="M5.4 5.4l5.2 5.2M10.6 5.4l-5.2 5.2" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>',
];

const tooltip = document.getElementById("tooltip");
const fmtDay = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", timeZone: "UTC" });
const fmtFull = new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
const fmtMonth = new Intl.DateTimeFormat("en-GB", { month: "short", timeZone: "UTC" });
const fmtWeekday = new Intl.DateTimeFormat("en-GB", { weekday: "long", timeZone: "UTC" });
const asDate = (s) => new Date(s + "T00:00:00Z");

function el(tag, attrs, parent) {
  const node = document.createElementNS(NS, tag);
  for (const k in attrs) node.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(node);
  return node;
}

function chip(band, bands) {
  return `<svg viewBox="0 0 16 16" aria-hidden="true">${BAND_ICON[band].replaceAll("C", BAND_COLOR[band])}</svg>${bands[band].name}`;
}

function bandOf(value, bands) {
  for (let i = 0; i < bands.length; i++) if (bands[i].max === null || value < bands[i].max) return i;
  return bands.length - 1;
}

function showTip(html, x, y) {
  tooltip.innerHTML = html;
  tooltip.classList.add("on");
  const w = tooltip.offsetWidth;
  const h = tooltip.offsetHeight;
  let left = x + 14;
  if (left + w > window.innerWidth - 8) left = x - w - 14;
  let top = y - h - 12;
  if (top < 8) top = y + 16;
  tooltip.style.left = Math.max(8, left) + "px";
  tooltip.style.top = top + "px";
}

function hideTip() {
  tooltip.classList.remove("on");
}

function tipRow(color, label, value) {
  const key = color ? `<span class="key key-dot" style="background:${color}"></span>` : "";
  return `<div class="t-row">${key}${label}<b>${value}</b></div>`;
}

function bandTicks(maxValue, bands) {
  const ticks = [0];
  for (const b of bands) if (b.max !== null && b.max < maxValue) ticks.push(b.max);
  return ticks;
}

function frame(host, height, margin) {
  host.innerHTML = "";
  const width = Math.max(300, host.clientWidth);
  const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" }, host);
  return { svg, width, height, x0: margin.l, x1: width - margin.r, y0: height - margin.b, y1: margin.t };
}

function yAxis(f, maxValue, bands) {
  const y = (v) => f.y0 - (v / maxValue) * (f.y0 - f.y1);
  const ticks = bandTicks(maxValue, bands);
  ticks.forEach((t, i) => {
    el("line", { x1: f.x0, x2: f.x1, y1: y(t), y2: y(t), stroke: i === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, f.svg);
    el("text", { x: f.x0 - 8, y: y(t) + 4, "text-anchor": "end" }, f.svg).textContent = t;
    if (i > 0) {
      const name = bands[i].name.toLowerCase();
      el("text", { x: f.x1 + 8, y: y(t) - 5, class: "label" }, f.svg).textContent = name;
    }
  });
  return y;
}

function monthTicks(f, dates, x) {
  let last = -1;
  let lastX = -100;
  dates.forEach((d, i) => {
    const date = asDate(d);
    const m = date.getUTCMonth();
    if (m !== last && (date.getUTCDate() <= 3 || i === 0)) {
      const px = x(i);
      const next = dates.findIndex((v, j) => j > i && asDate(v).getUTCDate() === 1);
      const crowded = i === 0 && date.getUTCDate() > 3 && next > 0 && x(next) - px < 64;
      if (!crowded && px - lastX > 44 && px < f.x1 - 10) {
        el("text", { x: px, y: f.y0 + 20, "text-anchor": i === 0 ? "start" : "middle" }, f.svg).textContent =
          fmtMonth.format(date) + (m === 0 ? " " + date.getUTCFullYear() : "");
        lastX = px;
      }
      last = m;
    }
  });
}

function path(points) {
  return points.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("");
}

function hoverLayer(f, count, x, onMove) {
  const cross = el("line", { y1: f.y1, y2: f.y0, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" }, f.svg);
  const dots = el("g", {}, f.svg);
  const hit = el("rect", { x: f.x0, y: f.y1, width: f.x1 - f.x0, height: f.y0 - f.y1, fill: "transparent" }, f.svg);
  const move = (ev) => {
    const box = f.svg.getBoundingClientRect();
    const px = ((ev.clientX - box.left) / box.width) * f.width;
    const step = (f.x1 - f.x0) / (count - 1);
    const i = Math.max(0, Math.min(count - 1, Math.round((px - f.x0) / step)));
    cross.setAttribute("x1", x(i));
    cross.setAttribute("x2", x(i));
    cross.setAttribute("visibility", "visible");
    dots.innerHTML = "";
    onMove(i, dots, ev);
  };
  hit.addEventListener("pointermove", move);
  hit.addEventListener("pointerdown", move);
  hit.addEventListener("pointerleave", () => {
    cross.setAttribute("visibility", "hidden");
    dots.innerHTML = "";
    hideTip();
  });
}

function marker(parent, cx, cy, color) {
  el("circle", { cx, cy, r: 5, fill: color, stroke: "var(--surface)", "stroke-width": 2 }, parent);
}

function drawTrend(data) {
  const host = document.getElementById("chart-trend");
  const narrow = host.clientWidth < 560;
  const f = frame(host, narrow ? 280 : 360, { l: 34, r: narrow ? 66 : 82, t: 18, b: 34 });
  const hist = data.history.slice(narrow ? -30 : -60);
  document.getElementById("h-trend").textContent = `The last ${hist.length} days, and the next three`;
  const fc = data.forecast;
  const dates = hist.map((d) => d.date).concat(fc.map((d) => d.date));
  const top = Math.max(...hist.map((d) => d.pm25), ...fc.map((d) => d.pm25 + d.typical_error));
  const maxValue = Math.ceil((top * 1.12) / 5) * 5;
  const x = (i) => f.x0 + (i / (dates.length - 1)) * (f.x1 - f.x0);
  const y = yAxis(f, maxValue, data.bands);
  monthTicks(f, dates, x);

  const pts = hist.map((d, i) => [x(i), y(d.pm25)]);
  el("path", { d: path(pts) + `L${pts[pts.length - 1][0]} ${f.y0}L${pts[0][0]} ${f.y0}Z`, fill: "var(--s1)", opacity: 0.1 }, f.svg);
  el("path", { d: path(pts), fill: "none", stroke: "var(--s1)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);

  const n = hist.length;
  const fpts = [[x(n - 1), y(hist[n - 1].pm25)]].concat(fc.map((d, i) => [x(n + i), y(d.pm25)]));
  el("path", { d: path(fpts), fill: "none", stroke: "var(--s2)", "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, f.svg);
  fc.forEach((d, i) => {
    const cx = x(n + i);
    el("line", { x1: cx, x2: cx, y1: y(d.pm25 + d.typical_error), y2: y(Math.max(0, d.pm25 - d.typical_error)), stroke: "var(--s2)", "stroke-width": 1.5, opacity: 0.55, "stroke-linecap": "round" }, f.svg);
    marker(f.svg, cx, y(d.pm25), "var(--s2)");
  });
  const lastF = fc[fc.length - 1];
  el("text", { x: x(dates.length - 1) + 10, y: y(lastF.pm25) + 4, class: "value" }, f.svg).textContent = lastF.pm25.toFixed(1);
  el("line", { x1: x(n - 1), x2: x(n - 1), y1: f.y1, y2: f.y0, stroke: "var(--axis)", "stroke-width": 1 }, f.svg);
  el("text", { x: x(n - 1) - 6, y: f.y1 + 10, "text-anchor": "end", class: "label" }, f.svg).textContent = "latest full day";

  hoverLayer(f, dates.length, x, (i, dots, ev) => {
    const isF = i >= n;
    const item = isF ? fc[i - n] : hist[i];
    marker(dots, x(i), y(item.pm25), isF ? "var(--s2)" : "var(--s1)");
    const band = data.bands[bandOf(item.pm25, data.bands)].name;
    showTip(
      `<div class="t-head">${fmtFull.format(asDate(item.date))}</div>` +
        tipRow(isF ? "var(--s2)" : "var(--s1)", isF ? "Forecast" : "Observed", item.pm25.toFixed(1) + " µg/m³") +
        (isF ? tipRow("", "Typical error", "± " + item.typical_error.toFixed(1)) : "") +
        tipRow("", "Band", band),
      ev.clientX, ev.clientY);
  });

  document.getElementById("table-trend").innerHTML =
    "<table><thead><tr><th>Date</th><th>Kind</th><th class='num'>PM2.5 (µg/m³)</th><th>Band</th></tr></thead><tbody>" +
    fc.slice().reverse().map((d) => `<tr><td>${fmtFull.format(asDate(d.date))}</td><td>Forecast</td><td class='num'>${d.pm25.toFixed(1)} ± ${d.typical_error.toFixed(1)}</td><td>${data.bands[d.band].name}</td></tr>`).join("") +
    hist.slice().reverse().map((d) => `<tr><td>${fmtFull.format(asDate(d.date))}</td><td>Observed</td><td class='num'>${d.pm25.toFixed(1)}</td><td>${data.bands[bandOf(d.pm25, data.bands)].name}</td></tr>`).join("") +
    "</tbody></table>";
}

function drawBacktest(data) {
  const host = document.getElementById("chart-back");
  const narrow = host.clientWidth < 560;
  const f = frame(host, narrow ? 260 : 340, { l: 34, r: narrow ? 66 : 82, t: 14, b: 34 });
  const rows = data.backtest;
  const dates = rows.map((d) => d.date);
  const maxValue = Math.ceil((Math.max(...rows.map((d) => Math.max(d.actual, d.predicted))) * 1.08) / 5) * 5;
  const x = (i) => f.x0 + (i / (rows.length - 1)) * (f.x1 - f.x0);
  const y = yAxis(f, maxValue, data.bands);
  monthTicks(f, dates, x);
  el("path", { d: path(rows.map((d, i) => [x(i), y(d.actual)])), fill: "none", stroke: "var(--s1)", "stroke-width": 2, "stroke-linejoin": "round" }, f.svg);
  el("path", { d: path(rows.map((d, i) => [x(i), y(d.predicted)])), fill: "none", stroke: "var(--s2)", "stroke-width": 2, "stroke-linejoin": "round" }, f.svg);

  hoverLayer(f, rows.length, x, (i, dots, ev) => {
    const d = rows[i];
    marker(dots, x(i), y(d.actual), "var(--s1)");
    marker(dots, x(i), y(d.predicted), "var(--s2)");
    showTip(
      `<div class="t-head">${fmtFull.format(asDate(d.date))}</div>` +
        tipRow("var(--s1)", "Observed", d.actual.toFixed(1)) +
        tipRow("var(--s2)", "Forecast", d.predicted.toFixed(1)) +
        tipRow("", "Miss", Math.abs(d.actual - d.predicted).toFixed(1) + " µg/m³"),
      ev.clientX, ev.clientY);
  });

  document.getElementById("back-range").textContent = `${fmtDay.format(asDate(dates[0]))} ${asDate(dates[0]).getUTCFullYear()} to ${fmtDay.format(asDate(dates[dates.length - 1]))} ${asDate(dates[dates.length - 1]).getUTCFullYear()}`;
  document.getElementById("table-back").innerHTML =
    "<table><thead><tr><th>Date</th><th class='num'>Observed</th><th class='num'>Forecast</th><th class='num'>Miss</th></tr></thead><tbody>" +
    rows.map((d) => `<tr><td>${fmtFull.format(asDate(d.date))}</td><td class='num'>${d.actual.toFixed(1)}</td><td class='num'>${d.predicted.toFixed(1)}</td><td class='num'>${Math.abs(d.actual - d.predicted).toFixed(1)}</td></tr>`).join("") +
    "</tbody></table>";
}

function drawSkill(data) {
  const host = document.getElementById("chart-skill");
  const groups = [1, 2, 3].map((h) => ({ h, m: data.metrics["h" + h] }));
  const row = 74;
  const f = frame(host, groups.length * row + 30, { l: 96, r: 52, t: 6, b: 24 });
  const maxValue = Math.ceil(Math.max(...groups.map((g) => g.m.persistence_mae)) + 1);
  const x = (v) => f.x0 + (v / maxValue) * (f.x1 - f.x0);
  for (let t = 0; t <= maxValue; t += 2) {
    el("line", { x1: x(t), x2: x(t), y1: f.y1, y2: f.y0, stroke: t === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }, f.svg);
    el("text", { x: x(t), y: f.y0 + 16, "text-anchor": "middle" }, f.svg).textContent = t;
  }
  groups.forEach((g, i) => {
    const top = f.y1 + i * row + 16;
    el("text", { x: 0, y: top + 13, class: "value" }, f.svg).textContent = g.h === 1 ? "1 day ahead" : g.h + " days ahead";
    el("text", { x: 0, y: top + 31, class: "label" }, f.svg).textContent = g.m.below_persistence_pct.toFixed(0) + "% lower";
    [["Model", g.m.mae, "var(--s2)"], ["Persistence", g.m.persistence_mae, "var(--s3)"]].forEach(([name, v, color], j) => {
      const yy = top + j * 20;
      const w = x(v) - f.x0;
      const bar = el("path", { d: `M${f.x0} ${yy}h${w - 4}a4 4 0 0 1 4 4v8a4 4 0 0 1 -4 4h${-(w - 4)}z`, fill: color }, f.svg);
      el("text", { x: x(v) + 8, y: yy + 12, class: "value" }, f.svg).textContent = v.toFixed(2);
      const hit = el("rect", { x: f.x0, y: yy - 2, width: f.x1 - f.x0, height: 20, fill: "transparent" }, f.svg);
      hit.addEventListener("pointermove", (ev) => {
        bar.setAttribute("opacity", 0.82);
        showTip(`<div class="t-head">${g.h} day${g.h > 1 ? "s" : ""} ahead</div>` + tipRow(color, name, v.toFixed(2) + " µg/m³") + tipRow("", "Test days", g.m.test_days), ev.clientX, ev.clientY);
      });
      hit.addEventListener("pointerleave", () => {
        bar.setAttribute("opacity", 1);
        hideTip();
      });
    });
  });
  document.getElementById("table-skill").innerHTML =
    "<table><thead><tr><th>Horizon</th><th class='num'>Model MAE</th><th class='num'>Persistence MAE</th><th class='num'>Lower by</th><th class='num'>R²</th><th class='num'>Right band</th></tr></thead><tbody>" +
    groups.map((g) => `<tr><td>${g.h} day${g.h > 1 ? "s" : ""}</td><td class='num'>${g.m.mae.toFixed(2)}</td><td class='num'>${g.m.persistence_mae.toFixed(2)}</td><td class='num'>${g.m.below_persistence_pct.toFixed(1)}%</td><td class='num'>${g.m.r2.toFixed(2)}</td><td class='num'>${g.m.band_accuracy.toFixed(1)}%</td></tr>`).join("") +
    "</tbody></table>";
}

function fillStatic(data) {
  const hero = document.getElementById("hero");
  hero.dataset.band = data.latest.band;
  document.body.dataset.band = data.latest.band;
  document.getElementById("now-date").textContent = "Latest full day · " + fmtFull.format(asDate(data.latest.date));
  document.getElementById("now-value").textContent = data.latest.pm25.toFixed(0);
  document.getElementById("now-band").innerHTML = chip(data.latest.band, data.bands);
  const scaleMax = 100;
  document.getElementById("cards").innerHTML = data.forecast.map((d) => {
    const pos = Math.min(100, (d.pm25 / scaleMax) * 100);
    const marks = data.bands.filter((b) => b.max !== null).map((b) => `<b style="left:${(b.max / scaleMax) * 100}%"></b>`).join("");
    return `<article class="card">
      <div><div class="card-day">${d.horizon === 1 ? "Tomorrow" : "+" + d.horizon + " days"}</div><div class="card-date">${fmtWeekday.format(asDate(d.date))}, ${fmtDay.format(asDate(d.date))}</div></div>
      <div class="card-mid"><span class="chip">${chip(d.band, data.bands)}</span><div class="meter" role="img" aria-label="${d.pm25} on a scale from 0 to ${scaleMax}">${marks}<i style="left:${pos}%"></i></div></div>
      <div style="text-align:right"><div class="card-value">${d.pm25.toFixed(0)}</div><div class="card-error">± ${d.typical_error.toFixed(1)}</div></div>
    </article>`;
  }).join("");
  const h3 = data.metrics.h3;
  const tiles = [
    ["Three-day error", h3.mae.toFixed(1), "µg/m³", `mean absolute error on ${h3.test_days} unseen days`],
    ["Against persistence", h3.below_persistence_pct.toFixed(0) + "%", "lower", "than repeating today's value"],
    ["Right band", h3.band_accuracy.toFixed(0) + "%", "", "of three-day forecasts land in the correct band"],
    ["Variance explained", h3.r2.toFixed(2), "R²", "three days ahead, on the held-out year"],
  ];
  document.getElementById("tiles").innerHTML = tiles.map((t) => `<div class="tile"><div class="tile-label">${t[0]}</div><div class="tile-value">${t[1]}<small>${t[2]}</small></div><div class="tile-foot">${t[3]}</div></div>`).join("");
  document.getElementById("updated").textContent = data.updated;
  document.getElementById("train-days").textContent = data.training_days.toLocaleString("en-GB");
  document.getElementById("first-day").textContent = fmtFull.format(asDate(data.first_day));
}

function haze(pm25) {
  const canvas = document.getElementById("haze");
  const ctx = canvas.getContext("2d");
  const still = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const count = Math.round(Math.max(40, Math.min(320, pm25 * 3.2)));
  let w = 0;
  let h = 0;
  let visible = true;
  const dots = Array.from({ length: count }, () => ({ x: Math.random(), y: Math.random(), r: 0.6 + Math.random() * 1.8, a: 0.1 + Math.random() * 0.34, v: 0.00012 + Math.random() * 0.00042, p: Math.random() * 6.28 }));
  const size = () => {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    w = canvas.clientWidth;
    h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  const draw = (t) => {
    ctx.clearRect(0, 0, w, h);
    for (const d of dots) {
      if (!still) d.x = (d.x + d.v) % 1;
      const yy = d.y * h + Math.sin(t / 2600 + d.p) * 6;
      ctx.beginPath();
      ctx.arc(d.x * w, yy, d.r, 0, 6.2832);
      ctx.fillStyle = `rgba(255, 238, 214, ${d.a})`;
      ctx.fill();
    }
    if (!still && visible) requestAnimationFrame(draw);
  };
  size();
  window.addEventListener("resize", size);
  new IntersectionObserver((entries) => {
    const was = visible;
    visible = entries[0].isIntersecting;
    if (visible && !was && !still) requestAnimationFrame(draw);
  }).observe(canvas);
  requestAnimationFrame(draw);
}

function theme() {
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved) root.dataset.theme = saved;
  } catch (e) {}
  document.getElementById("theme").addEventListener("click", () => {
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try {
      localStorage.setItem("theme", root.dataset.theme);
    } catch (e) {}
  });
}

theme();
fetch("data/dashboard.json", { cache: "no-store" })
  .then((r) => r.json())
  .then((data) => {
    fillStatic(data);
    haze(data.latest.pm25);
    const draw = () => {
      drawTrend(data);
      drawSkill(data);
      drawBacktest(data);
    };
    draw();
    let timer;
    new ResizeObserver(() => {
      clearTimeout(timer);
      timer = setTimeout(draw, 120);
    }).observe(document.querySelector("main"));
  })
  .catch(() => {
    document.querySelector("main").insertAdjacentHTML("afterbegin", '<section class="panel"><h2>The forecast data did not load</h2><p class="note">Reload the page in a minute. The dashboard rebuilds every morning.</p></section>');
  });
