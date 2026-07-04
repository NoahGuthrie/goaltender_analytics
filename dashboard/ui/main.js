/* ── GSAx 2.0 — Dashboard JS ───────────────────────────── */

const S = {
  meta: { seasons: [], teams: [] },
  lb: [],
  sort: { col: "gsax", dir: "desc" },
  filters: { season: null, team: null, minShots: 200 },
  goalies: [],
};

/* ── API helper ──────────────────────────────────────────── */

async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.status + " " + path);
  return r.json();
}

/* ── Boot ─────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", async () => {
  try { S.meta = await api("/api/meta"); } catch (e) { console.warn("meta failed", e); }
  route();
});
window.addEventListener("hashchange", route);

function route() {
  const h = location.hash || "#leaderboard";
  document.querySelectorAll("nav .nav-link").forEach(a => {
    a.classList.toggle("active", a.getAttribute("href") === h.split("/")[0]);
  });
  const m = document.getElementById("main-content");
  if (h.startsWith("#goalie/"))        return viewGoalie(m, +h.split("/")[1]);
  if (h.startsWith("#projections"))    return viewProjections(m);
  if (h.startsWith("#head2head"))      return viewH2H(m);
  if (h.startsWith("#defense-impact")) return viewDefense(m);
  return viewLeaderboard(m);
}

/* ════════════════════════════════════════════════════════════
   VIEW: LEADERBOARD
   ════════════════════════════════════════════════════════════ */

async function viewLeaderboard(m) {
  m.innerHTML =
    '<h2>Leaderboard</h2>' +
    '<p class="subtitle">All goalies ranked by cumulative GSAx 2.0.</p>' +
    filterHTML() +
    '<div id="tbl"></div>';
  bindFilters();
  await loadTable();
}

function filterHTML() {
  const ss = S.meta.seasons, ts = S.meta.teams, f = S.filters;
  return '<div class="filters">' +
    '<div class="filter-item"><label>Season</label><select id="f-season"><option value="">All Seasons</option>' +
    ss.map(s => '<option value="' + s + '"' + (f.season == s ? ' selected' : '') + '>' + fmtSeason(s) + '</option>').join('') +
    '</select></div>' +
    '<div class="filter-item"><label>Team</label><select id="f-team"><option value="">All Teams</option>' +
    ts.map(t => '<option value="' + t + '"' + (f.team === t ? ' selected' : '') + '>' + t + '</option>').join('') +
    '</select></div>' +
    '<div class="filter-item"><label>Min Shots</label><input id="f-shots" type="number" value="' + f.minShots + '" min="1" step="50" style="width:80px"></div>' +
    '<span class="filter-count" id="f-count"></span>' +
    '</div>';
}

function bindFilters() {
  document.getElementById("f-season").onchange = e => { S.filters.season = e.target.value || null; loadTable(); };
  document.getElementById("f-team").onchange   = e => { S.filters.team   = e.target.value || null; loadTable(); };
  document.getElementById("f-shots").onchange  = e => { S.filters.minShots = +e.target.value || 200; loadTable(); };
}

async function loadTable() {
  const el = document.getElementById("tbl");
  el.innerHTML = '<p class="state-msg">Loading…</p>';
  const p = new URLSearchParams();
  if (S.filters.season) p.set("season", S.filters.season);
  if (S.filters.team)   p.set("team", S.filters.team);
  p.set("min_shots", S.filters.minShots);
  try {
    S.lb = await api("/api/leaderboard?" + p);
    S.goalies = S.lb.map(g => ({ id: g.goalie_id, name: g.goalie_name }));
    renderTable();
  } catch (e) {
    el.innerHTML = '<p class="state-msg err">Failed to load data. Is the backend running?</p>';
  }
}

function renderTable() {
  const data = [...S.lb];
  const { col, dir } = S.sort;
  data.sort((a, b) => {
    const va = a[col] ?? -Infinity, vb = b[col] ?? -Infinity;
    return dir === "desc" ? vb - va : va - vb;
  });
  document.getElementById("f-count").textContent = data.length + " goalies";

  const cols = [
    { k: "_rk",   l: "Rk",         fn: (_, i) => i + 1 },
    { k: "goalie_name", l: "Goalie", fn: g => '<a href="#goalie/' + g.goalie_id + '">' + g.goalie_name + '</a>', cls: "name-cell" },
    { k: "shots", l: "Shots",       fn: g => g.shots.toLocaleString(), cls: "right" },
    { k: "sv_pct",l: "Sv%",         fn: g => fmtSvPct(g.sv_pct), cls: "right mono" },
    { k: "gsax",  l: "GSAx",        fn: g => colorVal(g.gsax, v => (v >= 0 ? "+" : "") + v.toFixed(1)), cls: "right bold" },
    { k: "gsax_pct", l: "%ile",     fn: g => g.gsax_pct != null ? g.gsax_pct.toFixed(0) : "—", cls: "right" },
    { k: "rci",   l: "RCI",         fn: g => g.rci != null ? g.rci.toFixed(2) : "—", cls: "right mono" },
    { k: "isolated_talent", l: "DSIS Talent", fn: g => g.isolated_talent != null ? fmtTalent(g.isolated_talent) : "—", cls: "right mono" },
  ];

  let html = '<div class="tbl-wrap"><table><thead><tr>';
  for (const c of cols) {
    const sorted = S.sort.col === c.k;
    const arrow = sorted ? (S.sort.dir === "desc" ? "▾" : "▴") : "";
    html += '<th class="' + (sorted ? "sorted" : "") + '" data-col="' + c.k + '">' + c.l + '<span class="arrow">' + arrow + '</span></th>';
  }
  html += '</tr></thead><tbody>';
  for (let i = 0; i < data.length; i++) {
    const g = data[i];
    html += '<tr data-id="' + g.goalie_id + '">';
    for (const c of cols) html += '<td class="' + (c.cls || '') + '">' + c.fn(g, i) + '</td>';
    html += '</tr>';
  }
  html += '</tbody></table></div>';
  document.getElementById("tbl").innerHTML = html;

  // Sort handlers
  document.querySelectorAll("thead th[data-col]").forEach(th => {
    th.onclick = () => {
      const k = th.dataset.col;
      if (k === "_rk") return;
      if (S.sort.col === k) S.sort.dir = S.sort.dir === "desc" ? "asc" : "desc";
      else S.sort = { col: k, dir: "desc" };
      renderTable();
    };
  });
  // Row click (but not on links)
  document.querySelectorAll("tbody tr[data-id]").forEach(tr => {
    tr.onclick = e => { if (e.target.tagName !== "A") location.hash = "#goalie/" + tr.dataset.id; };
  });
}

/* ════════════════════════════════════════════════════════════
   VIEW: GOALIE DETAIL
   ════════════════════════════════════════════════════════════ */

async function viewGoalie(m, id) {
  m.innerHTML = '<p class="state-msg">Loading…</p>';
  try {
    const d = await api("/api/goalie/" + id);
    const career = d.career || [];
    const shots  = d.recent_shots || [];
    const proj   = d.projection || {};
    const base   = d.base_metrics || [];
    const name   = d.name || String(id);

    const totShots = career.reduce((s, r) => s + r.shots, 0);
    const totGoals = career.reduce((s, r) => s + r.goals, 0);
    const totGsax  = career.reduce((s, r) => s + (r.gsax || 0), 0);
    const svPct    = totShots > 0 ? 1 - totGoals / totShots : 0;
    const avgRci   = base.length ? (base.reduce((s, r) => s + (r.rci || 0), 0) / base.length) : null;
    const projGsax = proj.proj_1yr_talent_per_shot != null ? proj.proj_1yr_talent_per_shot * 1500 : null;

    m.innerHTML =
      '<div class="detail-head">' +
        '<a href="#leaderboard" class="back-link">← Leaderboard</a>' +
        '<h2>' + name + '</h2>' +
      '</div>' +
      '<div class="kpi-row">' +
        kpi("Career Shots", totShots.toLocaleString()) +
        kpi("Career Sv%", fmtSvPct(svPct)) +
        kpi("GSAx Total", colorVal(totGsax, v => (v >= 0 ? "+" : "") + v.toFixed(1))) +
        kpi("Avg RCI", avgRci != null ? avgRci.toFixed(2) : "—") +
        kpi("Proj. GSAx", projGsax != null ? colorVal(projGsax, v => (v >= 0 ? "+" : "") + v.toFixed(1)) : "—") +
      '</div>' +
      '<div class="grid-2">' +
        '<div class="panel"><div class="panel-head">Career GSAx by Season</div><div class="panel-body chart-box" id="ch-career"></div></div>' +
        '<div class="panel"><div class="panel-head">Shot Map — Defensive Zone</div><div class="panel-body chart-box" id="ch-shots"></div></div>' +
      '</div>' +
      seasonTableHTML(career);

    requestAnimationFrame(() => {
      drawCareer(career);
      drawShotMap(shots);
    });
  } catch (e) {
    m.innerHTML = '<p class="state-msg err">Failed to load goalie.</p>';
    console.error(e);
  }
}

function seasonTableHTML(career) {
  if (!career.length) return '';
  let h = '<div class="tbl-wrap"><table><thead><tr>' +
    '<th>Season</th><th>Shots</th><th>Goals</th><th>Sv%</th><th>GSAx</th>' +
    '</tr></thead><tbody>';
  for (const r of career) {
    h += '<tr>' +
      '<td>' + fmtSeason(r.season) + '</td>' +
      '<td class="right">' + r.shots.toLocaleString() + '</td>' +
      '<td class="right">' + r.goals + '</td>' +
      '<td class="right mono">' + fmtSvPct(r.sv_pct) + '</td>' +
      '<td class="right bold">' + colorVal(r.gsax, v => (v >= 0 ? "+" : "") + v.toFixed(1)) + '</td>' +
      '</tr>';
  }
  h += '</tbody></table></div>';
  return h;
}

function kpi(label, value) {
  return '<div class="kpi"><div class="kpi-label">' + label + '</div><div class="kpi-val">' + value + '</div></div>';
}

/* ── Career Bar Chart ────────────────────────────────────── */

function drawCareer(career) {
  const box = document.getElementById("ch-career");
  if (!box || !career.length) return;
  const W = box.clientWidth, H = 260;
  const mg = { t: 15, r: 15, b: 30, l: 40 };

  const svg = d3.select(box).append("svg").attr("viewBox", "0 0 " + W + " " + H);

  const x = d3.scaleBand().domain(career.map(d => d.season)).range([mg.l, W - mg.r]).padding(0.35);
  const yMax = d3.max(career, d => Math.abs(d.gsax)) || 10;
  const y = d3.scaleLinear().domain([-yMax * 1.15, yMax * 1.15]).range([H - mg.b, mg.t]);

  // grid
  svg.selectAll(".grid").data(y.ticks(5)).enter().append("line")
    .attr("x1", mg.l).attr("x2", W - mg.r)
    .attr("y1", d => y(d)).attr("y2", d => y(d))
    .attr("stroke", "#eee");
  // zero line
  svg.append("line").attr("x1", mg.l).attr("x2", W - mg.r)
    .attr("y1", y(0)).attr("y2", y(0)).attr("stroke", "#bbb");
  // bars
  svg.selectAll("rect").data(career).enter().append("rect")
    .attr("x", d => x(d.season)).attr("width", x.bandwidth())
    .attr("y", d => d.gsax >= 0 ? y(d.gsax) : y(0))
    .attr("height", d => Math.abs(y(d.gsax) - y(0)))
    .attr("fill", d => d.gsax >= 0 ? "var(--pos)" : "var(--neg)");
  // axes
  svg.append("g").attr("transform", "translate(0," + (H - mg.b) + ")")
    .call(d3.axisBottom(x).tickFormat(d => String(d).slice(2, 4) + "-" + String(d).slice(6)))
    .selectAll("text").attr("font-size", "10px").attr("fill", "#777");
  svg.append("g").attr("transform", "translate(" + mg.l + ",0)")
    .call(d3.axisLeft(y).ticks(5).tickSize(-W + mg.l + mg.r))
    .selectAll("text").attr("font-size", "10px").attr("fill", "#777");
  svg.selectAll(".tick line").attr("stroke", "#eee");
  svg.select(".domain").remove();
}

/* ── Shot Map with Defensive Zone ────────────────────────── */

function drawShotMap(shots) {
  const box = document.getElementById("ch-shots");
  if (!box) return;

  // NHL coordinates: net at x=89, blue line at x=25, boards at x≈100
  // We show defensive zone: x ∈ [25, 100], y ∈ [-42.5, 42.5]
  const W = box.clientWidth, H = 260;
  const pad = 10;

  const svg = d3.select(box).append("svg").attr("viewBox", "0 0 " + W + " " + H);
  const g = svg.append("g");

  // scale: map rink coords to pixel coords
  const xS = d3.scaleLinear().domain([24, 101]).range([pad, W - pad]);
  const yS = d3.scaleLinear().domain([-44, 44]).range([H - pad, pad]);

  // ─ Rink markings ─

  // Boards (rounded end)
  const boardColor = "#bbb";
  // straight sides
  g.append("line").attr("x1", xS(25)).attr("y1", yS(-42.5)).attr("x2", xS(89)).attr("y2", yS(-42.5)).attr("stroke", boardColor);
  g.append("line").attr("x1", xS(25)).attr("y1", yS(42.5)).attr("x2", xS(89)).attr("y2", yS(42.5)).attr("stroke", boardColor);
  // blue line
  g.append("line").attr("x1", xS(25)).attr("y1", yS(-42.5)).attr("x2", xS(25)).attr("y2", yS(42.5)).attr("stroke", "#2471a3").attr("stroke-width", 2.5);
  // end boards (arc behind net)
  const arcPath = d3.arc()({ innerRadius: 0, outerRadius: 1, startAngle: 0, endAngle: Math.PI });
  // approximate rounded corners with a path
  const boardPath = "M " + xS(89) + " " + yS(-42.5) +
    " Q " + xS(100) + " " + yS(-42.5) + " " + xS(100) + " " + yS(-28) +
    " L " + xS(100) + " " + yS(28) +
    " Q " + xS(100) + " " + yS(42.5) + " " + xS(89) + " " + yS(42.5);
  g.append("path").attr("d", boardPath).attr("fill", "none").attr("stroke", boardColor);

  // Goal line
  g.append("line").attr("x1", xS(89)).attr("y1", yS(-42.5)).attr("x2", xS(89)).attr("y2", yS(42.5))
    .attr("stroke", "#e74c3c").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3");

  // Net (rectangle behind goal line)
  g.append("rect").attr("x", xS(89)).attr("y", yS(3)).attr("width", xS(93) - xS(89)).attr("height", yS(-3) - yS(3))
    .attr("fill", "none").attr("stroke", "#555").attr("stroke-width", 1.5);

  // Crease (semicircle, 6ft radius in front of net)
  const creaseR = Math.abs(xS(89) - xS(83));
  g.append("path")
    .attr("d", d3.arc()({ innerRadius: 0, outerRadius: creaseR, startAngle: -Math.PI / 2, endAngle: Math.PI / 2 }))
    .attr("transform", "translate(" + xS(89) + "," + yS(0) + ")")
    .attr("fill", "rgba(173, 216, 230, 0.15)").attr("stroke", "#e74c3c").attr("stroke-width", 1);

  // Faceoff circles (at x=69, y=±22)
  const foRadius = Math.abs(xS(69) - xS(54)); // 15ft radius
  [22, -22].forEach(fy => {
    g.append("circle").attr("cx", xS(69)).attr("cy", yS(fy)).attr("r", foRadius)
      .attr("fill", "none").attr("stroke", "#c0392b").attr("stroke-width", 0.7);
    g.append("circle").attr("cx", xS(69)).attr("cy", yS(fy)).attr("r", 2.5)
      .attr("fill", "#c0392b");
  });

  // ─ Shots ─
  const valid = shots.filter(d => d.adjusted_x != null && d.adjusted_y != null);
  if (!valid.length) {
    svg.append("text").attr("x", W / 2).attr("y", H / 2).attr("text-anchor", "middle")
      .attr("fill", "#999").attr("font-size", "12px").text("No shot data available");
    return;
  }

  // saves
  g.selectAll(".save").data(valid.filter(d => !d.is_goal)).enter().append("circle")
    .attr("cx", d => xS(d.adjusted_x)).attr("cy", d => yS(d.adjusted_y))
    .attr("r", 2.5).attr("fill", "rgba(100,100,100,0.2)");
  // goals
  g.selectAll(".goal").data(valid.filter(d => d.is_goal)).enter().append("circle")
    .attr("cx", d => xS(d.adjusted_x)).attr("cy", d => yS(d.adjusted_y))
    .attr("r", 4).attr("fill", "var(--neg)").attr("stroke", "#fff").attr("stroke-width", 0.5);

  // Legend
  const ly = H - 5;
  g.append("circle").attr("cx", xS(30)).attr("cy", ly - 3).attr("r", 2.5).attr("fill", "rgba(100,100,100,0.4)");
  g.append("text").attr("x", xS(30) + 7).attr("y", ly).attr("font-size", "10px").attr("fill", "#777").text("Save");
  g.append("circle").attr("cx", xS(45)).attr("cy", ly - 3).attr("r", 3.5).attr("fill", "var(--neg)");
  g.append("text").attr("x", xS(45) + 7).attr("y", ly).attr("font-size", "10px").attr("fill", "#777").text("Goal");
}

/* ════════════════════════════════════════════════════════════
   VIEW: PROJECTIONS
   ════════════════════════════════════════════════════════════ */

async function viewProjections(m) {
  m.innerHTML =
    '<h2>True Talent Projections</h2>' +
    '<p class="subtitle">Kalman filter projections normalized to a 1,500-shot season.</p>' +
    '<div id="proj-tbl"><p class="state-msg">Loading…</p></div>';
  try {
    const data = await api("/api/projections");
    if (!data || !data.length) { document.getElementById("proj-tbl").innerHTML = '<p class="state-msg">No projection data.</p>'; return; }
    data.sort((a, b) => (b.proj_1yr_talent_per_shot || -Infinity) - (a.proj_1yr_talent_per_shot || -Infinity));

    let h = '<div class="tbl-wrap"><table><thead><tr>' +
      '<th>Rk</th><th>Goalie</th><th>Proj. GSAx</th><th>80% CI</th><th>Career Shots</th>' +
      '</tr></thead><tbody>';
    data.forEach((g, i) => {
      const pr = g.proj_1yr_talent_per_shot != null ? g.proj_1yr_talent_per_shot * 1500 : null;
      const lo = g.proj_1yr_ci_lower != null ? g.proj_1yr_ci_lower * 1500 : null;
      const hi = g.proj_1yr_ci_upper != null ? g.proj_1yr_ci_upper * 1500 : null;
      h += '<tr data-id="' + g.goalie_id + '">' +
        '<td class="right">' + (i + 1) + '</td>' +
        '<td class="name-cell"><a href="#goalie/' + g.goalie_id + '">' + g.goalie_name + '</a></td>' +
        '<td class="right bold">' + (pr != null ? colorVal(pr, v => (v >= 0 ? "+" : "") + v.toFixed(1)) : "—") + '</td>' +
        '<td class="right mono muted">' + (lo != null && hi != null ? "[" + lo.toFixed(1) + ", " + hi.toFixed(1) + "]" : "—") + '</td>' +
        '<td class="right">' + (g.career_shots != null ? g.career_shots.toLocaleString() : "—") + '</td>' +
        '</tr>';
    });
    h += '</tbody></table></div>';
    document.getElementById("proj-tbl").innerHTML = h;
    document.querySelectorAll("#proj-tbl tr[data-id]").forEach(tr => {
      tr.onclick = e => { if (e.target.tagName !== "A") location.hash = "#goalie/" + tr.dataset.id; };
    });
  } catch (e) {
    document.getElementById("proj-tbl").innerHTML = '<p class="state-msg err">Failed to load projections.</p>';
  }
}

/* ════════════════════════════════════════════════════════════
   VIEW: HEAD-TO-HEAD
   ════════════════════════════════════════════════════════════ */

async function viewH2H(m) {
  if (!S.goalies.length) {
    try { const d = await api("/api/leaderboard?min_shots=200"); S.goalies = d.map(g => ({ id: g.goalie_id, name: g.goalie_name })); }
    catch { m.innerHTML = '<p class="state-msg err">Failed to load goalies.</p>'; return; }
  }
  const opts = S.goalies.map((g, i) => '<option value="' + g.id + '"' + (i === 1 ? ' selected' : '') + '>' + g.name + '</option>').join('');
  const seasOpts = S.meta.seasons.map(s => '<option value="' + s + '">' + fmtSeason(s) + '</option>').join('');

  m.innerHTML =
    '<h2>Head-to-Head</h2>' +
    '<p class="subtitle">Compare two goalies side by side.</p>' +
    '<div class="h2h-bar">' +
      '<select id="h2h-1">' + opts.replace(' selected', '') + '</select>' +
      '<span class="h2h-vs">vs</span>' +
      '<select id="h2h-2">' + opts + '</select>' +
      '<select id="h2h-s"><option value="">Career</option>' + seasOpts + '</select>' +
    '</div>' +
    '<div id="h2h-out"></div>';

  const go = async () => {
    const g1 = document.getElementById("h2h-1").value;
    const g2 = document.getElementById("h2h-2").value;
    const s  = document.getElementById("h2h-s").value;
    const p = new URLSearchParams({ g1, g2 });
    if (s) p.set("season", s);
    const out = document.getElementById("h2h-out");
    out.innerHTML = '<p class="state-msg">Loading…</p>';
    try {
      const data = await api("/api/head2head?" + p);
      if (data.length < 2) { out.innerHTML = '<p class="state-msg">Not enough data.</p>'; return; }
      out.innerHTML = renderH2H(data);
    } catch { out.innerHTML = '<p class="state-msg err">Error loading comparison.</p>'; }
  };

  document.getElementById("h2h-1").onchange = go;
  document.getElementById("h2h-2").onchange = go;
  document.getElementById("h2h-s").onchange = go;
  go();
}

function renderH2H(data) {
  const metrics = [
    { k: "shots",        l: "Shots Faced",  fn: v => v.toLocaleString() },
    { k: "sv_pct",       l: "Save %",       fn: v => fmtSvPct(v) },
    { k: "gsax",         l: "GSAx",         fn: v => colorVal(v, x => (x >= 0 ? "+" : "") + x.toFixed(1)) },
    { k: "avg_traffic",  l: "Avg Traffic",  fn: v => v != null ? v.toFixed(2) : "—" },
    { k: "avg_speed",    l: "Avg Puck Speed",fn: v => v != null ? v.toFixed(1) : "—" },
    { k: "avg_movement", l: "Avg Lateral",  fn: v => v != null ? v.toFixed(1) + "°" : "—" },
  ];
  let h = '<div class="grid-2">';
  data.forEach(g => {
    h += '<div class="panel"><div class="panel-head">' + g.goalie_name + '</div>';
    h += '<div class="tbl-wrap"><table><tbody>';
    metrics.forEach(met => {
      const v = g[met.k];
      h += '<tr><td class="muted">' + met.l + '</td><td class="right bold">' + (v != null ? met.fn(v) : "—") + '</td></tr>';
    });
    h += '</tbody></table></div></div>';
  });
  h += '</div>';
  return h;
}

/* ════════════════════════════════════════════════════════════
   VIEW: TEAM DEFENSE
   ════════════════════════════════════════════════════════════ */

async function viewDefense(m) {
  m.innerHTML =
    '<h2>Team Defense Impact (DSIS)</h2>' +
    '<p class="subtitle">Bayesian estimates of each team\'s defensive system impact on expected goals per shot. Negative = better defense.</p>' +
    '<div class="filters"><div class="filter-item"><label>Season</label><select id="f-dsis-s"><option value="">All</option>' +
    S.meta.seasons.map(s => '<option value="' + s + '">' + fmtSeason(s) + '</option>').join('') +
    '</select></div></div>' +
    '<div id="dsis-tbl"><p class="state-msg">Loading…</p></div>';

  const go = async () => {
    const s = document.getElementById("f-dsis-s").value;
    const p = new URLSearchParams();
    if (s) p.set("season", s);
    const el = document.getElementById("dsis-tbl");
    try {
      const data = await api("/api/team-defense?" + p);
      if (!data.length) { el.innerHTML = '<p class="state-msg">No data.</p>'; return; }
      let h = '<div class="tbl-wrap"><table><thead><tr><th>Rk</th><th>Team</th><th>Season</th><th>Impact / Shot</th><th>95% CI</th></tr></thead><tbody>';
      data.forEach((t, i) => {
        const v = t.dsis_team_defense_impact_per_shot;
        const ci = t.dsis_team_std != null ? (t.dsis_team_std * 1.96).toFixed(4) : "—";
        h += '<tr><td class="right">' + (i + 1) + '</td><td class="bold">' + t.team_name + '</td>' +
          '<td>' + fmtSeason(t.season) + '</td>' +
          '<td class="right bold">' + colorVal(v, x => (x > 0 ? "+" : "") + x.toFixed(4), true) + '</td>' +
          '<td class="right mono muted">±' + ci + '</td></tr>';
      });
      h += '</tbody></table></div>';
      el.innerHTML = h;
    } catch { el.innerHTML = '<p class="state-msg err">Error loading DSIS data.</p>'; }
  };

  document.getElementById("f-dsis-s").onchange = go;
  go();
}

/* ════════════════════════════════════════════════════════════
   HELPERS
   ════════════════════════════════════════════════════════════ */

function fmtSeason(s) {
  const str = String(s);
  if (str.length === 8) return str.slice(0, 4) + "-" + str.slice(6);
  return str;
}

function fmtSvPct(v) {
  if (v == null) return "—";
  return "." + (v * 1000).toFixed(0);
}

function fmtTalent(v) {
  return (v >= 0 ? "+" : "") + v.toFixed(4);
}

// colorVal: wraps formatted string in pos/neg span
// invertGood: if true, negative is good (for DSIS)
function colorVal(v, fmt, invertGood) {
  if (v == null) return "—";
  const good = invertGood ? v < 0 : v >= 0;
  return '<span class="' + (good ? "pos" : "neg") + '">' + fmt(v) + '</span>';
}
