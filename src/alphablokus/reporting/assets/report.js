/* AlphaBlokus run report — renders the embedded JSON payload client-side.
 *
 * No frameworks, no fetches, no build step: everything below is hand-rolled
 * SVG so the report opens from file:// with zero network access. Charts
 * re-render on resize and on theme toggle (colours are read from CSS
 * variables at render time).
 */
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("report-data").textContent);

  // ------------------------------------------------------------------
  // DOM + SVG helpers
  // ------------------------------------------------------------------

  var SVG_NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs) {
    var node = document.createElement(tag);
    applyAttrs(node, attrs);
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }

  function svgEl(tag, attrs) {
    var node = document.createElementNS(SVG_NS, tag);
    applyAttrs(node, attrs);
    for (var i = 2; i < arguments.length; i++) append(node, arguments[i]);
    return node;
  }

  function applyAttrs(node, attrs) {
    if (!attrs) return;
    Object.keys(attrs).forEach(function (key) {
      var value = attrs[key];
      if (value === null || value === undefined) return;
      if (key === "class") node.setAttribute("class", value);
      else if (key === "text") node.textContent = value;
      else if (key === "html") node.innerHTML = value;
      else if (key.indexOf("on") === 0) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value);
    });
  }

  function append(parent, child) {
    if (child === null || child === undefined || child === false) return;
    if (Array.isArray(child)) { child.forEach(function (c) { append(parent, c); }); return; }
    if (typeof child === "string") { parent.appendChild(document.createTextNode(child)); return; }
    parent.appendChild(child);
  }

  function cssVar(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function fmtNum(v) {
    if (v === null || v === undefined || isNaN(v)) return "—";
    var abs = Math.abs(v);
    if (abs >= 1e6) return (v / 1e6).toFixed(1) + "M";
    if (abs >= 1e4) return (v / 1e3).toFixed(1) + "k";
    if (abs >= 100) return String(Math.round(v));
    if (abs >= 1) return String(parseFloat(v.toPrecision(4)));
    if (abs === 0) return "0";
    return String(parseFloat(v.toPrecision(3)));
  }

  function fmtPct(v) { return Math.round(v * 100) + "%"; }

  function niceTicks(lo, hi, count) {
    if (!isFinite(lo) || !isFinite(hi)) { lo = 0; hi = 1; }
    if (lo === hi) { lo -= 1; hi += 1; }
    var span = hi - lo;
    var step0 = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = span / count / step0;
    var step = step0 * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
    var ticks = [];
    for (var v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
      ticks.push(Math.abs(v) < step * 1e-6 ? 0 : v);
    }
    return ticks;
  }

  // ------------------------------------------------------------------
  // Theme + chart registry (charts re-render on toggle / resize)
  // ------------------------------------------------------------------

  var chartRegistry = [];

  function registerChart(renderFn) {
    chartRegistry.push(renderFn);
    renderFn();
  }

  function rerenderCharts() {
    chartRegistry.forEach(function (fn) { fn(); });
  }

  function initTheme() {
    var stored = null;
    try { stored = localStorage.getItem("alphablokus-report-theme"); } catch (e) { /* file:// may block */ }
    var preferred = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    setTheme(stored || preferred, false);
  }

  function setTheme(theme, rerender) {
    document.documentElement.setAttribute("data-theme", theme);
    var button = document.querySelector(".theme-toggle");
    if (button) button.textContent = theme === "dark" ? "☀ Light" : "◐ Dark";
    try { localStorage.setItem("alphablokus-report-theme", theme); } catch (e) { /* ignore */ }
    if (rerender) rerenderCharts();
  }

  function toggleTheme() {
    var current = document.documentElement.getAttribute("data-theme") || "light";
    setTheme(current === "dark" ? "light" : "dark", true);
  }

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(rerenderCharts, 150);
  });

  function palette() {
    return {
      accent: cssVar("--accent"),
      violet: "#8a63d2",
      ok: cssVar("--ok"),
      warn: cssVar("--warn"),
      alert: cssVar("--alert"),
      gray: cssVar("--ink-3"),
      ink: cssVar("--ink"),
      surface: cssVar("--surface"),
      whiteP: cssVar("--white-player"),
      blackP: cssVar("--black-player"),
    };
  }

  // ------------------------------------------------------------------
  // Tooltip singleton
  // ------------------------------------------------------------------

  var tooltip = null;

  function showTooltip(clientX, clientY, titleText, rows) {
    if (!tooltip) {
      tooltip = el("div", { class: "tooltip" });
      document.body.appendChild(tooltip);
    }
    tooltip.innerHTML = "";
    append(tooltip, el("div", { class: "t-title", text: titleText }));
    rows.forEach(function (row) {
      append(tooltip, el("div", { class: "t-row" },
        el("span", { class: "swatch", style: "background:" + row.color }),
        row.label + ": ",
        el("strong", { text: row.value })));
    });
    tooltip.style.display = "block";
    var rect = tooltip.getBoundingClientRect();
    var x = clientX + 14, y = clientY + 12;
    if (x + rect.width > window.innerWidth - 8) x = clientX - rect.width - 14;
    if (y + rect.height > window.innerHeight - 8) y = clientY - rect.height - 12;
    tooltip.style.left = x + "px";
    tooltip.style.top = y + "px";
  }

  function hideTooltip() {
    if (tooltip) tooltip.style.display = "none";
  }

  // ------------------------------------------------------------------
  // Line chart
  // ------------------------------------------------------------------
  //
  // spec = {
  //   series: [{name, x: [], y: [], color, dash, width, dots: true|[bool per pt],
  //             band: {lo: [], hi: []}}],
  //   height, yLabel, xLabel, yFmt, xInt, includeZero, diag,
  //   hlines: [{y, label, color, dash}], vmarks: [{x, label}], legend
  // }

  function lineChart(container, specFn) {
    var hidden = {};
    registerChart(function () { renderLineChart(container, specFn(), hidden); });
  }

  function renderLineChart(container, spec, hidden) {
    container.innerHTML = "";
    var pal = palette();
    var series = spec.series.filter(function (s) { return s.y && s.y.length; });
    if (!series.length) return;

    var showLegend = spec.legend !== false && series.length > 1;
    if (showLegend) {
      var legend = el("div", { class: "legend" });
      series.forEach(function (s) {
        var item = el("span", { class: "item" + (hidden[s.name] ? " off" : "") },
          el("span", { class: "swatch", style: "background:" + s.color }),
          s.name);
        item.addEventListener("click", function () {
          hidden[s.name] = !hidden[s.name];
          renderLineChart(container, spec, hidden);
        });
        append(legend, item);
      });
      container.appendChild(legend);
    }

    var visible = series.filter(function (s) { return !hidden[s.name]; });
    if (!visible.length) visible = series;

    var W = Math.max(300, container.clientWidth || 640);
    var H = spec.height || 240;
    var m = { t: 12, r: 16, b: spec.xLabel ? 40 : 26, l: 54 };

    var xs = [], ys = [];
    visible.forEach(function (s) {
      xs = xs.concat(s.x);
      ys = ys.concat(s.y.filter(function (v) { return v !== null && !isNaN(v); }));
      if (s.band) ys = ys.concat(s.band.lo, s.band.hi);
    });
    (spec.hlines || []).forEach(function (h) { ys.push(h.y); });
    var xMin = Math.min.apply(null, xs), xMax = Math.max.apply(null, xs);
    var yMin = Math.min.apply(null, ys), yMax = Math.max.apply(null, ys);
    if (spec.includeZero) { yMin = Math.min(yMin, 0); yMax = Math.max(yMax, 0); }
    if (spec.diag) { yMin = Math.min(yMin, xMin); yMax = Math.max(yMax, xMax); }
    var pad = (yMax - yMin || 1) * 0.08;
    yMin -= pad; yMax += pad;
    if (xMin === xMax) { xMin -= 1; xMax += 1; }

    var plotW = W - m.l - m.r, plotH = H - m.t - m.b;
    function sx(v) { return m.l + ((v - xMin) / (xMax - xMin)) * plotW; }
    function sy(v) { return m.t + plotH - ((v - yMin) / (yMax - yMin)) * plotH; }

    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });

    var yTicks = niceTicks(yMin, yMax, 5);
    yTicks.forEach(function (t) {
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: sy(t), y2: sy(t) }));
      svg.appendChild(svgEl("text", { class: "tick-label", x: m.l - 7, y: sy(t) + 3.5, "text-anchor": "end",
        text: (spec.yFmt || fmtNum)(t) }));
    });

    var xTicks = niceTicks(xMin, xMax, Math.min(8, Math.max(3, Math.floor(plotW / 90))));
    if (spec.xInt) xTicks = xTicks.filter(function (t) { return Math.abs(t - Math.round(t)) < 1e-9; });
    xTicks.forEach(function (t) {
      svg.appendChild(svgEl("text", { class: "tick-label", x: sx(t), y: H - m.b + 16, "text-anchor": "middle",
        text: (spec.xFmt || fmtNum)(t) }));
    });

    if (spec.xLabel) {
      svg.appendChild(svgEl("text", { class: "axis-label", x: m.l + plotW / 2, y: H - 6, "text-anchor": "middle",
        text: spec.xLabel }));
    }
    if (spec.yLabel) {
      svg.appendChild(svgEl("text", { class: "axis-label", x: 12, y: m.t + plotH / 2, "text-anchor": "middle",
        transform: "rotate(-90 12 " + (m.t + plotH / 2) + ")", text: spec.yLabel }));
    }

    (spec.vmarks || []).forEach(function (mark) {
      var x = sx(mark.x);
      if (x < m.l - 1 || x > W - m.r + 1) return;
      svg.appendChild(svgEl("line", { x1: x, x2: x, y1: m.t, y2: m.t + plotH,
        stroke: pal.gray, "stroke-width": 0.7, "stroke-dasharray": "2 3", opacity: 0.6 }));
      if (mark.label) {
        svg.appendChild(svgEl("text", { class: "tick-label", x: x + 3, y: m.t + 9, text: mark.label, opacity: 0.8 }));
      }
    });

    if (spec.diag) {
      svg.appendChild(svgEl("line", { x1: sx(Math.max(xMin, yMin)), y1: sy(Math.max(xMin, yMin)),
        x2: sx(Math.min(xMax, yMax)), y2: sy(Math.min(xMax, yMax)),
        stroke: pal.gray, "stroke-width": 1, "stroke-dasharray": "4 4", opacity: 0.7 }));
    }

    (spec.hlines || []).forEach(function (h) {
      svg.appendChild(svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(h.y), y2: sy(h.y),
        stroke: h.color || pal.gray, "stroke-width": 1, "stroke-dasharray": h.dash || "5 4" }));
      if (h.label) {
        svg.appendChild(svgEl("text", { class: "tick-label", x: W - m.r, y: sy(h.y) - 5, "text-anchor": "end",
          fill: h.color || pal.gray, text: h.label }));
      }
    });

    visible.forEach(function (s) {
      if (s.band) {
        var d = "M";
        s.x.forEach(function (x, i) { d += sx(x) + " " + sy(s.band.lo[i]) + " L"; });
        for (var i = s.x.length - 1; i >= 0; i--) d += sx(s.x[i]) + " " + sy(s.band.hi[i]) + (i ? " L" : "");
        svg.appendChild(svgEl("path", { d: d + " Z", fill: s.color, opacity: 0.12, stroke: "none" }));
      }
    });

    visible.forEach(function (s) {
      var d = "";
      s.x.forEach(function (x, i) {
        var y = s.y[i];
        if (y === null || isNaN(y)) return;
        d += (d ? " L" : "M") + sx(x).toFixed(1) + " " + sy(y).toFixed(1);
      });
      svg.appendChild(svgEl("path", { d: d, fill: "none", stroke: s.color,
        "stroke-width": s.width || 1.8, "stroke-dasharray": s.dash || null,
        "stroke-linejoin": "round", "stroke-linecap": "round" }));
      if (s.dots) {
        s.x.forEach(function (x, i) {
          if (s.y[i] === null || isNaN(s.y[i])) return;
          var filled = s.dots === true || s.dots[i];
          svg.appendChild(svgEl("circle", { cx: sx(x), cy: sy(s.y[i]), r: 3,
            fill: filled ? s.color : pal.surface, stroke: s.color, "stroke-width": 1.4 }));
        });
      }
    });

    // Hover: nearest shared-x tooltip + guide line.
    var guide = svgEl("line", { y1: m.t, y2: m.t + plotH, stroke: pal.gray, "stroke-width": 0.8, opacity: 0 });
    svg.appendChild(guide);
    var overlay = svgEl("rect", { x: m.l, y: m.t, width: plotW, height: plotH, fill: "transparent" });
    overlay.addEventListener("pointermove", function (event) {
      var rect = svg.getBoundingClientRect();
      var px = (event.clientX - rect.left) * (W / rect.width);
      var xVal = xMin + ((px - m.l) / plotW) * (xMax - xMin);
      var best = null;
      visible.forEach(function (s) {
        s.x.forEach(function (x, i) {
          var dist = Math.abs(x - xVal);
          if (!best || dist < best.dist) best = { dist: dist, x: x };
        });
      });
      if (!best) return;
      guide.setAttribute("x1", sx(best.x));
      guide.setAttribute("x2", sx(best.x));
      guide.setAttribute("opacity", 0.5);
      var rows = [];
      visible.forEach(function (s) {
        var i = s.x.indexOf(best.x);
        if (i < 0 || s.y[i] === null || isNaN(s.y[i])) return;
        rows.push({ color: s.color, label: s.name, value: (spec.yFmt || fmtNum)(s.y[i]) });
      });
      showTooltip(event.clientX, event.clientY, (spec.xTitle || "x") + " " + (spec.xFmt || fmtNum)(best.x), rows);
    });
    overlay.addEventListener("pointerleave", function () {
      guide.setAttribute("opacity", 0);
      hideTooltip();
    });
    svg.appendChild(overlay);

    var holder = el("div", { class: "chart" });
    holder.appendChild(svg);
    container.appendChild(holder);
  }

  // ------------------------------------------------------------------
  // Stacked bar chart
  // ------------------------------------------------------------------
  //
  // spec = {x: [], stacks: [{name, y: [], color}], height, yLabel, xTitle, yFmt}

  function barChart(container, specFn) {
    registerChart(function () { renderBarChart(container, specFn()); });
  }

  function renderBarChart(container, spec) {
    container.innerHTML = "";
    var pal = palette();
    var W = Math.max(300, container.clientWidth || 640);
    var H = spec.height || 240;
    var m = { t: 12, r: 16, b: 28, l: 54 };
    var plotW = W - m.l - m.r, plotH = H - m.t - m.b;

    var legend = el("div", { class: "legend" });
    spec.stacks.forEach(function (s) {
      append(legend, el("span", { class: "item" },
        el("span", { class: "swatch", style: "background:" + s.color }), s.name));
    });
    container.appendChild(legend);

    var totals = spec.x.map(function (_, i) {
      return spec.stacks.reduce(function (sum, s) { return sum + (s.y[i] || 0); }, 0);
    });
    var yMax = Math.max.apply(null, totals) * 1.06 || 1;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img" });

    niceTicks(0, yMax, 4).forEach(function (t) {
      var y = m.t + plotH - (t / yMax) * plotH;
      svg.appendChild(svgEl("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: y, y2: y }));
      svg.appendChild(svgEl("text", { class: "tick-label", x: m.l - 7, y: y + 3.5, "text-anchor": "end",
        text: (spec.yFmt || fmtNum)(t) }));
    });

    var band = plotW / spec.x.length;
    var barW = Math.max(2, Math.min(30, band * 0.7));
    var labelEvery = Math.ceil(spec.x.length / Math.max(3, Math.floor(plotW / 60)));

    spec.x.forEach(function (xVal, i) {
      var cx = m.l + band * (i + 0.5);
      var yCursor = m.t + plotH;
      spec.stacks.forEach(function (s) {
        var value = s.y[i] || 0;
        var h = (value / yMax) * plotH;
        if (h > 0) {
          svg.appendChild(svgEl("rect", { x: cx - barW / 2, y: yCursor - h, width: barW, height: h,
            fill: s.color, rx: 1 }));
        }
        yCursor -= h;
      });
      if (i % labelEvery === 0) {
        svg.appendChild(svgEl("text", { class: "tick-label", x: cx, y: H - m.b + 16, "text-anchor": "middle",
          text: String(xVal) }));
      }
      var hover = svgEl("rect", { x: cx - band / 2, y: m.t, width: band, height: plotH, fill: "transparent" });
      hover.addEventListener("pointermove", function (event) {
        var rows = spec.stacks.map(function (s) {
          return { color: s.color, label: s.name, value: (spec.yFmt || fmtNum)(s.y[i] || 0) };
        });
        showTooltip(event.clientX, event.clientY, (spec.xTitle || "x") + " " + xVal, rows);
      });
      hover.addEventListener("pointerleave", hideTooltip);
      svg.appendChild(hover);
    });

    if (spec.yLabel) {
      svg.appendChild(svgEl("text", { class: "axis-label", x: 12, y: m.t + plotH / 2, "text-anchor": "middle",
        transform: "rotate(-90 12 " + (m.t + plotH / 2) + ")", text: spec.yLabel }));
    }

    var holder = el("div", { class: "chart" });
    holder.appendChild(svg);
    container.appendChild(holder);
  }

  // ------------------------------------------------------------------
  // Sparkline (signal tiles)
  // ------------------------------------------------------------------

  function sparkline(values, width, height) {
    var min = Math.min.apply(null, values), max = Math.max.apply(null, values);
    if (min === max) { min -= 1; max += 1; }
    var points = values.map(function (v, i) {
      var x = (i / (values.length - 1)) * (width - 2) + 1;
      var y = height - 1.5 - ((v - min) / (max - min)) * (height - 3);
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    return svgEl("svg", { class: "s-spark", viewBox: "0 0 " + width + " " + height, width: width, height: height },
      svgEl("polyline", { points: points }));
  }

  // ------------------------------------------------------------------
  // Page scaffolding
  // ------------------------------------------------------------------

  function card(title, sub, bodyNode) {
    var c = el("div", { class: "card" }, el("h3", { text: title }));
    if (sub) append(c, el("p", { class: "card-sub", html: sub }));
    if (bodyNode) append(c, bodyNode);
    return c;
  }

  function chartCard(title, sub, renderInto) {
    var body = el("div");
    var c = card(title, sub, body);
    renderInto(body);
    return c;
  }

  function placeholder(title, why) {
    return el("div", { class: "placeholder" }, el("strong", { text: title + " — not recorded. " }), why);
  }

  function section(id, title, badgeKind, badgeText, desc) {
    var head = el("div", { class: "section-head" }, el("h2", { text: title }));
    if (badgeText) append(head, el("span", { class: "badge " + badgeKind, text: badgeText }));
    var node = el("section", { class: "report-section", id: id }, head);
    if (desc) append(node, el("p", { class: "section-desc", html: desc }));
    return node;
  }

  // ------------------------------------------------------------------
  // Header, verdict, signal tiles
  // ------------------------------------------------------------------

  function buildTopbar(sections) {
    var nav = el("nav");
    sections.forEach(function (s) { append(nav, el("a", { href: "#" + s.id, text: s.label })); });
    return el("div", { class: "topbar" },
      el("div", { class: "brand" }, "AlphaBlokus ", el("span", { text: "· " + DATA.meta.run_name })),
      nav,
      el("button", { class: "theme-toggle", onclick: toggleTheme, text: "◐ Dark" }));
  }

  function buildHeader() {
    var header = el("header", { class: "run-header" },
      el("h1", { text: DATA.meta.run_name }),
      el("div", { class: "sub", text: "Training run report · generated " + DATA.meta.date }));
    var chips = el("div", { class: "chips" });
    DATA.meta.chips.forEach(function (chip) { append(chips, el("span", { class: "chip", text: chip })); });
    append(header, chips);
    return header;
  }

  function buildVerdict() {
    var v = DATA.verdict;
    return el("div", { class: "verdict " + v.status },
      el("span", { class: "dot" }),
      el("div", null,
        el("div", { class: "s-label", style: "font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink-3);", text: "Is this run improving, or fooling itself?" }),
        el("h2", { text: v.headline }),
        el("p", { text: v.detail })));
  }

  function buildSignals() {
    var grid = el("div", { class: "signals" });
    DATA.signals.forEach(function (s) {
      var tile = el("a", { class: "signal " + s.status, href: s.href || "#" },
        el("div", { class: "s-label", text: s.label }),
        el("div", { class: "s-value", text: s.value }),
        el("div", { class: "s-sub", text: s.sub }));
      if (s.spark && s.spark.length > 2) append(tile, sparkline(s.spark, 52, 20));
      append(grid, tile);
    });
    return grid;
  }

  // ------------------------------------------------------------------
  // External evidence section
  // ------------------------------------------------------------------

  function withAlpha(hexColor, alpha) {
    var hex = hexColor.replace("#", "");
    if (hex.length === 3) hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
    var r = parseInt(hex.slice(0, 2), 16), g = parseInt(hex.slice(2, 4), 16), b = parseInt(hex.slice(4, 6), 16);
    return "rgba(" + r + "," + g + "," + b + "," + alpha.toFixed(3) + ")";
  }

  function rateColor(rate) {
    // Diverging around 50%: red (losing) → neutral → green (winning).
    var pal = palette();
    if (rate >= 0.5) return "background:" + withAlpha(pal.ok, (rate - 0.5) * 1.3 + 0.12);
    return "background:" + withAlpha(pal.alert, (0.5 - rate) * 1.3 + 0.12);
  }

  function buildLadderCard() {
    var ladder = DATA.ladder;
    if (!ladder || !ladder.entries.length) {
      var note = ladder && ladder.history.length
        ? "Mini-ladder history exists but no per-level result JSONs were found."
        : "The Pentobi ladder is the only instrument that has resolved differences the arena calls a tie. " +
          "Run scripts/mini_ladder.py (or scripts/pentobi_benchmark.py) against this run's checkpoints.";
      return placeholder("Pentobi ladder", note);
    }

    var body = el("div");

    if (ladder.drift || ladder.alarm_file) {
      var drift = ladder.drift;
      append(body, el("div", { class: "flag-banner" },
        el("strong", { text: "⚠ Drift circuit-breaker tripped. " }),
        drift
          ? drift.consecutive_drops + " consecutive evaluations ≥5pp below best (" + drift.best_before + " at " +
            drift.best_score.toFixed(3) + "); tripped at " + drift.tripped_at + " (" + drift.tripped_score.toFixed(3) +
            "). Resume from the keep-best checkpoint."
          : "MiniLadder/DRIFT_ALARM present in the run directory."));
    }

    var levels = [];
    ladder.entries.forEach(function (entry) {
      entry.levels.forEach(function (l) { if (levels.indexOf(l.level) < 0) levels.push(l.level); });
    });
    levels.sort(function (a, b) { return a - b; });

    var thead = el("tr", null, el("th", { text: "checkpoint" }), el("th", { text: "when" }), el("th", { text: "games" }));
    levels.forEach(function (l) { append(thead, el("th", { text: "L" + l })); });
    append(thead, el("th", { text: "weighted" }));

    var table = el("table", { class: "ladder" }, thead);
    var bestLabel = ladder.keep_best ? ladder.keep_best.label : null;
    ladder.entries.forEach(function (entry) {
      var row = el("tr", null,
        el("td", { class: "net-name" }, entry.net,
          entry.net === bestLabel ? el("span", { class: "keep-best-tag", text: "KEEP-BEST" }) : null),
        el("td", { class: "meta", text: entry.timestamp || "—" }),
        el("td", { class: "meta", text: (entry.games_per_level || "?") + "/lvl · " + (entry.sims || "?") + " sims" }));
      var byLevel = {};
      entry.levels.forEach(function (l) { byLevel[l.level] = l; });
      levels.forEach(function (l) {
        var cell = byLevel[l];
        if (!cell) { append(row, el("td", { class: "meta", text: "—" })); return; }
        var title = cell.wins + "–" + cell.losses + "–" + cell.draws +
          (cell.ci ? " · 95% CI [" + fmtPct(cell.ci[0]) + ", " + fmtPct(cell.ci[1]) + "]" : "");
        append(row, el("td", { class: "cell-rate", style: rateColor(cell.win_rate), title: title,
          text: fmtPct(cell.win_rate) }));
      });
      append(row, el("td", { text: entry.weighted_score !== null ? entry.weighted_score.toFixed(3) : "—" }));
      append(table, row);
    });
    append(body, el("div", { class: "ladder-wrap" }, table));

    if (ladder.history.length > 1) {
      var chartHolder = el("div", { style: "margin-top:14px;" });
      append(body, chartHolder);
      lineChart(chartHolder, function () {
        var pal = palette();
        var xs = ladder.history.map(function (p, i) { return p.generation !== null ? p.generation : i; });
        return {
          series: [{ name: "weighted ladder score", x: xs,
            y: ladder.history.map(function (p) { return p.weighted_score; }),
            color: pal.accent, dots: true }],
          height: 190, xInt: true, xTitle: "gen", xLabel: "generation (evaluation order)",
          yLabel: "weighted score",
        };
      });
    }

    var sub = ladder.from_mini_ladder
      ? "Selection instrument: keep-best + drift circuit-breaker recomputed from MiniLadder/history.json."
      : "Result JSONs from PentobiLadder/. No mini-ladder selection history was recorded for this run — " +
        "keep-best below is derived from these results alone.";
    append(body, el("p", { class: "card-sub", style: "margin-top:10px;", text: sub }));
    return card("Pentobi ladder — the run's verdict", "Win rate vs the Pentobi engine per difficulty level. Cells show win rate (hover for W–L–D and CI); green = winning record.", body);
  }

  function buildExternalSection() {
    var s = section("external", "External evidence", "external", "cannot be gamed by the loop",
      "Signals anchored outside the self-play loop: the Pentobi ladder, the pooled BayesElo tournament " +
      "(an independent code path over all checkpoints), and the game's ground-truth invariances. " +
      "In the one run that regressed 44 Elo, these were the only honest warnings — and they fired from generation 5.");

    append(s, buildLadderCard());

    var grid = el("div", { class: "card-grid" });

    if (DATA.tournament) {
      append(grid, chartCard("Pool Elo (BayesElo tournament)",
        "Every checkpoint rated on one shared scale by a pooled round-robin — the rigorous strength curve. Gen 0 is the anchor.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "pool Elo", x: DATA.tournament.gens, y: DATA.tournament.rating,
                color: pal.accent, dots: true }],
              height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "Elo vs gen-0 anchor",
              hlines: [{ y: 0, label: "gen-0 anchor" }], includeZero: true,
            };
          });
        }));
    } else {
      append(grid, placeholder("Pool Elo (BayesElo)",
        "No Tournament/tournament_ratings.parquet — enable tournament.run_at_end or run scripts/tournament_elo.py."));
    }

    if (DATA.symmetry) {
      append(grid, chartCard("Policy symmetry KL",
        "KL divergence between the policy on a position and on its symmetric transforms — a ground-truth invariance. " +
        "A healthy net holds flat; a monotonic rise is the drift signature that preceded the L4→L3 regression.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [
                { name: "mean over positions", x: DATA.symmetry.gens, y: DATA.symmetry.kl_mean, color: pal.accent, dots: true },
                { name: "worst position", x: DATA.symmetry.gens, y: DATA.symmetry.kl_max, color: pal.violet, dash: "4 3", width: 1.3 },
              ],
              height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "KL (nats)",
            };
          });
        }));
    } else {
      append(grid, placeholder("Policy symmetry KL", "SymmetryDiagnostic table absent for this run."));
    }

    if (DATA.pvc && DATA.pvc.value_symmetry_mae) {
      append(grid, chartCard("Value symmetry MAE",
        "Mean |V(s) − V(sym(s))| over symmetric transforms — the value head's equivariance error. " +
        "Rose 0.10 → 0.25 during the regression while value loss looked better than ever.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "value symmetry MAE", x: DATA.pvc.gens, y: DATA.pvc.value_symmetry_mae,
                color: pal.accent, dots: true }],
              height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "MAE",
            };
          });
        }));
    } else {
      append(grid, placeholder("Value symmetry MAE",
        "PolicyValueConsistency table absent (or predates the value-symmetry column)."));
    }

    if (DATA.target_entropy) {
      append(grid, chartCard("Self-play target entropy",
        "Entropy of the stored search-policy targets at harvest (band = p10–p90 across episodes). " +
        "A sharp collapse — gen 17 of the rerun fell 0.79 → 0.51 nats — precedes the trainer chasing degenerate targets.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "mean target entropy", x: DATA.target_entropy.gens, y: DATA.target_entropy.mean,
                color: pal.accent, dots: true,
                band: { lo: DATA.target_entropy.p10, hi: DATA.target_entropy.p90 } }],
              height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "entropy (nats)",
            };
          });
        }));
    } else {
      append(grid, placeholder("Self-play target entropy", "SelfPlayProfiling table absent for this run."));
    }

    append(s, grid);
    return s;
  }

  // ------------------------------------------------------------------
  // Arena instrument section
  // ------------------------------------------------------------------

  function buildInstrumentSection() {
    var s = section("instrument", "Arena instrument", "instrument", "measurement health",
      "The candidate-vs-incumbent arena decides (or telemeters) weight flow, but in Blokus Duo ~93–97% of decisive " +
      "deterministic games are won by the first mover, so between near-equal nets the score is structurally pinned " +
      "near 0.500. These checks ask whether the gate measured anything at all.");

    var arena = DATA.arena;
    if (!arena) {
      append(s, placeholder("Arena data", "ArenaData table absent for this run."));
      return s;
    }

    if (arena.red_flags.length) {
      var list = el("ul");
      arena.red_flags.forEach(function (flag) { append(list, el("li", { text: flag })); });
      append(s, el("div", { class: "flag-banner" },
        el("strong", { text: "⚠ Instrument red flags" }), list));
    } else {
      append(s, el("div", { class: "ok-banner",
        text: "No pinning signature detected: no exact-0.500 scores, score variance consistent with independent games" +
          (arena.white_rate !== null && arena.white_rate !== undefined
            ? ", white won " + fmtPct(arena.white_rate) + " of decisive games." : ".") }));
    }

    var grid = el("div", { class: "card-grid" });

    var gateNote = arena.gate_mode === "threshold"
      ? "Gate: accept at score ≥ " + arena.threshold + "."
      : arena.gate_mode === "regression_guard"
        ? "Gate: regression guard (reject only clear losses) — the dashed line is the nominal threshold, kept for scale."
        : "Gate: always accept — the arena is telemetry only.";
    append(grid, chartCard("Arena score per generation",
      "score = (wins + ½·draws) / games, candidate vs incumbent. Filled points = accepted. " + gateNote,
      function (body) {
        lineChart(body, function () {
          var pal = palette();
          return {
            series: [{ name: "arena score", x: arena.gens, y: arena.score, color: pal.accent, dots: arena.accepted }],
            height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "score",
            hlines: [
              { y: 0.5, label: "0.500", color: pal.gray, dash: "2 3" },
              { y: arena.threshold, label: "threshold " + arena.threshold, color: pal.warn },
            ],
          };
        });
      }));

    if (arena.white_wins) {
      append(grid, chartCard("Decisive games by colour",
        "Who actually won arena games: the first mover (White) or the second (Black). When White takes nearly every " +
        "decisive game, the gate is measuring colour, not strength — this chart is why the arena gate is nearly information-free.",
        function (body) {
          barChart(body, function () {
            var pal = palette();
            return {
              x: arena.gens,
              stacks: [
                { name: "White wins", y: arena.white_wins, color: pal.whiteP },
                { name: "Black wins", y: arena.black_wins, color: pal.blackP },
                { name: "Draws", y: arena.draws, color: pal.gray },
              ],
              height: 230, xTitle: "gen", yLabel: "games",
            };
          });
        }));
    } else {
      append(grid, placeholder("Decisive games by colour",
        "This run predates per-colour arena logging (white_wins/black_wins) — the single groupby that would have " +
        "caught colour pinning three runs earlier."));
    }

    if (DATA.rolling_elo) {
      append(grid, chartCard("Rolling arena-derived Elo",
        "Chained estimate from the same arena games (candidate rated vs current incumbent; benchmark rolls forward on " +
        "acceptance). Self-referential — read the trend, trust the pool tournament for the rating.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "rolling Elo", x: DATA.rolling_elo.gens, y: DATA.rolling_elo.elo,
                color: pal.violet, dots: DATA.rolling_elo.accepted || true }],
              height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "Elo (chained)",
            };
          });
        }));
    }

    if (DATA.legacy_elo) {
      append(grid, chartCard("Elo vs frozen gen-0 (retired metric)",
        "Kept for older runs only: rated against a frozen random-init baseline, so it saturates once the net always wins. " +
        "Superseded by the rolling Elo + pooled tournament.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "Elo vs gen-0", x: DATA.legacy_elo.gens, y: DATA.legacy_elo.elo,
                color: pal.gray, dots: true }],
              height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "Elo",
            };
          });
        }));
    }

    append(s, grid);
    return s;
  }

  // ------------------------------------------------------------------
  // Self-referential training telemetry
  // ------------------------------------------------------------------

  function buildInternalSection() {
    var s = section("internal", "Training telemetry", "internal", "self-referential",
      "");
    append(s, el("div", { class: "self-referential-note" },
      el("span", { text: "⚠" }),
      el("span", null,
        "Everything in this section is measured against the loop's own outputs — its buffer, its eval set, its own " +
        "search targets. During the 20-generation run that lost 44 Elo, loss fell, acceptance hit 100% and eval top-1 " +
        "read 0.99. These curves diagnose ", el("em", { text: "how" }), " training moved, not ",
        el("em", { text: "whether" }), " the run improved.")));

    var grid = el("div", { class: "card-grid" });

    if (DATA.training) {
      append(grid, chartCard("Loss per generation",
        "Mean loss over each generation's final epoch. In a gated loop falling loss tracks progress; with the gate " +
        "open it can simply mean the loop made its own data easier to predict.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            var series = [
              { name: "total", x: DATA.training.gens, y: DATA.training.total, color: pal.accent, width: 2.2, dots: true },
              { name: "policy (π)", x: DATA.training.gens, y: DATA.training.pi, color: pal.violet },
              { name: "value (v)", x: DATA.training.gens, y: DATA.training.v, color: pal.ok },
            ];
            var auxColors = [pal.warn, pal.alert, pal.gray];
            Object.keys(DATA.training.aux).forEach(function (name, i) {
              series.push({ name: name.replace("_", " ") + " (aux)", x: DATA.training.gens,
                y: DATA.training.aux[name], color: auxColors[i % auxColors.length], dash: "4 3", width: 1.3 });
            });
            return { series: series, height: 240, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "loss" };
          });
        }));

      append(grid, chartCard("Per-batch loss timeline",
        "EWM-smoothed raw batches across the whole run; dashed marks are generation boundaries (where fresh self-play " +
        "data entered the buffer).",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            var t = DATA.training.timeline;
            var every = Math.ceil(t.gen_starts.length / 8);
            return {
              series: [
                { name: "total", x: t.x, y: t.total, color: pal.accent, width: 1.6 },
                { name: "policy (π)", x: t.x, y: t.pi, color: pal.violet, width: 1.3 },
                { name: "value (v)", x: t.x, y: t.v, color: pal.ok, width: 1.3 },
              ],
              height: 240, xTitle: "batch", xLabel: "training batch (all generations)", yLabel: "loss",
              vmarks: t.gen_starts.filter(function (_, i) { return i % every === 0 && i > 0; })
                .map(function (gs) { return { x: gs[1], label: "g" + gs[0] }; }),
            };
          });
        }));
    } else {
      append(grid, placeholder("Training loss", "TrainingData table absent in this run directory (older runs synced without it)."));
    }

    if (DATA.accuracy) {
      append(grid, chartCard("Eval-set policy agreement",
        "Net's top-1/top-5 agreement with a frozen eval set sampled from generation-1 self-play. The targets are the " +
        "same lineage's search output — near-1.0 readings certify fit, not strength.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            var series = [
              { name: "top-1", x: DATA.accuracy.gens, y: DATA.accuracy.top1, color: pal.accent, dots: true },
              { name: "top-5", x: DATA.accuracy.gens, y: DATA.accuracy.top5, color: pal.violet },
            ];
            if (DATA.accuracy.mcts_top1) {
              series.push({ name: "top-1 (vs MCTS)", x: DATA.accuracy.gens, y: DATA.accuracy.mcts_top1,
                color: pal.accent, dash: "4 3", width: 1.2 });
            }
            if (DATA.accuracy.mcts_top5) {
              series.push({ name: "top-5 (vs MCTS)", x: DATA.accuracy.gens, y: DATA.accuracy.mcts_top5,
                color: pal.violet, dash: "4 3", width: 1.2 });
            }
            return { series: series, height: 240, xInt: true, xTitle: "gen", xLabel: "generation",
              yLabel: "agreement", yFmt: fmtPct };
          });
        }));
    }

    if (DATA.pvc) {
      append(grid, chartCard("Policy–value consistency",
        "Does the policy agree with a one-ply value lookahead (Q₁(a) = −V(child))? A trend, not a target: a healthy " +
        "net rises then plateaus below 100% (the policy sees deeper than one ply). Watch for late drops.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [
                { name: "Spearman ρ", x: DATA.pvc.gens, y: DATA.pvc.spearman, color: pal.accent, dots: true },
                { name: "argmax match", x: DATA.pvc.gens, y: DATA.pvc.argmax_match, color: pal.violet },
              ],
              height: 240, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "consistency",
            };
          });
        }));
    }

    if (DATA.calibration) {
      append(grid, chartCard("Value-head calibration",
        "Reliability at generation " + DATA.calibration.reliability.generation +
        ": mean actual outcome per predicted-value bucket; the diagonal is perfect calibration. " +
        "Note: 73% of self-play outcomes are White wins, and this view is colour-blind (plateau R8a).",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            var r = DATA.calibration.reliability;
            return {
              series: [{ name: "actual outcome", x: r.centers, y: r.actual, color: pal.accent, dots: true }],
              height: 240, diag: true, xTitle: "predicted", xLabel: "predicted value", yLabel: "mean actual outcome",
            };
          });
        }));

      append(grid, chartCard("Calibration error per generation",
        "Count-weighted |predicted − actual| across value buckets.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "calibration error", x: DATA.calibration.gens, y: DATA.calibration.error,
                color: pal.violet, dots: true }],
              height: 240, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "error",
            };
          });
        }));
    }

    if (DATA.net_entropy) {
      append(grid, chartCard("Network policy entropy",
        "Entropy of the raw network policy on the frozen eval set (no search). Falling entropy = sharpening priors; " +
        "a cliff means the policy is collapsing onto few moves.",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "net policy entropy", x: DATA.net_entropy.gens, y: DATA.net_entropy.mean,
                color: pal.accent, dots: true }],
              height: 240, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "entropy (nats)",
            };
          });
        }));
    }

    if (DATA.lr) {
      append(grid, chartCard("Learning rate",
        "The LR the optimiser actually trained at each generation — the ground truth for schedule comparisons " +
        "(two of three past runs ran a different schedule than their committed config).",
        function (body) {
          lineChart(body, function () {
            var pal = palette();
            return {
              series: [{ name: "learning rate", x: DATA.lr.gens, y: DATA.lr.lr, color: pal.gray, dots: true }],
              height: 240, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "LR", includeZero: true,
            };
          });
        }));
    }

    append(s, grid);
    return s;
  }

  // ------------------------------------------------------------------
  // Replay browser
  // ------------------------------------------------------------------

  function buildReplaySection() {
    var s = section("replays", "Arena replays", "instrument", "game browser",
      "Step through recorded arena games move by move. Player 1 is the incumbent, player 2 the freshly-trained " +
      "candidate; colours are per game. Select an alternative to see the move MCTS considered instead of the one it played.");
    if (!DATA.replays) {
      append(s, placeholder("Arena replays", "No ArenaReplays partitions in this run directory."));
      return s;
    }
    var app = el("div", { class: "replay-app" });
    append(s, app);
    new ReplayBrowser(app, DATA.replays);
    return s;
  }

  function ReplayBrowser(root, replays) {
    this.replays = replays;
    this.gens = Object.keys(replays.gens).map(Number).sort(function (a, b) { return a - b; });
    this.state = { gen: this.gens[0], game: 0, move: 0, alt: null };

    this.side = el("div", { class: "replay-side" });
    this.main = el("div", { class: "replay-main card" });
    append(root, [this.side, this.main]);

    var self = this;
    document.addEventListener("keydown", function (event) {
      if (event.target.tagName === "INPUT" || event.target.tagName === "SELECT") return;
      var rect = root.getBoundingClientRect();
      if (rect.bottom < 0 || rect.top > window.innerHeight) return;
      if (event.key === "ArrowLeft") { self.step(-1); event.preventDefault(); }
      if (event.key === "ArrowRight") { self.step(1); event.preventDefault(); }
    });

    this.renderSide();
    this.renderMain();
  }

  ReplayBrowser.prototype.currentGame = function () {
    return this.replays.gens[String(this.state.gen)][this.state.game];
  };

  ReplayBrowser.prototype.step = function (delta) {
    var moves = this.currentGame().moves.length;
    var next = Math.max(0, Math.min(moves, this.state.move + delta));
    if (next === this.state.move) return;
    this.state.move = next;
    this.state.alt = null;
    this.renderMain();
  };

  ReplayBrowser.prototype.selectGame = function (gen, gameIdx) {
    this.state.gen = gen;
    this.state.game = gameIdx;
    this.state.move = 0;
    this.state.alt = null;
    this.renderSide();
    this.renderMain();
  };

  ReplayBrowser.prototype.renderSide = function () {
    var self = this;
    this.side.innerHTML = "";
    var cardNode = el("div", { class: "card" });

    var genRow = el("div", { class: "replay-gens" });
    this.gens.forEach(function (gen) {
      append(genRow, el("button", {
        class: gen === self.state.gen ? "active" : "",
        text: "Gen " + gen,
        onclick: function () { self.selectGame(gen, 0); },
      }));
    });
    append(cardNode, genRow);

    var games = el("div", { class: "replay-games" });
    this.replays.gens[String(this.state.gen)].forEach(function (game, i) {
      append(games, el("button", {
        class: i === self.state.game ? "active" : "",
        onclick: function () { self.selectGame(self.state.gen, i); },
      },
        el("span", { class: "g-dot " + game.winner }),
        "G" + (game.idx + 1) + " — " + game.label));
    });
    append(cardNode, games);
    append(this.side, cardNode);
  };

  ReplayBrowser.prototype.boardSvg = function () {
    var replays = this.replays;
    var game = this.currentGame();
    var k = this.state.move;
    var cell = 26, pad = 3;
    var W = replays.cols * cell + pad * 2, H = replays.rows * cell + pad * 2;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H });

    // Base grid.
    for (var r = 0; r < replays.rows; r++) {
      for (var c = 0; c < replays.cols; c++) {
        svg.appendChild(svgEl("rect", { class: "board-cell", x: pad + c * cell, y: pad + r * cell,
          width: cell, height: cell, rx: 2 }));
      }
    }

    var isTTT = replays.game === "tictactoe";
    var altActive = this.state.alt !== null && k < game.moves.length;
    var paintUpTo = altActive ? k : k; // moves [0, k) are placed; alt previews replace move k
    for (var i = 0; i < paintUpTo; i++) {
      var move = game.moves[i];
      var side = move.p === 1 ? "white" : "black";
      var isLast = !altActive && i === paintUpTo - 1;
      for (var j = 0; j < move.cells.length; j++) {
        var cc = move.cells[j];
        svg.appendChild(svgEl("rect", { class: "board-cell " + side + (isLast ? " last" : ""),
          x: pad + cc[1] * cell, y: pad + cc[0] * cell, width: cell, height: cell, rx: 2 }));
        var label = isTTT ? (move.p === 1 ? "X" : "O") : String(cc[2]);
        svg.appendChild(svgEl("text", { class: "board-cell-label",
          x: pad + cc[1] * cell + cell / 2, y: pad + cc[0] * cell + cell / 2 + 2.4,
          style: "font-size:" + (isTTT ? 13 : 8.5) + "px", text: label }));
      }
    }

    if (altActive) {
      var alt = game.moves[k].alts[this.state.alt];
      var altSide = game.moves[k].p === 1 ? "white" : "black";
      alt.cells.forEach(function (ac) {
        svg.appendChild(svgEl("rect", { class: "board-ghost-fill " + altSide,
          x: pad + ac[1] * cell, y: pad + ac[0] * cell, width: cell, height: cell, rx: 2 }));
        svg.appendChild(svgEl("rect", { class: "board-ghost " + altSide,
          x: pad + ac[1] * cell + 1, y: pad + ac[0] * cell + 1, width: cell - 2, height: cell - 2, rx: 2 }));
      });
    }
    return svg;
  };

  ReplayBrowser.prototype.renderMain = function () {
    var self = this;
    var game = this.currentGame();
    var k = this.state.move;
    var total = game.moves.length;
    this.main.innerHTML = "";

    var boardHolder = el("div", { class: "board-svg-holder" });
    boardHolder.appendChild(this.boardSvg());

    var info = el("div", { class: "replay-info" });

    var controls = el("div", { class: "replay-controls" },
      el("button", { text: "⏮", disabled: k === 0 ? "disabled" : null, onclick: function () { self.state.move = 0; self.state.alt = null; self.renderMain(); } }),
      el("button", { text: "◀", disabled: k === 0 ? "disabled" : null, onclick: function () { self.step(-1); } }),
      el("button", { text: "▶", disabled: k === total ? "disabled" : null, onclick: function () { self.step(1); } }),
      el("button", { text: "⏭", disabled: k === total ? "disabled" : null, onclick: function () { self.state.move = total; self.state.alt = null; self.renderMain(); } }),
      el("span", { class: "move-counter", text: "move " + k + " / " + total }));
    append(info, controls);

    var scrub = el("input", { class: "replay-scrub", type: "range", min: 0, max: total, value: k });
    scrub.addEventListener("input", function () {
      self.state.move = parseInt(scrub.value, 10);
      self.state.alt = null;
      self.renderMain();
    });
    append(info, scrub);

    if (k > 0) {
      var last = game.moves[k - 1];
      var lastSide = last.p === 1 ? "white" : "black";
      append(info, el("div", { class: "move-caption" },
        el("span", { class: "mover " + lastSide, text: (last.p === 1 ? "White" : "Black") }),
        " played " + last.cap));
      if (last.prob !== null) {
        append(info, el("div", { class: "move-prob" },
          fmtPct(last.prob) + " of MCTS visits",
          el("div", { class: "prob-bar" }, el("div", { style: "width:" + Math.min(100, last.prob * 100) + "%" }))));
      }
    } else {
      append(info, el("div", { class: "move-caption", text: "Start of game — use ▶ or the arrow keys to step through." }));
    }

    if (k < total) {
      var next = game.moves[k];
      var nextSide = next.p === 1 ? "white" : "black";
      append(info, el("div", { class: "alts-title", text: "Next: " + (next.p === 1 ? "White" : "Black") + " to move" }));
      var chips = el("div", { class: "alt-chips" });
      append(chips, el("button", {
        class: this.state.alt === null ? "active" : "",
        onclick: function () { self.state.alt = null; self.renderMain(); },
      }, "Played: " + next.cap, next.prob !== null ? el("span", { class: "alt-prob", text: fmtPct(next.prob) }) : null));
      next.alts.forEach(function (alt, i) {
        append(chips, el("button", {
          class: self.state.alt === i ? "active" : "",
          onclick: function () { self.state.alt = self.state.alt === i ? null : i; self.renderMain(); },
        }, "Alt " + (i + 1) + ": " + alt.cap, el("span", { class: "alt-prob", text: fmtPct(alt.prob) })));
      });
      append(info, chips);
      if (next.alts.length) {
        append(info, el("div", { class: "alt-hint",
          text: "Selecting an alternative previews it (striped) on the pre-move board — press ▶ to see what was actually played." }));
      }
      if (this.state.alt !== null) {
        // When previewing an alt the played move is not yet on the board.
        boardHolder.innerHTML = "";
        boardHolder.appendChild(this.boardSvg());
      }
    }

    if (k === total) {
      append(info, el("div", { class: "replay-result " + game.winner, text: game.label }));
    }

    var whiteRole = game.p1_white ? "previous net" : "new candidate";
    var blackRole = game.p1_white ? "new candidate" : "previous net";
    append(info, el("div", { class: "replay-legend" },
      el("span", { class: "k" }, el("span", { class: "sq", style: "background:var(--white-player)" }), "White — " + whiteRole),
      el("span", { class: "k" }, el("span", { class: "sq", style: "background:var(--black-player)" }), "Black — " + blackRole)));

    var wrap = el("div", { class: "board-wrap" }, boardHolder, info);
    append(this.main, wrap);
  };

  // ------------------------------------------------------------------
  // Operational section
  // ------------------------------------------------------------------

  function buildOpsSection() {
    var s = section("ops", "Operations", "internal", "ops",
      "Wall-clock, throughput and memory — where the generation time goes.");
    var grid = el("div", { class: "card-grid" });
    var perf = DATA.perf || {};

    if (perf.timing) {
      var totalNote = perf.total_time_s
        ? "Total run time " + (perf.total_time_s / 3600).toFixed(1) + " h."
        : "";
      append(grid, chartCard("Time per generation", "Wall-clock per phase. " + totalNote, function (body) {
        barChart(body, function () {
          var pal = palette();
          var colors = { SelfPlay: pal.accent, Training: pal.violet, Arena: pal.ok };
          var extraColors = [pal.warn, pal.gray, pal.alert];
          var extraIdx = 0;
          return {
            x: perf.timing.gens,
            stacks: Object.keys(perf.timing.stages).map(function (stage) {
              var color = colors[stage] || extraColors[extraIdx++ % extraColors.length];
              return { name: stage, y: perf.timing.stages[stage], color: color };
            }),
            height: 230, xTitle: "gen", yLabel: "seconds",
          };
        });
      }));
    }

    if (perf.throughput) {
      append(grid, chartCard("Training throughput", "Mean samples/second per generation.", function (body) {
        lineChart(body, function () {
          var pal = palette();
          return {
            series: [{ name: "samples/s", x: perf.throughput.gens, y: perf.throughput.sps, color: pal.accent, dots: true }],
            height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "samples/s", includeZero: true,
          };
        });
      }));
    }

    if (DATA.selfplay) {
      append(grid, chartCard("Game length", "Moves per self-play game (band = p10–p90 across episodes).", function (body) {
        lineChart(body, function () {
          var pal = palette();
          return {
            series: [{ name: "mean moves/game", x: DATA.selfplay.gens, y: DATA.selfplay.moves_mean,
              color: pal.accent, dots: true, band: { lo: DATA.selfplay.moves_p10, hi: DATA.selfplay.moves_p90 } }],
            height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "moves",
          };
        });
      }));

      append(grid, chartCard("Search throughput", "Median MCTS simulations/second across episodes.", function (body) {
        lineChart(body, function () {
          var pal = palette();
          return {
            series: [{ name: "sims/s (median)", x: DATA.selfplay.gens, y: DATA.selfplay.sims_median,
              color: pal.violet, dots: true }],
            height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "sims/s", includeZero: true,
          };
        });
      }));
    }

    if (perf.memory) {
      append(grid, chartCard("Peak process memory", "Max RSS per phase per generation.", function (body) {
        lineChart(body, function () {
          var pal = palette();
          var colors = { SelfPlay: pal.accent, Training: pal.violet, Arena: pal.ok };
          var extraColors = [pal.warn, pal.gray, pal.alert];
          var extraIdx = 0;
          return {
            series: Object.keys(perf.memory.stages).map(function (stage) {
              return { name: stage, x: perf.memory.gens, y: perf.memory.stages[stage],
                color: colors[stage] || extraColors[extraIdx++ % extraColors.length] };
            }),
            height: 230, xInt: true, xTitle: "gen", xLabel: "generation", yLabel: "GB", includeZero: true,
          };
        });
      }));
    }

    if (!grid.children.length) {
      append(s, placeholder("Operational telemetry", "No Timings / TrainingThroughput / ResourceUsage tables."));
    } else {
      append(s, grid);
    }
    return s;
  }

  // ------------------------------------------------------------------
  // Config + footer
  // ------------------------------------------------------------------

  function buildConfig() {
    var details = el("details", { class: "config", id: "config" },
      el("summary", { text: "Run configuration" }));
    var table = el("table", { class: "config-table" });
    DATA.meta.config_rows.forEach(function (row) {
      append(table, el("tr", null, el("td", { text: row[0] }), el("td", { class: "mono", text: row[1] })));
    });
    append(details, table);
    if (DATA.meta.missing_tables.length) {
      append(details, el("div", { class: "missing-tables",
        text: "Metric tables not present in this run directory: " + DATA.meta.missing_tables.join(", ") + "." }));
    }
    return details;
  }

  function buildFooter() {
    return el("footer", { class: "report-footer" },
      "AlphaBlokus run report · " + DATA.meta.run_name + " · generated " + DATA.meta.date +
      " · self-contained (works offline) · charts hand-rolled SVG, no dependencies");
  }

  // ------------------------------------------------------------------
  // Assemble
  // ------------------------------------------------------------------

  function build() {
    var navSections = [
      { id: "external", label: "External evidence" },
      { id: "instrument", label: "Arena instrument" },
      { id: "internal", label: "Training telemetry" },
      { id: "replays", label: "Replays" },
      { id: "ops", label: "Operations" },
      { id: "config", label: "Config" },
    ];

    var body = document.body;
    body.appendChild(buildTopbar(navSections));
    var page = el("div", { class: "page" });
    body.appendChild(page);

    append(page, buildHeader());
    append(page, buildVerdict());
    append(page, buildSignals());
    append(page, buildExternalSection());
    append(page, buildInstrumentSection());
    append(page, buildInternalSection());
    append(page, buildReplaySection());
    append(page, buildOpsSection());
    append(page, buildConfig());
    append(page, buildFooter());
  }

  initTheme();
  build();
})();
