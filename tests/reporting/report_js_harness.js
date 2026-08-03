/* Drive the report's shipped JavaScript outside a browser.
 *
 * The previous replay browser crashed the moment it painted a placed move
 * (an undeclared variable, fatal under "use strict"), and a render-only check
 * never noticed: the initial position paints nothing. So this harness does not
 * check that the page renders — it *uses* it. It builds the page from a real
 * report.html, then plays a game to the end, steps, scrubs, selects
 * alternatives, switches generations and games, and toggles the theme, failing
 * on the first error any of that raises.
 *
 * The DOM shim below implements only what assets/report.js touches. It is
 * deliberately dumb: no layout, no CSS cascade, no SVG semantics — just enough
 * object graph for the report's own code paths to execute for real.
 *
 * Usage: node report_js_harness.js <report.html>   (prints JSON, exits non-zero
 * on any error, with every collected error on stderr.)
 */
"use strict";

const fs = require("fs");

const errors = [];

// ---------------------------------------------------------------------------
// Minimal DOM
// ---------------------------------------------------------------------------

class ClassList {
  constructor(node) { this.node = node; }
  get _names() { return String(this.node.attributes["class"] || "").split(/\s+/).filter(Boolean); }
  contains(name) { return this._names.indexOf(name) >= 0; }
}

class Node {
  constructor(tagName, namespace) {
    this.tagName = String(tagName).toUpperCase();
    this.localName = String(tagName);
    this.namespace = namespace || null;
    this.children = [];
    this.parentNode = null;
    this.attributes = {};
    this.listeners = {};
    this.style = {};
    this._text = "";
    this.value = "";
    this.disabled = false;
    this.open = false;
    // Fixed, non-zero geometry: charts divide by clientWidth.
    this.clientWidth = 720;
    this.clientHeight = 240;
  }

  get classList() { return new ClassList(this); }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
    if (name === "disabled") this.disabled = true;
    if (name === "value") this.value = String(value);
  }

  getAttribute(name) {
    return Object.prototype.hasOwnProperty.call(this.attributes, name) ? this.attributes[name] : null;
  }

  removeAttribute(name) { delete this.attributes[name]; }

  appendChild(child) {
    if (child.parentNode) {
      const siblings = child.parentNode.children;
      const at = siblings.indexOf(child);
      if (at >= 0) siblings.splice(at, 1);
    }
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  addEventListener(type, fn) {
    (this.listeners[type] = this.listeners[type] || []).push(fn);
  }

  removeEventListener(type, fn) {
    const list = this.listeners[type] || [];
    const at = list.indexOf(fn);
    if (at >= 0) list.splice(at, 1);
  }

  dispatch(type, event) {
    const target = Object.assign({ target: this, preventDefault() {}, stopPropagation() {} }, event || {});
    (this.listeners[type] || []).slice().forEach((fn) => fn.call(this, target));
  }

  set textContent(value) { this._text = String(value); this.children = []; }

  get textContent() {
    if (this.children.length === 0) return this._text;
    return this._text + this.children.map((c) => c.textContent).join("");
  }

  set innerHTML(value) {
    // The report only ever assigns "" (to clear) or small trusted fragments;
    // treating any assignment as "replace my contents with this text" is
    // enough, and keeps the shim from pretending to be an HTML parser.
    this.children = [];
    this._text = String(value);
  }

  get innerHTML() { return this.textContent; }

  getBoundingClientRect() {
    return { x: 0, y: 0, top: 0, left: 0, bottom: 400, right: 720, width: 720, height: 400 };
  }

  scrollIntoView() {}

  // Depth-first walk used by the query helpers below.
  * walk() {
    for (const child of this.children) {
      yield child;
      yield* child.walk();
    }
  }

  // Supports the shapes actually queried here: "#id", ".class", "tag",
  // "tag.class" and space-separated descendant combinations of those.
  matchesSimple(selector) {
    const parts = selector.match(/^([a-zA-Z][\w-]*)?((?:[.#][\w-]+)*)$/);
    if (!parts) return false;
    if (parts[1] && this.localName !== parts[1]) return false;
    const tokens = parts[2].match(/[.#][\w-]+/g) || [];
    return tokens.every((token) => (token[0] === "#"
      ? this.getAttribute("id") === token.slice(1)
      : this.classList.contains(token.slice(1))));
  }

  matches(selector) {
    const parts = String(selector).trim().split(/\s+/);
    const own = parts.pop();
    if (!this.matchesSimple(own)) return false;
    let node = this.parentNode;
    while (parts.length && node) {
      if (node.matchesSimple(parts[parts.length - 1])) parts.pop();
      node = node.parentNode;
    }
    return parts.length === 0;
  }

  querySelector(selector) {
    for (const node of this.walk()) if (node.matches(selector)) return node;
    return null;
  }

  querySelectorAll(selector) {
    const out = [];
    for (const node of this.walk()) if (node.matches(selector)) out.push(node);
    return out;
  }
}

function makeDocument() {
  const documentElement = new Node("html");
  const body = new Node("body");
  documentElement.appendChild(body);
  const doc = {
    documentElement: documentElement,
    body: body,
    createElement: (tag) => new Node(tag),
    createElementNS: (ns, tag) => new Node(tag, ns),
    createTextNode: (text) => {
      const node = new Node("#text");
      node._text = String(text);
      return node;
    },
    getElementById: (id) => documentElement.querySelector("#" + id),
    querySelector: (selector) => (documentElement.matches(selector)
      ? documentElement
      : documentElement.querySelector(selector)),
    querySelectorAll: (selector) => documentElement.querySelectorAll(selector),
    addEventListener: (type, fn) => documentElement.addEventListener(type, fn),
    removeEventListener: (type, fn) => documentElement.removeEventListener(type, fn),
  };
  return doc;
}

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

function extractPayloadAndScript(html) {
  const payload = html.match(
    /<script id="report-data" type="application\/json">\n([\s\S]*?)\n<\/script>/
  );
  if (!payload) throw new Error("report.html has no #report-data payload");
  // The writer escapes "</" inside the JSON so it cannot close the tag early;
  // undo that here exactly as a browser's JSON.parse of textContent would see it.
  const scripts = html.match(/<script>\n([\s\S]*?)\n<\/script>/);
  if (!scripts) throw new Error("report.html has no inline report script");
  return { payloadText: payload[1], source: scripts[1] };
}

function run(reportPath) {
  const html = fs.readFileSync(reportPath, "utf8");
  const { payloadText, source } = extractPayloadAndScript(html);

  const document = makeDocument();
  const dataScript = new Node("script");
  dataScript.setAttribute("id", "report-data");
  dataScript.textContent = payloadText;
  document.body.appendChild(dataScript);

  // Timers are modelled properly — cleared ones must stop firing — so that
  // "playback stopped on its own" means the same thing here as in a browser.
  let timerSeq = 0;
  const liveTimers = new Map();
  const pumpOnce = () => {
    const ids = Array.from(liveTimers.keys());
    if (!ids.length) return false;
    liveTimers.get(ids[ids.length - 1])();
    return true;
  };
  const sandbox = {
    document: document,
    console: console,
    JSON: JSON,
    Math: Math,
    Object: Object,
    Array: Array,
    String: String,
    Number: Number,
    isNaN: isNaN,
    parseInt: parseInt,
    parseFloat: parseFloat,
    setTimeout: (fn) => { liveTimers.set(++timerSeq, fn); return timerSeq; },
    clearTimeout: (id) => { liveTimers.delete(id); },
    setInterval: (fn) => { liveTimers.set(++timerSeq, fn); return timerSeq; },
    clearInterval: (id) => { liveTimers.delete(id); },
    getComputedStyle: () => ({ getPropertyValue: () => "#2a78d6" }),
    localStorage: {
      store: {},
      getItem(key) { return Object.prototype.hasOwnProperty.call(this.store, key) ? this.store[key] : null; },
      setItem(key, value) { this.store[key] = String(value); },
    },
    matchMedia: () => ({ matches: false }),
    innerWidth: 1280,
    innerHeight: 800,
    location: { hash: "" },
    devicePixelRatio: 1,
    addEventListener: () => {},
    removeEventListener: () => {},
  };
  sandbox.window = sandbox;
  sandbox.self = sandbox;

  const vm = require("vm");
  const context = vm.createContext(sandbox);
  vm.runInContext(source, context, { filename: "report.js" });

  const actions = [];
  const guard = (label, fn) => {
    try {
      fn();
      actions.push(label);
    } catch (err) {
      errors.push(label + ": " + (err && err.message ? err.message : String(err)));
    }
  };

  const controls = () => document.querySelectorAll(".replay-controls button");
  const byLabel = (needle) => controls().filter((b) => b.textContent.indexOf(needle) >= 0)[0];
  const counter = () => {
    const node = document.querySelector(".move-counter");
    return node ? node.textContent : "(no counter)";
  };
  const totalMoves = () => {
    const match = /move (\d+) \/ (\d+)/.exec(counter());
    return match ? Number(match[2]) : -1;
  };
  const currentMove = () => {
    const match = /move (\d+) \/ (\d+)/.exec(counter());
    return match ? Number(match[1]) : -1;
  };

  // 0. A run with no recorded replays has no browser to drive; the page still
  //    has to survive everything else, so carry on with `total` at 0. Callers
  //    that expect replays assert on `moves` in the returned summary.
  const total = Math.max(0, totalMoves());
  const hasReplays = total > 0;
  if (!hasReplays) actions.push("no replay browser (no replays recorded)");

  if (hasReplays) driveReplayBrowser(total, guard, byLabel, currentMove, document, pumpOnce);

  // Theme toggling re-renders every chart; do it in both directions.
  guard("theme to dark", () => document.querySelector(".theme-toggle").dispatch("click"));
  guard("theme to light", () => document.querySelector(".theme-toggle").dispatch("click"));

  // The key must exist and open, since every event flag links to it.
  const key = document.getElementById("key");
  if (!key) errors.push("no #key element: raised events would link nowhere");
  guard("open the key", () => document.querySelectorAll(".key-link").forEach((a) => a.dispatch("click")));
  if (key && !key.open) errors.push("#key did not open when a key link was clicked");

  return {
    actions: actions,
    moves: total,
    signals: document.querySelectorAll(".signal").length,
    charts: document.querySelectorAll(".chart").length,
    keyRows: document.querySelectorAll(".key-row").length,
    theme: document.documentElement.getAttribute("data-theme"),
  };
}

function driveReplayBrowser(total, guard, byLabel, currentMove, document, pumpOnce) {
  // 1. Press Play, then let the game run to the very end by pumping the
  //    interval callback the way a browser's timer would.
  guard("press play", () => byLabel("Play").dispatch("click"));
  const playing = () => Boolean(byLabel("Pause"));
  if (!playing()) errors.push("play did not switch the button to Pause");
  guard("play to the end", () => {
    for (let i = 0; i < total + 5; i++) {
      if (!pumpOnce()) break;
      if (currentMove() >= total) break;
    }
  });
  if (currentMove() !== total) {
    errors.push("playback stopped at " + currentMove() + " of " + total);
  }
  if (playing()) errors.push("playback did not stop at the final position");

  // 2. Step backwards and forwards with the buttons.
  guard("step back", () => byLabel("‹").dispatch("click"));
  if (currentMove() !== total - 1) errors.push("step back landed on " + currentMove());
  guard("step forward", () => byLabel("›").dispatch("click"));
  if (currentMove() !== total) errors.push("step forward landed on " + currentMove());

  // 3. Arrow keys.
  guard("arrow left", () => document.documentElement.dispatch("keydown", { key: "ArrowLeft" }));
  guard("arrow right", () => document.documentElement.dispatch("keydown", { key: "ArrowRight" }));

  // 4. Drag the scrubber across every position, including both ends. The same
  //    element must survive every update: a range input replaced mid-drag ends
  //    the drag after one jump, so identity here is the drag working.
  guard("scrub every position", () => {
    const scrub = document.querySelector(".replay-scrub");
    for (let move = 0; move <= total; move++) {
      scrub.value = String(move);
      scrub.dispatch("input");
      if (currentMove() !== move) throw new Error("scrub to " + move + " landed on " + currentMove());
      if (document.querySelector(".replay-scrub") !== scrub) {
        throw new Error("the scrubber was replaced during the drag (move " + move + ")");
      }
    }
  });

  // 5. Select each alternative offered at every position, then the played move.
  guard("select alternatives", () => {
    for (let move = 0; move < total; move++) {
      const scrub = document.querySelector(".replay-scrub");
      scrub.value = String(move);
      scrub.dispatch("input");
      const chips = document.querySelectorAll(".alt-chips button");
      chips.forEach((chip) => chip.dispatch("click"));
    }
  });

  // 6. Switch games and generations.
  guard("switch games", () => {
    document.querySelectorAll(".replay-games button").forEach((button) => {
      button.dispatch("click");
      byLabel("Play").dispatch("click");   // play from a fresh game
      pumpOnce();
      const pauseButton = byLabel("Pause");  // absent if the game already ended
      if (pauseButton) pauseButton.dispatch("click");
    });
  });
  guard("switch generations", () => {
    document.querySelectorAll(".replay-gens button").forEach((button) => button.dispatch("click"));
  });

  // 7. First / last position buttons.
  guard("jump to first", () => byLabel("⏮").dispatch("click"));
  guard("jump to last", () => byLabel("⏭").dispatch("click"));
}

const target = process.argv[2];
if (!target) {
  console.error("usage: node report_js_harness.js <report.html>");
  process.exit(2);
}

let result = null;
try {
  result = run(target);
} catch (err) {
  errors.push("fatal: " + (err && err.stack ? err.stack : String(err)));
}

if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log(JSON.stringify(result));
