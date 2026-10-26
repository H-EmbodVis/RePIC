(function () {
  "use strict";

  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- hero ledger: bars grow once on load ---- */
  var ledger = document.querySelector(".ledger");
  if (ledger) {
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { ledger.classList.add("in"); });
    });
  }

  /* ---- videos play while visible ---- */
  var videos = document.querySelectorAll("video[data-autoplay]");
  if (reduced || !("IntersectionObserver" in window)) {
    videos.forEach(function (v) { v.controls = true; });
  } else {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        var v = e.target;
        if (e.isIntersecting) {
          var p = v.play();
          if (p && p.catch) p.catch(function () { v.controls = true; });
        } else {
          v.pause();
        }
      });
    }, { threshold: 0.35 });
    videos.forEach(function (v) { io.observe(v); });
  }

  /* ---- BibTeX copy ---- */
  var copyBtn = document.getElementById("copy-bibtex");
  var bib = document.getElementById("bibtex");
  if (copyBtn && bib) {
    copyBtn.addEventListener("click", function () {
      var text = bib.textContent;
      var done = function () {
        copyBtn.textContent = "Copied";
        setTimeout(function () { copyBtn.textContent = "Copy"; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () { fallback(text, done); });
      } else {
        fallback(text, done);
      }
    });
  }
  function fallback(text, done) {
    var area = document.createElement("textarea");
    area.value = text;
    area.setAttribute("readonly", "");
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    try { document.execCommand("copy"); done(); } catch (err) { /* ignore */ }
    document.body.removeChild(area);
  }

  /* ---- noise-aligned transfer explorer (Eq. 6-7, synthetic curves) ---- */
  var root = document.getElementById("explorer");
  if (!root) return;

  var NS = "http://www.w3.org/2000/svg";
  var H = 32;        // action horizon
  var S = 10;        // integration steps
  var SHIFT = 5;     // sigma shift of the Fast-WAM action scheduler
  var W = 960, HT = 300, ML = 40, MR = 14, MT = 14, MB = 34;
  var SCALE = 50;    // px per unit of action value

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  var rand = mulberry32(11);
  function gauss() {
    var u = 1 - rand(), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  var epsPrev = [], epsNew = [];
  for (var i = 0; i < H; i++) { epsPrev.push(0.6 * gauss()); epsNew.push(0.6 * gauss()); }

  function clean(j) {
    var t = j / H;
    return 0.95 * Math.sin(2 * Math.PI * 1.1 * t + 0.4) + 0.35 * Math.sin(2 * Math.PI * 2.6 * t + 1.0);
  }
  function bend(j) { return 0.8 * Math.cos(2 * Math.PI * 0.8 * (j / H) + 0.2); }
  function sigma(s) {
    var u = 1 - s / S;
    return SHIFT * u / (1 + (SHIFT - 1) * u);
  }
  // cached solver state of the previous replan at noise level sg (clean endpoint, noisy endpoint,
  // plus a deviation from the straight line between them, as a real solver trajectory has)
  function cached(j, sg) {
    return (1 - sg) * clean(j) + sg * epsPrev[j] + 2.0 * sg * (1 - sg) * bend(j);
  }

  var slider = document.getElementById("ex-level");
  var out = document.getElementById("ex-readout");
  var radios = root.querySelectorAll('input[name="ex-r"]');
  var align = document.getElementById("ex-align");
  var gNoise = document.getElementById("ex-noise");
  var gCache = document.getElementById("ex-cache");
  var gDraft = document.getElementById("ex-draft");
  var zone = document.getElementById("ex-zone");
  var zoneLabelL = document.getElementById("ex-zone-l");
  var zoneLabelR = document.getElementById("ex-zone-r");
  var axis = document.getElementById("ex-axis");

  var plotW = W - ML - MR;
  var y0 = MT + (HT - MT - MB) / 2;
  function px(h) { return ML + (h * plotW) / (H - 1); }
  function py(v) { return y0 - v * SCALE; }
  function line(vals) {
    return vals.map(function (v, h) { return px(h).toFixed(1) + "," + py(v).toFixed(1); }).join(" ");
  }
  function text(x, y, str, anchor) {
    var t = document.createElementNS(NS, "text");
    t.setAttribute("x", x); t.setAttribute("y", y); t.setAttribute("class", "s-label");
    if (anchor) t.setAttribute("text-anchor", anchor);
    t.textContent = str;
    return t;
  }

  // static axis
  var base = document.createElementNS(NS, "line");
  base.setAttribute("x1", ML); base.setAttribute("x2", W - MR);
  base.setAttribute("y1", y0); base.setAttribute("y2", y0); base.setAttribute("class", "s-axis");
  axis.appendChild(base);
  [1, 8, 16, 24, 32].forEach(function (h) {
    axis.appendChild(text(px(h - 1), HT - 20, String(h), "middle"));
  });
  axis.appendChild(text(ML + plotW / 2, HT - 4, "action step h", "middle"));
  axis.appendChild(text(ML - 8, y0 + 4, "0", "end"));

  function selectedR() {
    for (var k = 0; k < radios.length; k++) if (radios[k].checked) return parseInt(radios[k].value, 10);
    return 16;
  }
  function fmt(x, d) { return x.toFixed(d); }

  function render() {
    var s = parseInt(slider.value, 10);
    var sg = sigma(s);
    var R = selectedR();
    var aligned = align.checked;
    var tail = H - R;           // positions h < tail come from the shifted suffix (0-indexed)
    var noise = epsNew.slice();
    var cache = [], draft = [];
    for (var h = 0; h < H; h++) {
      var q = h < tail ? h + R : h;                   // Eq. 6
      var xq = cached(q, sg);
      cache.push(xq);
      draft.push(aligned ? xq + sg * (epsNew[h] - epsPrev[q]) : xq);   // Eq. 7
    }
    gNoise.setAttribute("points", line(noise));
    gCache.setAttribute("points", line(cache));
    gDraft.setAttribute("points", line(draft));

    var zx = px(0) - plotW / (2 * (H - 1));
    var zw = px(tail - 1) - px(0) + plotW / (H - 1);
    zone.setAttribute("x", zx); zone.setAttribute("width", Math.max(zw, 0));
    zone.setAttribute("y", MT); zone.setAttribute("height", HT - MT - MB);
    zoneLabelL.setAttribute("x", zx + 6);
    zoneLabelL.setAttribute("y", MT + 12);
    zoneLabelR.setAttribute("x", W - MR - 4);
    zoneLabelR.setAttribute("y", MT + 12);

    var msg = "Noise level s = " + s + " of " + S + " (σ = " + fmt(sg, 3) + "). ";
    if (s === 0) {
      msg += aligned
        ? "At σ = 1 the correction swaps the old noise for the new one, so the draft equals the fresh noise εₙ at every position."
        : "Without alignment the draft starts from the previous replan’s noise, which no longer matches the fresh noise εₙ (grey).";
    } else if (s === S) {
      msg += "At σ = 0 the correction vanishes: the draft is exactly the clean actions the previous solve produced, shifted to the new horizon.";
    } else {
      msg += aligned
        ? "The draft is the shifted cached state plus σ times the gap between new and old noise, so the cached curve’s shape is kept and only its noise level is re-matched."
        : "The draft is the shifted cached state, still tied to the previous noise realization.";
    }
    out.textContent = msg;
  }

  slider.addEventListener("input", render);
  align.addEventListener("change", render);
  radios.forEach(function (r) { r.addEventListener("change", render); });
  render();
})();
