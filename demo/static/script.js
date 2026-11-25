const $ = (id) => document.getElementById(id);
const input = $("review-input");
const analyzeBtn = $("analyze-btn");
const modelReady = !analyzeBtn.disabled;

const LABELS = {
  food: "Food",
  price: "Price",
  service: "Service",
  ambiance: "Ambiance",
  miscellaneous: "Miscellaneous",
};

function updateCount() {
  $("char-num").textContent = input.value.length;
}

function show(view) {
  for (const id of ["result-empty", "result-error", "result"]) {
    $(id).classList.toggle("hidden", id !== view);
  }
}

function showError(message) {
  $("result-error").textContent = message;
  show("result-error");
}

function render(data) {
  const tags = $("tags");
  tags.replaceChildren(
    ...data.aspects.map((a) => {
      const el = document.createElement("span");
      el.className = "tag";
      el.dataset.aspect = a;
      el.textContent = LABELS[a] || a;
      return el;
    }),
  );

  $("score-chart").replaceChildren(
    ...data.scores.map((s) => {
      const li = document.createElement("li");
      li.dataset.aspect = s.aspect;
      li.className = s.predicted ? "is-on" : "is-off";
      li.innerHTML = `
        <span class="sc-name"></span>
        <span class="sc-track">
          <span class="sc-fill" style="width:${(s.score * 100).toFixed(1)}%"></span>
          <span class="sc-tick" style="left:${(s.threshold * 100).toFixed(1)}%"></span>
        </span>
        <span class="sc-val"></span>
        <span class="sc-decision"></span>`;
      li.querySelector(".sc-name").textContent = LABELS[s.aspect] || s.aspect;
      li.querySelector(".sc-val").textContent = `${s.score.toFixed(3)} / ${s.threshold.toFixed(2)}`;
      li.querySelector(".sc-decision").textContent = s.predicted ? "Predicted" : "Not predicted";
      li.title = `${LABELS[s.aspect]}: score ${s.score.toFixed(3)}, threshold ${s.threshold.toFixed(2)}`;
      return li;
    }),
  );

  $("fallback-note").classList.toggle("hidden", !data.fallback_used);
  $("latency").textContent = `${data.latency_ms} ms on CPU`;
  $("processed-text").textContent = data.processed_text;
  show("result");
}

async function analyze() {
  const text = input.value.trim();
  if (!text) {
    showError("Enter a review first, or pick one of the samples.");
    input.focus();
    return;
  }

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";
  try {
    const resp = await fetch("/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await resp.json().catch(() => ({}));
    if (!resp.ok) throw new Error(data.error || `Request failed (${resp.status}).`);
    render(data);
  } catch (err) {
    showError(err instanceof TypeError ? "Could not reach the server. Check your connection and try again." : err.message);
  } finally {
    analyzeBtn.disabled = !modelReady;
    analyzeBtn.textContent = "Analyze review";
  }
}

input.addEventListener("input", updateCount);
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) analyze();
});
analyzeBtn.addEventListener("click", analyze);
$("clear-btn").addEventListener("click", () => {
  input.value = "";
  updateCount();
  show("result-empty");
  input.focus();
});
$("sample-list").addEventListener("click", (e) => {
  const btn = e.target.closest(".sample");
  if (!btn) return;
  input.value = btn.dataset.text;
  updateCount();
  if (modelReady) analyze();
});
