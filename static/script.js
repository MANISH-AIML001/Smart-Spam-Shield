const messageInput = document.getElementById("messageInput");
const charCount = document.getElementById("charCount");
const scanBtn = document.getElementById("scanBtn");
const gaugeArc = document.getElementById("gaugeArc");
const gaugeValue = document.getElementById("gaugeValue");
const gaugeLabel = document.getElementById("gaugeLabel");
const verdictBadge = document.getElementById("verdictBadge");
const triggerRow = document.getElementById("triggerRow");
const featurePanel = document.getElementById("featurePanel");
const featureGrid = document.getElementById("featureGrid");
const channelBtns = document.querySelectorAll(".channel-btn");

const CIRCUMFERENCE = 578; // 2 * PI * 92 (matches the SVG radius)

const VERDICT_META = {
  safe: { color: "var(--safe)", label: "looks safe", badge: "Safe message" },
  suspicious: { color: "var(--suspicious)", label: "worth a second look", badge: "Suspicious" },
  spam: { color: "var(--spam)", label: "likely spam", badge: "Spam detected" },
};

messageInput.addEventListener("input", () => {
  charCount.textContent = `${messageInput.value.length} characters`;
});

channelBtns.forEach((btn) => {
  btn.addEventListener("click", () => {
    channelBtns.forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    messageInput.placeholder = btn.dataset.placeholder;
  });
});

scanBtn.addEventListener("click", async () => {
  const message = messageInput.value.trim();
  if (!message) {
    messageInput.focus();
    return;
  }

  scanBtn.disabled = true;
  scanBtn.textContent = "Scanning...";
  gaugeLabel.textContent = "scanning";

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    if (!res.ok) throw new Error("prediction failed");
    const data = await res.json();
    renderResult(data);
  } catch (err) {
    gaugeLabel.textContent = "error — is the server running?";
  } finally {
    scanBtn.disabled = false;
    scanBtn.textContent = "Scan message";
  }
});

function renderResult(data) {
  const meta = VERDICT_META[data.verdict] || VERDICT_META.suspicious;

  // Show the number that matches the verdict's direction:
  // spam verdict -> show spam%, safe/suspicious verdict -> show safe%
  const confidence = data.verdict === "spam" ? data.spam_confidence : data.safe_confidence;

  // Animate the gauge arc
  const offset = CIRCUMFERENCE - (CIRCUMFERENCE * confidence) / 100;
  gaugeArc.style.stroke = meta.color;
  // Force reflow so the transition re-triggers on repeated scans
  gaugeArc.style.transition = "none";
  gaugeArc.style.strokeDashoffset = CIRCUMFERENCE;
  void gaugeArc.getBoundingClientRect();
  gaugeArc.style.transition = "";
  requestAnimationFrame(() => {
    gaugeArc.style.strokeDashoffset = offset;
  });

  gaugeValue.textContent = `${confidence}%`;
  gaugeLabel.textContent = data.verdict === "spam" ? "spam score" : "safe score";

  verdictBadge.textContent = meta.badge;
  verdictBadge.className = `verdict-badge ${data.verdict}`;
  verdictBadge.classList.remove("hidden");

  if (data.triggers && data.triggers.length > 0) {
    triggerRow.innerHTML = data.triggers
      .map((w) => `<span class="trigger-chip">${escapeHtml(w)}</span>`)
      .join("");
    triggerRow.classList.remove("hidden");
  } else {
    triggerRow.classList.add("hidden");
  }

  featureGrid.innerHTML = data.features
    .map(
      (f) => `
      <div class="feature-cell">
        <div class="feature-cell__label">${escapeHtml(f.label)}</div>
        <div class="feature-cell__value">${f.value}</div>
      </div>`
    )
    .join("");
  featurePanel.classList.remove("hidden");
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}