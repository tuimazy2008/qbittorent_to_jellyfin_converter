// "Проверить" buttons: send the current form values to /api/test/<service>.
document.addEventListener("click", async (ev) => {
  const btn = ev.target.closest("button[data-test]");
  if (!btn) return;
  const out = btn.parentElement.querySelector(".test-result");
  out.className = "test-result";
  out.textContent = "проверяю…";
  btn.disabled = true;
  try {
    const r = await fetch("/api/test/" + btn.dataset.test, { method: "POST", body: new FormData(btn.closest("form")) });
    const j = await r.json();
    out.textContent = j.message;
    out.classList.add(j.ok ? "ok" : "bad");
    if (j.models) {
      const dl = document.getElementById("llm-models");
      if (dl) dl.replaceChildren(...j.models.map((m) => Object.assign(document.createElement("option"), { value: m })));
    }
  } catch (e) {
    out.textContent = "Ошибка: " + e;
    out.classList.add("bad");
  } finally {
    btn.disabled = false;
  }
});
