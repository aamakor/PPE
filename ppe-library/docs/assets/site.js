"use strict";
const status = document.getElementById("copy-status");
document.querySelectorAll("[data-copy]").forEach(button => {
  button.addEventListener("click", async () => {
    const content = document.getElementById(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(content.textContent);
      const original = button.textContent;
      button.textContent = "Copied";
      if (status) status.textContent = "Code copied to clipboard.";
      setTimeout(() => { button.textContent = original; }, 1600);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(content);
      const selection = window.getSelection();
      selection.removeAllRanges(); selection.addRange(range);
      if (status) status.textContent = "Code selected. Use your keyboard copy command.";
    }
  });
});
const count = document.getElementById("objective-count");
const controls = document.getElementById("weight-controls");
function updatePreference() {
  const weights = Array.from(controls.querySelectorAll("input"), input => Number(input.value));
  controls.querySelectorAll("output").forEach((output, index) => { output.textContent = weights[index].toFixed(1); });
  const selected = weights.filter(value => value > 0).length;
  document.getElementById("preference-status").textContent = selected ? `${selected} objective${selected === 1 ? "" : "s"} selected` : "Select at least one objective.";
  document.getElementById("preference-code").textContent = selected ? `ScriptedInteraction([[${weights.map(value => value === 0 ? "0" : String(-value)).join(", ")}]])` : "# Select at least one objective above.";
}
function renderWeights() {
  controls.replaceChildren();
  for (let i = 0; i < Number(count.value); i++) {
    const row = document.createElement("div"); row.className = "weight-row";
    const label = document.createElement("label"); label.htmlFor = `weight-${i}`; label.textContent = `Objective ${i + 1}`;
    const input = document.createElement("input"); input.id = label.htmlFor; input.type = "range"; input.min = "0"; input.max = "1"; input.step = "0.1"; input.value = i === 0 ? "1" : "0";
    const output = document.createElement("output"); output.htmlFor = input.id;
    input.addEventListener("input", updatePreference);
    row.append(label, input, output); controls.append(row);
  }
  updatePreference();
}
if (count && controls) { count.addEventListener("change", renderWeights); renderWeights(); }
