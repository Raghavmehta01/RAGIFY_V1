const API = "/ask";

const form = document.getElementById("ask-form");
const textarea = document.getElementById("question");
const submitBtn = document.getElementById("submit");
const clearBtn = document.getElementById("clear");
const result = document.getElementById("result");

function renderMarkdown(text) {
  return text.replace(/^### (.*)$/gm, '<h3>$1</h3>');
}

async function ask(q) {
  submitBtn.disabled = true;
  result.innerHTML = "Thinking...";
  try {
    const resp = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q })
    });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    const data = await resp.json();
    result.innerHTML = renderMarkdown(data.markdown);
  } catch (e) {
    result.innerHTML = `<span class="error">Error: ${e.message}</span>`;
  } finally {
    submitBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = textarea.value.trim();
  if (!q) return;
  ask(q);
});

clearBtn.addEventListener("click", () => {
  textarea.value = "";
  result.textContent = "";
  textarea.focus();
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    textarea.value = chip.textContent;
    textarea.focus();
  });
});
