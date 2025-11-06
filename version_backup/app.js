const API = "/ask";
const URL_API = "/add-url";

const form = document.getElementById("ask-form");
const urlForm = document.getElementById("url-form");
const textarea = document.getElementById("question");
const submitBtn = document.getElementById("submit");
const clearBtn = document.getElementById("clear");
const result = document.getElementById("result");
const urlSection = document.getElementById("url-input-section");
const urlInput = document.getElementById("url-input");
const urlStatus = document.getElementById("url-status");
const addUrlBtn = document.getElementById("add-url-btn");

function renderMarkdown(text) {
  return text.replace(/^### (.*)$/gm, '<h3>$1</h3>');
}

async function ask(q, provider) {
  submitBtn.disabled = true;
  result.innerHTML = "Thinking...";
  urlSection.style.display = "none";
  urlStatus.textContent = "";
  
  try {
    const resp = await fetch(API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q, provider: provider })
    });
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    const data = await resp.json();
    result.innerHTML = renderMarkdown(data.markdown);
    
    // Show URL input if answer indicates insufficient information
    if (data.needs_url) {
      urlSection.style.display = "block";
    }
  } catch (e) {
    result.innerHTML = `<span class="error">Error: ${e.message}</span>`;
  } finally {
    submitBtn.disabled = false;
  }
}

async function addUrl(url) {
  addUrlBtn.disabled = true;
  urlStatus.textContent = "Adding URL to knowledge base...";
  urlStatus.className = "";
  
  try {
    const resp = await fetch(URL_API, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: url })
    });
    
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    
    const data = await resp.json();
    if (data.success) {
      urlStatus.textContent = "✅ " + data.message;
      urlStatus.className = "success";
      urlInput.value = "";
      
      // Ask the question again after a short delay
      setTimeout(() => {
        const question = textarea.value.trim();
        if (question) {
          const provider = document.getElementById("provider").value;
          ask(question, provider);
        }
      }, 2000);
    } else {
      urlStatus.textContent = "❌ " + data.message;
      urlStatus.className = "error";
    }
  } catch (e) {
    urlStatus.textContent = "❌ Error: " + e.message;
    urlStatus.className = "error";
  } finally {
    addUrlBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = textarea.value.trim();
  if (!q) return;
  const provider = document.getElementById("provider").value;
  ask(q, provider);
});

urlForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const url = urlInput.value.trim();
  if (!url) return;
  addUrl(url);
});

clearBtn.addEventListener("click", () => {
  textarea.value = "";
  result.textContent = "";
  urlSection.style.display = "none";
  urlStatus.textContent = "";
  textarea.focus();
});

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    textarea.value = chip.textContent;
    textarea.focus();
  });
});
