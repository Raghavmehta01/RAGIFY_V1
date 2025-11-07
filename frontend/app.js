const API = "/ask";
const URL_API = "/add-url";
const FILE_API = "/upload-file";

const form = document.getElementById("ask-form");
const urlForm = document.getElementById("url-form");
const fileForm = document.getElementById("file-form");
const textarea = document.getElementById("question");
const submitBtn = document.getElementById("submit");
const clearBtn = document.getElementById("clear");
const result = document.getElementById("result");
const urlSection = document.getElementById("url-input-section");
const urlInput = document.getElementById("url-input");
const urlStatus = document.getElementById("url-status");
const addUrlBtn = document.getElementById("add-url-btn");
const fileInput = document.getElementById("file-input");
const uploadFileBtn = document.getElementById("upload-file-btn");
const fileStatus = document.getElementById("file-status");

function renderMarkdown(text) {
  if (!text) return '';
  
  let html = text;
  
  // Process code blocks first (before other processing)
  html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
    return '<pre><code>' + escapeHtml(code) + '</code></pre>';
  });
  
  // Process inline code
  html = html.replace(/`([^`\n]+)`/g, (match, code) => {
    return '<code>' + escapeHtml(code) + '</code>';
  });
  
  // Split by lines for line-by-line processing
  const lines = html.split('\n');
  const processedLines = [];
  let inList = false;
  let listItems = [];
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();
    
    // Headers
    if (line.startsWith('### ')) {
      if (inList) {
        processedLines.push('</ul>');
        inList = false;
      }
      processedLines.push('<h3>' + escapeHtml(line.substring(4)) + '</h3>');
      continue;
    }
    if (line.startsWith('## ')) {
      if (inList) {
        processedLines.push('</ul>');
        inList = false;
      }
      processedLines.push('<h2>' + escapeHtml(line.substring(3)) + '</h2>');
      continue;
    }
    if (line.startsWith('# ')) {
      if (inList) {
        processedLines.push('</ul>');
        inList = false;
      }
      processedLines.push('<h1>' + escapeHtml(line.substring(2)) + '</h1>');
      continue;
    }
    
    // List items
    if (/^[\*\-\+] /.test(line) || /^\d+\. /.test(line)) {
      if (!inList) {
        inList = true;
        processedLines.push('<ul>');
      }
      const listContent = line.replace(/^[\*\-\+] /, '').replace(/^\d+\. /, '');
      processedLines.push('<li>' + processInlineMarkdown(listContent) + '</li>');
      continue;
    }
    
    // End list if we hit a non-list line
    if (inList && line !== '') {
      processedLines.push('</ul>');
      inList = false;
    }
    
    // Empty line
    if (line === '') {
      processedLines.push('');
      continue;
    }
    
    // Regular paragraph
    processedLines.push('<p>' + processInlineMarkdown(line) + '</p>');
  }
  
  // Close any open list
  if (inList) {
    processedLines.push('</ul>');
  }
  
  return processedLines.join('\n');
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function processInlineMarkdown(text) {
  // Escape HTML first
  let html = escapeHtml(text);
  
  // Process bold (**text** or __text__) first - this takes priority over italic
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__([^_]+)__/g, '<strong>$1</strong>');
  
  // Process italic (*text* or _text_) - only match if not already processed as bold
  // Match single asterisks that aren't part of double asterisks
  html = html.replace(/\*([^*\n]+?)\*/g, (match, content) => {
    // Skip if this was already part of a bold match (check if content has HTML tags)
    if (content.includes('<strong>') || content.includes('</strong>')) {
      return match;
    }
    return '<em>' + content + '</em>';
  });
  
  html = html.replace(/_([^_\n]+?)_/g, (match, content) => {
    // Skip if this was already part of a bold match
    if (content.includes('<strong>') || content.includes('</strong>')) {
      return match;
    }
    return '<em>' + content + '</em>';
  });
  
  // Links [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>');
  
  return html;
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

async function uploadFile(file) {
  uploadFileBtn.disabled = true;
  fileStatus.textContent = "Uploading and processing file...";
  fileStatus.className = "";
  
  try {
    const formData = new FormData();
    formData.append("file", file);
    
    const resp = await fetch(FILE_API, {
      method: "POST",
      body: formData
    });
    
    if (!resp.ok) {
      const t = await resp.text();
      throw new Error(`HTTP ${resp.status}: ${t}`);
    }
    
    const data = await resp.json();
    if (data.success) {
      fileStatus.textContent = "✅ " + data.message;
      fileStatus.className = "success";
      fileInput.value = "";
    } else {
      fileStatus.textContent = "❌ " + data.message;
      fileStatus.className = "error";
    }
  } catch (e) {
    fileStatus.textContent = "❌ Error: " + e.message;
    fileStatus.className = "error";
  } finally {
    uploadFileBtn.disabled = false;
  }
}

fileForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return;
  
  // Validate file type
  const ext = file.name.split('.').pop().toLowerCase();
  if (!['pdf', 'doc', 'docx', 'txt'].includes(ext)) {
    fileStatus.textContent = "❌ Please upload a PDF, Word, or Text document (.pdf, .doc, .docx, .txt)";
    fileStatus.className = "error";
    return;
  }
  
  // Validate file size (10MB limit)
  if (file.size > 10 * 1024 * 1024) {
    fileStatus.textContent = "❌ File too large. Maximum size is 10MB.";
    fileStatus.className = "error";
    return;
  }
  
  uploadFile(file);
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
