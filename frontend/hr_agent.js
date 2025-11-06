const HR_API = "/hr-agent";
let currentService = null;

function openService(service) {
  currentService = service;
  document.querySelectorAll('.hr-service-card').forEach(card => {
    card.style.display = 'none';
  });
  document.getElementById('hr-chat-section').style.display = 'block';
  
  const titles = {
    'leave': '📅 Leave Management',
    'payroll': '💰 Payroll Management',
    'recruitment': '🎯 Recruitment Management'
  };
  
  document.getElementById('chat-title').textContent = titles[service];
  addSystemMessage(`Welcome to ${titles[service]}! How can I help you today?`);
}

function closeService() {
  currentService = null;
  document.getElementById('hr-chat-section').style.display = 'none';
  document.querySelectorAll('.hr-service-card').forEach(card => {
    card.style.display = 'block';
  });
  document.getElementById('hr-chat-messages').innerHTML = '';
}

function addSystemMessage(text) {
  const messagesDiv = document.getElementById('hr-chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message system-message';
  messageDiv.textContent = text;
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addUserMessage(text) {
  const messagesDiv = document.getElementById('hr-chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message user-message';
  messageDiv.textContent = text;
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function addAssistantMessage(text) {
  const messagesDiv = document.getElementById('hr-chat-messages');
  const messageDiv = document.createElement('div');
  messageDiv.className = 'message assistant-message';
  messageDiv.textContent = text;
  messagesDiv.appendChild(messageDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

const hrChatForm = document.getElementById('hr-chat-form');
const hrMessageInput = document.getElementById('hr-message-input');
const hrSendBtn = document.getElementById('hr-send-btn');

hrChatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!currentService) return;
  
  const message = hrMessageInput.value.trim();
  if (!message) return;
  
  addUserMessage(message);
  hrMessageInput.value = '';
  hrSendBtn.disabled = true;
  hrSendBtn.textContent = 'Sending...';
  
  try {
    const resp = await fetch(HR_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        service: currentService,
        message: message
      })
    });
    
    if (!resp.ok) {
      throw new Error(`HTTP ${resp.status}`);
    }
    
    const data = await resp.json();
    addAssistantMessage(data.response);
  } catch (e) {
    addAssistantMessage(`Error: ${e.message}`);
  } finally {
    hrSendBtn.disabled = false;
    hrSendBtn.textContent = 'Send';
  }
});

