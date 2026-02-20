/* =================================================================
   Heathcliff Blob UI – chat.js
   Chat overlay logic, API communication, DOM manipulation.
   Depends on blob.js exposing  window.setBlobState(name)
   ================================================================= */

const chatOverlay = document.getElementById("chat-overlay");
const chatInput = document.getElementById("chat-input");
const sendBtn = document.getElementById("send-btn");

// =====================================================================
//  DOM HELPERS
// =====================================================================

/**
 * Create and append a chat bubble.
 * When role is "user" the overlay is cleared first so only the latest
 * query / response pair is ever visible.
 */
function addBubble(text, role) {
  if (role === "user") {
    chatOverlay.innerHTML = "";
  }

  const bubble = document.createElement("div");
  bubble.className = "chat-bubble " + role;
  bubble.textContent = text;
  chatOverlay.appendChild(bubble);
}

/** Show an animated "Thinking" indicator. */
function showThinking() {
  removeThinking();
  const bubble = document.createElement("div");
  bubble.className = "chat-bubble thinking";
  bubble.textContent = "Thinking";
  chatOverlay.appendChild(bubble);
}

/** Remove the thinking indicator. */
function removeThinking() {
  const el = chatOverlay.querySelector(".thinking");
  if (el) el.remove();
}

// =====================================================================
//  API COMMUNICATION
// =====================================================================

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;

  chatInput.value = "";
  sendBtn.disabled = true;

  // Show user bubble & switch blob to thinking
  addBubble(text, "user");
  window.setBlobState("thinking");
  showThinking();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    removeThinking();

    if (response.ok) {
      const data = await response.json();
      window.setBlobState("speaking");
      addBubble(data.response, "assistant");

      // Return to idle after a delay proportional to response length
      const delay = Math.min(Math.max(data.response.length * 30, 2000), 8000);
      setTimeout(() => window.setBlobState("idle"), delay);
    } else {
      window.setBlobState("idle");
      addBubble("Something went wrong. Please try again.", "assistant");
    }
  } catch (_err) {
    removeThinking();
    window.setBlobState("idle");
    addBubble("Could not reach Heathcliff. Is the server running?", "assistant");
  }

  sendBtn.disabled = false;
  chatInput.focus();
}

// =====================================================================
//  EVENT LISTENERS
// =====================================================================

sendBtn.addEventListener("click", sendMessage);

chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Escape clears the input field
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    chatInput.value = "";
    chatInput.focus();
  }
});

// Auto-focus on load
chatInput.focus();
