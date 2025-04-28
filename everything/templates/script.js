const sendButton = document.getElementById("send-button");
const userInput = document.getElementById("user-input");
const chatBox = document.getElementById("chat-box");

sendButton.addEventListener("click", async () => {
  const message = userInput.value;
  if (message.trim() === "") return;

  // Display user message
  chatBox.innerHTML += `<div class="user-message"><strong>You:</strong> ${message}</div>`;
  userInput.value = ""; // Clear input field

  // Send message to the backend
  const response = await fetch("http://localhost:5000/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message: message }),
  });

  const data = await response.json();
  const botResponse = data.response;

  // Display bot response
  chatBox.innerHTML += `<div class="bot-message"><strong>Chatbot:</strong> ${botResponse}</div>`;
  chatBox.scrollTop = chatBox.scrollHeight; // Scroll to the bottom
});
