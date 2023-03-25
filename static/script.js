document.querySelector(".box input-box").addEventListener("click", function() {
    document.querySelector(".textarea").focus();
  });

  // function openOverlay() {
  //   document.getElementById("overlay").style.display = "flex";
  // }
  
  // function closeOverlay() {
  //   document.getElementById("overlay").style.display = "none";
  // }
  
  // const clickableCard = document.querySelector('.clickable-card');
  // const overlay = clickableCard.querySelector('.overlay');

  // clickableCard.addEventListener('click', function() {
  //   overlay.style.display = 'block';
  // });

  // overlay.addEventListener('click', function(event) {
  //   if (event.target.classList.contains('overlay')) {
  //     overlay.style.display = 'none';
  //   }
  // });

// Get the chat button and chat window elements

// Get the chat button and chat window elements
const chatBtn = document.querySelector('.chat-btn');
const chatContainer = document.querySelector('.chat-container');
const closeBtn = document.querySelector('.close-btn');

// Add a click event listener to the chat button
chatBtn.addEventListener('click', () => {
  // Toggle the chat window
  chatContainer.classList.toggle('show');
});

// Add a click event listener to the close button
closeBtn.addEventListener('click', () => {
  // Hide the chat window
  chatContainer.classList.remove('show');
});

$(document).ready(function() {
  $('form').submit(function(e) {
      e.preventDefault();
      $.ajax({
          url: '/submit-suggestion',
          type: 'POST',
          data: $('form').serialize(),
          success: function(response) {
              var message = JSON.parse(response).message;
              alert(message);
          }
      });
  });
});