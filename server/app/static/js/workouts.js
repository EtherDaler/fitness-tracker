const customWorkoutList = document.getElementById("custom-workout-list");
const generatedWorkoutList = document.getElementById("generated-workout-list");

for (const workout of workouts) {
  var exercises = ""
  for (const exercise in workout.exercises) {
    const el = `
    <div class="workout-coolection">
      <div class="workout-image">
        <img src="../static/gif/${workout.exercises[exercise].gif_link}" alt="exercise animation" width="50" height="100">
      </div>
      <div class="workout-name-wrap">
        <h5>${workout.exercises[exercise].name}</h5>
      </div>
    </div>`;
    exercises += el;
  }
  const details = document.createElement("details");
  details.className = "dashboard-page";
  details.innerHTML = `
  <summary>
    <div class="workout-card gap" style="background: rgba(0,0,0,0)">
      <h4>${workout.name}</h4>
      <img src="../static/images/gradient-arrow.svg" alt="" />
    </div>
  </summary>
  <div class="wrokouts-flexed" style="width: 80%; margin: auto; margin-top: 20px">${exercises}</div>
  <button class="btn" onclick="window.location.href='/workouts/${workout.id}/start'" style="width: 20%; margin: auto; margin-top:40px">Начать</button>
  `;
  details.style.cursor = "pointer";
  // details.onclick = function () {
  //   window.location.href = `/workouts/${workout.id}/start`;
  // };

  if (workout.description === "auto-generated") {
    generatedWorkoutList.appendChild(details);
  } else {
    customWorkoutList.appendChild(details);
  }
}

var workoutCards = document.querySelectorAll(".workout-card");
workoutCards.forEach(function (card) {
  card.addEventListener("click", function () {
    var popupWrap = this.querySelector(".exercise-popup-wraps");
    popupWrap.classList.add("active");
  });
});

var closeBtns = document.querySelectorAll(".cross");
closeBtns.forEach(function (btn) {
  btn.addEventListener("click", function (event) {
    var popupWrap = this.closest(".exercise-popup-wraps");
    popupWrap.classList.remove("active");
    event.stopPropagation();
  });
});
