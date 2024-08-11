let customExercises = [];
let editWorkoutPage = document.getElementById("edit-workout-page");
let editBox = document.getElementById("edit-box");
let workoutContainer = document.getElementById("workout-container");

const observer = new MutationObserver(updateUI);
observer.observe(document.getElementById("workout-page"), {
  attributes: true,
  attributeFilter: ["style"],
});

observer.observe(document.getElementById("edit-workout-page"), {
  attributes: true,
  attributeFilter: ["style"],
});

document
  .getElementById("add-workout-submit")
  .addEventListener("click", async () => {
    let exercises = [];
    let name = document.getElementById("workout-name").value;

    if (!name) {
      return iziToast.show({
        color: "yellow",
        position: "topRight",
        timeout: 1500,
        message: "Workout name is required",
      });
    }

    document
      .querySelectorAll("#custom-exercises .timer-edit")
      .forEach((element) => {
        const type = element.querySelector("input[name='type']").value;
        const exercise = {
          repetitions: 0,
          rest_time: 0,
          exercise_id: "",
        };

        if (type === "exercise") {
          const repetitions = parseInt(
            element.querySelector("input[name='repetitions']").value
          );
          const exercise_id = element.querySelector(
            "input[name='exercise_id']"
          ).value;

          if (Number.isNaN(repetitions) || repetitions <= 0) {
            return iziToast.show({
              color: "yellow",
              position: "topRight",
              timeout: 1500,
              message: "Repetitions must be integer",
            });
          }
          exercise.repetitions = repetitions;
          exercise.exercise_id = exercise_id;
        }

        if (type === "rest" && exercises.length > 0) {
          let lastExercise = exercises.pop();
          const restTimer = parseInt(
            element.querySelector("input[name='rest']").value
          );
          if (Number.isNaN(restTimer) || restTimer <= 0) {
            return iziToast.show({
              color: "yellow",
              position: "topRight",
              timeout: 1500,
              message: "Rest time must be integer",
            });
          }

          lastExercise.rest_time = restTimer;
          exercises.push(lastExercise);
          return;
        }

        exercises.push(exercise);
      });

    if (exercises.length === 0) {
      return iziToast.show({
        color: "yellow",
        position: "topRight",
        timeout: 1500,
        message: "Cannot create empty workout",
      });
    }

    const accessToken = getCookie("access_token");
    if (!accessToken) {
      window.location.href = "/login";
      return;
    }

    const payload = {
      name,
      description: `Custom workout created by ${name} with exercises [${exercises
        .map((ex) => ex.exercise_id)
        .join(",")}]`,
      exercises: exercises,
    };

    console.log("Payload: ", payload);

    const response = await fetch("/api/v1/workouts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(payload),
    });

    if (response.status >= 500) {
      try {
        const res = await response.json();
        return iziToast.show({
          color: "yellow",
          position: "topRight",
          timeout: 5000,
          message: res.message,
        });
      } catch {
        return iziToast.show({
          color: "red",
          position: "topRight",
          timeout: 5000,
          message: "Произошла ошибка на сервере. Попробуйте позже",
        });
      }
    }

    if (response.status === 409) {
      return iziToast.show({
        color: "red",
        position: "topRight",
        timeout: 5000,
        message: "Номер телефона уже существует!",
      });
    }

    if (response.status === 401) {
      window.location.href = "/login";
      return;
    }

    if (response.status === 422) {
      return iziToast.show({
        color: "yellow",
        position: "topRight",
        timeout: 5000,
        message: "Данные успешно обновлены",
      });
    }

    iziToast.show({
      color: "green",
      position: "topRight",
      timeout: 3000,
      message: "Created a new exercise!",
    });

    setTimeout(() => {
      window.location.href = "/dashboard";
    }, 3000);
  });

function updateUI(mutations) {
  mutations.forEach((mutation) => {
    if (mutation.type === "attributes" && mutation.attributeName === "style") {
      if (editWorkoutPage.style.display === "none") {
        workoutContainer.innerHTML = "";
        exercises.forEach((exercise) => {
          createWorkoutCard(exercise);
        });
      } else {
        editBox.innerHTML = "";
        editBox.appendChild(createExerciseList());
      }
    }
  });
}

function createWorkoutCard(exercise) {
  const card = document.createElement("div");
  card.className = "workout-card new";
  card.innerHTML = `
  <img
      src="../static/gif/${exercise.gif_link}"
      alt=""
      class="exercise-img"
    />
    <h4>${exercise.name}</h4>
    <img src="../static/images/add-btn.svg" alt="" style="cursor: pointer;" />
    <div class="exercise-popup-wraps">
      <div class="exercise-pop-wrapper">
        <h4>${exercise.name}</h4>
        <div class="workout">
          <img src="../static/gif/${exercise.gif_link}" alt="" />
        </div>
        <div class="timer-count">
          <div class="plus-minus" type="minus" style="cursor: pointer;">
            <img src="../static/images/minus.svg" alt="" />
          </div>
          <div class="timer-wraps">
            <input value="20" name="repetitions"/>
          </div>
          <div class="plus-minus" type="plus" style="cursor: pointer;">
            <img src="../static/images/plus.svg" alt="" />
          </div>
        </div>
        <div class="flex align-center add-flex gap-20">
          <button class="btn" type="add" >Добавить</button>
          <div class="cross">
            <img src="../static/images/cross.svg" alt="" />
          </div>
        </div>
    </div>
  `;

  workoutContainer.appendChild(card);

  let reps = 20;
  const exercisePopup = card.querySelector(".exercise-popup-wraps");
  const closeBtn = card.querySelector(".cross");
  const plusBtn = card.querySelector(".plus-minus[type='plus']");
  const minusBtn = card.querySelector(".plus-minus[type='minus']");
  const repetitionsInput = card.querySelector("input[name='repetitions']");
  const addBtn = card.querySelector(".btn[type='add']");

  card.addEventListener("click", () => exercisePopup.classList.add("active"));
  closeBtn.addEventListener("click", (event) => {
    exercisePopup.classList.remove("active");
    event.stopPropagation();
  });

  repetitionsInput.addEventListener("input", () => {
    let value = parseInt(repetitionsInput.value);
    if (!Number.isNaN(value) || value > 0) {
      reps = value;
    }
  });

  plusBtn.addEventListener("click", () => {
    reps += 1;
    repetitionsInput.value = reps;
  });

  minusBtn.addEventListener("click", () => {
    if (reps <= 1) return;
    reps -= 1;
    repetitionsInput.value = reps;
  });

  addBtn.addEventListener("click", () => {
    const newExercise = {
      name: exercise.name,
      id: Date.now(), // a unique id to identify the exercise based on the time it was added
      exercise_id: exercise.id,
      gif_link: exercise.gif_link,
      video_link: exercise.video_link,
      repetitions: repetitionsInput.value,
      rest_timeout: 0,
    };
    customExercises.push(newExercise);
    console.log("Custom exercises: ", customExercises);

    iziToast.show({
      color: "green", // blue, red, green, yellow
      position: "topRight",
      timeout: 500,
      message: `Added ${exercise.name}`,
    });

    exercisePopup.classList.remove("active");
    showPage("edit-workout-page");
  });
}

function createExerciseList() {
  const list = document.createElement("div");
  list.id = "custom-exercises";
  const title = document.createElement("h5");
  title.textContent = "Список упражнений";
  list.appendChild(title);

  if (customExercises.length > 0) {
    customExercises.forEach((exercise) => {
      const exerciseElement = document.createElement("div");
      exerciseElement.className = "timer-edit";
      exerciseElement.innerHTML = `
        <div class="squat-wrap">
          <img src="../static/gif/${exercise.gif_link}" alt="" />
        </div>
        <input value="exercise" name="type" hidden />
        <h3>${exercise.name}</h3>
        <div class="timer-wrap">
          <input value="${exercise.repetitions}" name="repetitions"/>
          <input value="${exercise.exercise_id}" name="exercise_id" hidden />
          <input value="${exercise.id}" name="id" hidden />
        </div>
        `;

      let repsInput = exerciseElement.querySelector(
        "input[name='repetitions']"
      );

      repsInput?.addEventListener("input", () => {
        const idx = customExercises.findIndex((ex) => ex.id === exercise.id);
        if (idx !== -1) {
          customExercises[idx].repetitions = repsInput.value;
        }
      });

      list.appendChild(exerciseElement);

      if (exercise.rest_timeout > 0) {
        const restElement = document.createElement("div");
        restElement.className = "timer-edit";
        restElement.innerHTML = `
          <div class="center-wrap">
            <img src="../static/images/dots.svg" alt="" />
          </div>
          <input value="rest" name="type" hidden />
          <h3>Отдых</h3>
          <div class="timer-wrap">
            <input value="${exercise.rest_timeout}" name="rest" />
          </div>
          `;

        let restInput = restElement.querySelector("input[name='rest']");

        restInput.addEventListener("input", () => {
          const idx = customExercises.findIndex((ex) => ex.id === exercise.id);
          if (idx !== -1) {
            customExercises[idx].rest_timeout = restInput.value;
          }
        });
        list.appendChild(restElement);
      }
    });
    editBox.appendChild(createButtons());
  } else {
    const container = document.createElement("div");
    container.className = "text-edit-wrap";

    const addExerciseBtn = document.createElement("div");
    addExerciseBtn.className = "list-exercise";
    addExerciseBtn.id = "add-exercise-btn";
    addExerciseBtn.style.cursor = "pointer";
    addExerciseBtn.onclick = () => showPage("workout-page");
    addExerciseBtn.innerHTML =
      "<img src='../static/images/plus.svg' alt='' /> Упражнение";
    container.appendChild(addExerciseBtn);
    list.appendChild(container);
  }

  return list;
}

function createButtons() {
  const buttonsWrapper = document.createElement("div");
  buttonsWrapper.className = "flex align-center gap-20 btns-wrap";

  const restBtn = document.createElement("button");
  restBtn.className = "white-btn";
  restBtn.innerHTML = "<img src='../static/images/plus.svg' alt='' />Отдых";
  buttonsWrapper.appendChild(restBtn);

  const exerciseBtn = document.createElement("button");
  exerciseBtn.className = "white-btn";
  exerciseBtn.onclick = () => showPage("workout-page");
  exerciseBtn.innerHTML =
    "<img src='../static/images/plus.svg' alt='' />Упражнение";
  buttonsWrapper.appendChild(exerciseBtn);

  restBtn.onclick = () => {
    let lastExercise = customExercises[customExercises.length - 1];
    if (lastExercise && lastExercise.rest_timeout > 0) {
      return;
    }

    lastExercise.rest_timeout = 10;
    customExercises.pop();
    customExercises.push(lastExercise);
    updateUI([{ type: "attributes", attributeName: "style" }]);
  };

  return buttonsWrapper;
}

function showPage(page) {
  const editWorkoutPage = document.getElementById("edit-workout-page");
  const workoutPage = document.getElementById("workout-page");

  if (page === "edit-workout-page") {
    editWorkoutPage.style.display = "";
    workoutPage.style.display = "none";
  } else {
    workoutPage.style.display = "";
    editWorkoutPage.style.display = "none";
  }
}

function addNewExercise(exercise) {
  customExercises.push(exercise);
  showPage("edit-workout-page");
}
