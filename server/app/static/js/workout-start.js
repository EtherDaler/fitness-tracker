let startTime;
let endTime;
let seconds = 0;
let minutes = 0;
let isResting = false;
let isDownloading = false;
let restTimer = 0;
let repetitions = 0;
let connectionId = "";
let isCompleted = false;
const fps = 21;
const interval = 1000 / fps;

const video = document.getElementById("video");
const helperVideo = document.getElementById("helper-video");
const repetitionsCountElement = document.getElementById("repetition-count");
const timerElement = document.getElementById("timer");
const completeButton = document.getElementById("complete-btn");
const downloadButtonElement = document.getElementById("download-button");
const protocol = window.location.protocol === "https:" ? "wss" : "ws";
const ws = new WebSocket(`${protocol}://${window.location.host}/api/v1/ws`);
const accessToken = getCookie("access_token");
// const receivedImage = document.getElementById("received-image");

let currentExercise;
let currentExerciseIdx = 0;
let exercises = exercise ? [exercise] : workout.exercises;
let screenModal = document.getElementById("screenModal");
// Get the modal
let modal = document.getElementById("myModal");

// MODAL TIMER
// Start with an initial value of 20 seconds
const TIME_LIMIT = 10;

// Initially, no time has passed, but this will count up
// and subtract from the TIME_LIMIT
let timePassed = 0;
let timeLeft = TIME_LIMIT;
const COLOR_CODES = {
  info: {
    color: "green"
  }
};

let remainingPathColor = COLOR_CODES.info.color;

let timerInterval = null;

const FULL_DASH_ARRAY = 283;

let audio = document.getElementById('music')
let audio_names = [
  'Energetic-Indie-Rock(chosic.com).mp3',
  'Jump-chosic.com_.mp3',
  'Luke-Bergs-Ascensionmp3(chosic.com).mp3',
  'Luke-Bergs-AuroraMp3(chosic.com).mp3',
  'Luke-Bergs-Beach-Vibes(chosic.com).mp3',
  'Luke-Bergs-Bliss(chosic.com) (1).mp3',
  'Luke-Bergs-Burning-In-My-Soul(chosic.com) (1).mp3',
  'Luke-Bergs-Burning-In-My-Soul(chosic.com).mp3',
  'Luke-Bergs-Dance-MP3(chosic.com).mp3',
  'Luke-Bergs-Daybreak(chosic.com).mp3',
  'Luke-Bergs-Dayfox-I´m-Happy(chosic.com).mp3',
  'Luke-Bergs-Paradise-chosic.com_.mp3',
  'Luke-Bergs-Soulful_MP3(chosic.com).mp3',
  'Luke-Bergs-Summertimemp3(chosic.com).mp3',
  'Luke-Bergs-Waesto-Follow-The-Sun(chosic.com).mp3'
];

function getRandomTrack() {
  let randomIndex = Math.floor(Math.random() * audio_names.length);
  return audio_names[randomIndex];
}


// Divides time left by the defined time limit.
function calculateTimeFraction() {
  const rawTimeFraction = timeLeft / TIME_LIMIT;
  return rawTimeFraction - (1 / TIME_LIMIT) * (1 - rawTimeFraction);
}
    
// Update the dasharray value as time passes, starting with 283
function setCircleDasharray() {
  const circleDasharray = `${(
    calculateTimeFraction() * FULL_DASH_ARRAY
  ).toFixed(0)} 283`;
  document
    .getElementById("base-timer-path-remaining")
    .setAttribute("stroke-dasharray", circleDasharray);
}

function startTimer() {
  let startTime = performance.now(); // Текущее время

  function updateTimer(timestamp) {
    const elapsedTime = Math.floor((timestamp - startTime) / 1000); // Прошедшее время в секундах
    timeLeft = TIME_LIMIT - elapsedTime; // Обновляем оставшееся время

    if (timeLeft <= 0) {
      timeLeft = 0;
      document.getElementById("base-timer-label").innerHTML = "Start"; // Отображаем "Start", когда время истекло
      setCircleDasharray();
      // Таймер завершен, можно запустить упражнение
      initializeExercise(exercises[currentExerciseIdx]);
      video.play();
      return; // Останавливаем обновление таймера
    }

    // Обновляем отображение таймера
    document.getElementById("base-timer-label").innerHTML = formatTimeLeft(timeLeft);
    setCircleDasharray();

    // Продолжаем обновление на следующем кадре
    requestAnimationFrame(updateTimer);
  }

  // Запускаем первый кадр
  requestAnimationFrame(updateTimer);
}

document.getElementById("app").innerHTML = `...`;
startTimer();


document.getElementById("app").innerHTML = `
<div class="base-timer">
  <svg class="base-timer__svg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <g class="base-timer__circle">
      <circle class="base-timer__path-elapsed" cx="50" cy="50" r="45"></circle>
      <path
        id="base-timer-path-remaining"
        stroke-dasharray="283"
        class="base-timer__path-remaining ${remainingPathColor}"
        d="
          M 50, 50
          m -45, 0
          a 45,45 0 1,0 90,0
          a 45,45 0 1,0 -90,0
        "
      ></path>
    </g>
  </svg>
  <span id="base-timer-label" class="base-timer__label">
    ${formatTimeLeft(timeLeft)}
  </span>
</div>
`;

function formatTimeLeft(time) {
  // The largest round integer less than or equal to the result of time divided being by 60.
  const minutes = Math.floor(time / 60);
  
  // Seconds are the remainder of the time divided by 60 (modulus operator)
  let seconds = time % 60;
  
  // If the value of seconds is less than 10, then display seconds with a leading zero
  if (seconds < 10) {
    seconds = `${seconds}`;
  }

  if (seconds <= 0) {
    seconds = `Start`;
  }

  // The output in MM:SS format
  return `${seconds}`;
}

// When the user clicks the button, open the modal 
function startModal() {
  modal.style.display = "block";
};

window.onload = function() {
  const track = getRandomTrack();
  audio.src = "../../../static/sounds/"+track;
  startModal();
};

window.addEventListener("load", (event) => {
  setTimeout(function(){ modal.style.display = "none"; }, 10500);
});



if (exercises.length > 0) {
  initializeExercise(exercises[currentExerciseIdx]);
}

initializeVideoStream();

setTimeout(() => {
  video.addEventListener("play", startVideoProcessing);
  initializeVideoStream();
}, 10000);

ws.onmessage = handleWebSocketMessage;
ws.onopen = function () {
    var t = setInterval(function(){
        if (ws.readyState != 1) {
            clearInterval(t);
            return;
        }
        ws.send(JSON.stringify({type:"ping"}));
    }, 30000);
}

completeButton.addEventListener("click", completeWorkout);

function initializeExercise(exercise) {
  currentExercise = exercise;
  restTimer = currentExercise.rest_time;
  helperVideo.src = `../../static/videos/${currentExercise.video_link}`;
  document.querySelectorAll("[type='exercise-name']").forEach((el) => {
    el.innerText = currentExercise.name;
  });
}

function initializeVideoStream() {
  if (navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        video.srcObject = stream;
      })
      .catch(() => {
        showToast(
          "red",
          "Пожалуйста, разрешите доступ к камере и микрофону, чтобы начать тренировку"
        );
        redirectToWorkouts();
      });
  } else {
    showToast("red", "Camera access is not supported in this browser");
    redirectToWorkouts();
  }
}

function handleWebSocketMessage(event) {
  const message = JSON.parse(event.data);

  // We do not process the image
  // if (message.type === "image") {
  //   receivedImage.src = `data:image/jpeg;base64,${message.data}`;
  // }

  if (message.type === "count") {
    repetitions = parseInt(message.data);
    repetitionsCountElement.innerText = repetitions;
  }
  if (message.type === "pong") {
    console.log("Received pong from server");
  }
  connectionId = message.connection_id;
}

function startVideoProcessing() {
  startTime = Date.now();
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");

  function sendFrame() {
    if (video.paused || video.ended) return;

    const aspectRatio = video.videoWidth / video.videoHeight;
    const targetHeight = 360;
    const targetWidth = aspectRatio * targetHeight;

    canvas.width = targetWidth;
    canvas.height = targetHeight;
    context.drawImage(video, 0, 0, targetWidth, targetHeight);

    canvas.toBlob(
      (blob) => {
        if (blob) {
          blob.arrayBuffer().then((buffer) => {
            /*
            bufferToBase64(buffer).then(base64String => {
              const data = {
                type: currentExercise.exercise_id,
                data: base64String,
                is_resting: isResting,
                is_downloading: isDownloading,
                is_completed: isCompleted,
              };
              ws.send(JSON.stringify(data));
            });
            */
            let b64Data = isResting ? "" : bufferToBase64(buffer);
            const data = JSON.stringify({
              type: currentExercise.exercise_id,
              data: b64Data,
              is_resting: isResting,
              is_downloading: isDownloading,
              is_completed: isCompleted,
            });
            ws.send(data);
          });
        }
      },
      "image/jpeg",
      0.5
    );

    setTimeout(sendFrame, interval);
  }

  setInterval(updateTimer, 1000);

  sendFrame();
}

async function completeWorkout() {
  isDownloading = true;
  completeButton.disabled = true;
  completeButton.innerText = "Скачивание...";

  try {
    // no need to download the video currently
    // const response = await fetch(
    //   `/api/v1/ws/download_video?connection_id=${connectionId}`,
    //   {
    //     method: "GET",
    //     headers: {
    //       "Content-Type": "application/json",
    //       Authorization: `Bearer ${accessToken}`,
    //     },
    //   }
    // );
    // if (!response.ok) {
    //   const data = await response.json();
    //   showToast("red", data.detail);
    //   resetCompleteButton();
    //   return;
    // }
    // const blob = await response.blob();
    // downloadBlob(blob, "video.mp4");
  } catch (e) {
    console.error("Failed to download the video", e);
    showToast("red", "Could not save the video. Please try again later.");
  } finally {
    resetCompleteButton();
  }

  ws.send(JSON.stringify({ type: "reset", connection_id: connectionId }));
  endTime = Date.now();
  endRestPeriod();
}

function showScreenShot() {
  screenModal.style.display = "block";
}

function hideScreenShot() {
    screenModal.style.display = "none";
}

function closeScreenShot() {
    setTimeout(function() { hideScreenShot(); }, 5000);
}

function takeShot() {
  const canvas = document.getElementById('canvas');
  const context = canvas.getContext('2d');
  // Устанавливаем размеры canvas равными размерам видео
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  // Рисуем текущее изображение с видео на canvas
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const watermarkImagePath = "../../static/Logo_Fit_it/Logo_Fit_it.png";
  let image = new Image();
  image.src = watermarkImagePath;
  image.width = 110;
  image.height = 50;
  image.onload = function() {
      // Устанавливаем позиции для водяного знака
      let x = canvas.width - image.width - 10; // 10 пикселей от правого края
      let y = canvas.height - image.height - 10; // 10 пикселей от нижнего края
      context.drawImage(image, x, y, image.width, image.height);

      canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const downloadLink = document.createElement('a');
      downloadLink.href = url;
      downloadLink.download = 'screenshot.png';
      downloadLink.innerText = 'Скачать скриншот';
      downloadLink.className = 'btn'; // Применяем класс для стилизации кнопки

      downloadLink.style.display = 'block';
      downloadLink.style.width = `${canvas.width}px`;
      downloadLink.style.margin = '10px auto';
      downloadLink.style.textAlign = 'center';

      // Добавляем кнопку скачивания под скриншотом
      const downloadContainer = document.getElementById('download-container');
      downloadContainer.innerHTML = ''; // Очищаем предыдущий контент
      downloadContainer.appendChild(downloadLink);
    });
  }
  showScreenShot();
  closeScreenShot();
}


function updateTimer() {
  if (isResting || isDownloading || !startTime) return;

  const totalTimeSpentSeconds = parseInt((Date.now() - startTime) / 1000);
  const displaySeconds = totalTimeSpentSeconds % 60;

  if (totalTimeSpentSeconds % 60 === 0 && totalTimeSpentSeconds > 0) {
    seconds = 0;
    minutes++;
  }

  timerElement.innerText = `${minutes}:${
    displaySeconds < 10 ? "0" : ""
  }${displaySeconds}`;

  if (repetitions >= currentExercise.repetitions && !isResting) {
    endTime = Date.now();
    saveSession();
    startRestPeriod();
  }
}

function saveSession() {
  fetch(`/api/v1/workouts/sessions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      workout_id: workout.id,
      repetitions: repetitions,
      workout_exercise_id: currentExercise.id,
      started_at: new Date(startTime).toISOString(),
      finished_at: new Date(endTime).toISOString(),
    }),
  })
    .then((res) => {
      if (res.status !== 200) {
        showToast("red", "Failed to save the exercise session");
      }
    })
    .catch((err) => {
      console.error("Failed to complete workout", err);
    });
}

function startRestPeriod() {
  isResting = true;
  document.getElementById("time-label").innerText = "Отдых";
  helperVideo.src = "";
  updateRestTimer();
}

function updateRestTimer() {
  timerElement.innerText = `${restTimer >= 0 ? restTimer : 0}s`;
  if (restTimer > 0) {
    setTimeout(() => {
      restTimer--;
      updateRestTimer();
    }, 1000);
  } else if (!isDownloading) {
    endRestPeriod();
  }
}

function endRestPeriod() {
  isResting = false;
  document.getElementById("time-label").innerText = "Длительность";
  startTime = Date.now();
  currentExerciseIdx++;

  if (currentExerciseIdx < exercises.length) {
    initializeExercise(exercises[currentExerciseIdx]);
    repetitions = 0;
    ws.send(JSON.stringify({ type: "reset", connection_id: connectionId }));
    helperVideo.play();
  } else if (!isCompleted) {
    isCompleted = true;
    takeShot()
    showToast("green", "Тренировка завершена");
    setTimeout(() => {
      window.location.href = "/dashboard";
    }, 7000);
  }
}


function bufferToBase64(buffer) {
  let binary = "";
  let bytes = new Uint8Array(buffer);
  let len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

/*
function bufferToBase64(buffer) {
  return new Promise((resolve, reject) => {
    let blob = new Blob([buffer], { type: "application/octet-binary" });
    let reader = new FileReader();
    reader.onloadend = () => resolve(reader.result.split(",")[1]); // Получаем только данные Base64
    reader.onerror = (error) => reject(error);
    reader.readAsDataURL(blob);
  });
}
*/
function showToast(color, message) {
  iziToast.show({
    color: color,
    position: "topRight",
    message: message,
    timeout: 5000,
  });
}

function redirectToWorkouts() {
  setTimeout(() => {
    window.location.href = "/workouts";
  }, 5000);
}

function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  window.URL.revokeObjectURL(url);
}

function resetCompleteButton() {
  completeButton.disabled = false;
  completeButton.innerText = "Завершить";
  isDownloading = false;
}
