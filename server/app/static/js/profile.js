const fileInput = document.getElementById("file-input");
const profilePic = document.getElementById("profile-pic");
const activityLevels = [1, 2, 3];
var activityLevel = 1;

// Проверка наличия DOM элементов
if (fileInput && profilePic) {
  profilePic.addEventListener("click", () => {
    fileInput.click();
  });

  fileInput.addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function (e) {
      profilePic.src = e.target.result;
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("photo", file);

    const accessToken = getCookie("access_token");

    const res = await fetch("/api/v1/users/photo", {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: formData,
    });

    if (res.status === 401) {
      window.location.href = "/login";
    } else if (res.status === 409) {
      iziToast.show({
        color: "red",
        position: "topRight",
        timeout: 5000,
        message: "Недопустимый тип файла!",
      });
    } else {
      iziToast.show({
        color: "green",
        position: "topRight",
        timeout: 5000,
        message: "Фотография профиля успешно обновлена",
      });
    }
  });
}

function changeActivityLevel(level) {
  activityLevel = level;
  document.getElementById(`btn-activity-level-${level}`).style.backgroundColor =
    "#958be4";

  activityLevels
    .filter((l) => l !== level)
    .forEach((l) => {
      document.getElementById(`btn-activity-level-${l}`).style = "";
    });
}

document.getElementById("male")?.addEventListener("click", () => {
  document.getElementById("female").checked = false;
  document.getElementById("male").checked = true;
});

document.getElementById("female")?.addEventListener("click", () => {
  document.getElementById("male").checked = false;
  document.getElementById("female").checked = true;
});

document.getElementById("edit")?.addEventListener("click", async () => {
  let name = document.getElementById("name").value;
  let gender = document.getElementById("male").checked ? "male" : "female";
  let height = parseInt(document.getElementById("height").value);
  let weight = parseInt(document.getElementById("weight").value);
  let age = parseInt(document.getElementById("age").value);
  let desiredWeight = parseInt(document.getElementById("desired-weight").value);

  // Проверка корректности данных
  if (Number.isNaN(activityLevel) || Number.isNaN(weight) || Number.isNaN(height) ||
      Number.isNaN(desiredWeight) || Number.isNaN(age)) {
    return iziToast.show({
      color: "yellow",
      position: "topRight",
      timeout: 5000,
      message: "Все поля обязательны для заполнения",
    });
  }

  const accessToken = getCookie("access_token");
  if (!accessToken) {
    window.location.href = "/login";
    return;
  }

  const response = await fetch("/api/v1/users", {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${accessToken}`,
    },
    body: JSON.stringify({
      name,
      gender,
      height,
      weight,
      age,
      desired_weight: desiredWeight,
      activity_level: activityLevel,
    }),
  });

  const resData = await response.json();

  if (response.status >= 500) {
    return iziToast.show({
      color: "red",
      position: "topRight",
      timeout: 5000,
      message: "Произошла ошибка на сервере. Попробуйте позже",
    });
  }

  if (response.status === 409) {
    return iziToast.show({
      color: "red",
      position: "topRight",
      timeout: 5000,
      message: resData.message || "Номер телефона уже существует!",
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
      message: "Ошибка валидации данных",
    });
  }

  iziToast.show({
    color: "green",
    position: "topRight",
    timeout: 5000,
    message: "Профиль успешно обновлен!",
  });

  setTimeout(() => {
    window.location.href = "/dashboard";
  }, 2000);
});

document.getElementById("submit")?.addEventListener("click", () => {
  window.location.href = "/dashboard";
});
