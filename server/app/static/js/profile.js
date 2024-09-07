const fileInput = document.getElementById("file-input");
const profilePic = document.getElementById("profile-pic");
const activityLevels = [1, 2, 3];
var activityLevel = 1;

function changeActivityLevel(level) {
  activityLevel = level;
  document.getElementById(btn-activity-level-${level}).style.backgroundColor =
    "#958be4";

  const others = activityLevels.filter((l) => l !== level);
  others.forEach((l) => {
    document.getElementById(btn-activity-level-${l}).style = "";
  });
}

document.getElementById("male").addEventListener("click", () => {
  document.getElementById("female").checked = false;
  document.getElementById("male").checked = true;
});

document.getElementById("female").addEventListener("click", () => {
  document.getElementById("male").checked = false;
  document.getElementById("female").checked = true;
});

profilePic.addEventListener("click", () => {
  fileInput.click();
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
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
      Authorization: Bearer ${accessToken},
    },
    body: formData,
  });

  if (res.status === 401) {
    window.location.href = "/login";
  }

  if (res.status === 409) {
    return iziToast.show({
      color: "red", // blue, red, green, yellow
      position: "topRight",
      timeout: 5000,
      message: "Недопустимый тип файла!",
    });
  }

  return iziToast.show({
    color: "green",
    position: "topRight",
    timeout: 5000,
    message: "Фотография профиля успешно обновлена",
  });
});

document.getElementById("edit").addEventListener("click", async () => {
  let name = document.getElementById("name").value;
  let gender = document.getElementById("male").checked ? "male" : "female";
  let height = document.getElementById("height").value;
  let weight = document.getElementById("weight").value;
  let age = document.getElementById("age").value;
  let desiredWeight = document.getElementById("desired-weight").value;

  if (
    Number.isNaN(parseInt(activityLevel)) ||
    Number.isNaN(parseInt(weight)) ||
    Number.isNaN(parseInt(height)) ||
    Number.isNaN(parseInt(desiredWeight)) ||
    Number.isNaN(parseInt(age))
  ) {
    return iziToast.show({
      color: "yellow",
      position: "topRight",
      timeout: 5000,
      message: "Все поля обязательны для заполнения",
    });
  }

  activityLevel = parseInt(activityLevel);
  weight = parseInt(weight);
  height = parseInt(height);
  desiredWeight = parseInt(desiredWeight);
  age = parseInt(age);

  const accessToken = getCookie("access_token");
  if (!accessToken) {
    window.location.href = "/login";
    return;
  }

  const response = await fetch("/api/v1/users", {
    method: "PUT",
    headers: {
      "Content-Type": "application/json",
      Authorization: Bearer ${accessToken},
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
    timeout: 5000,
    message: "Отредактировал ваш профиль!",
  });

  setTimeout(() => {
    window.location.href = "/login";
  }, 2000);
});

document.getElementById("submit").addEventListener("click", () => {
  window.location.href = "/dashboard";
});