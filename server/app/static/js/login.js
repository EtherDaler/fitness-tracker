document.getElementById("check").addEventListener("click", () => {
  if (document.getElementById("check").checked) {
    document.getElementById("submit").disabled = false;
  } else {
    document.getElementById("submit").disabled = true;
  }
});

document.getElementById("eye").addEventListener("click", () => {
  let password = document.getElementById("password");
  let eye = document.getElementById("eye");

  if (password.type === "password") {
    password.type = "text";
  } else {
    password.type = "password";
  }
});

document.getElementById("submit")?.addEventListener("click", async (e) => {
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;

  if (!email || !password) {
    return iziToast.show({
      color: "yellow", // blue, red, green, yellow
      position: "topRight",
      timeout: 5000,
      message: "Требуется адрес электронной почты и пароль!",
    });
  }

  const response = await fetch("/api/v1/auth/login", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email, password }),
  });

  try {
    const data = await response.json();

    if (response.status >= 400) {
      let message =
        typeof data.detail === "string" ? data.detail : data.detail[0].msg;

      return iziToast.show({
        color: "red",
        position: "topRight",
        timeout: 5000,
        message: message,
      });
    }

    iziToast.show({
      color: "green",
      position: "topRight",
      timeout: 5000,
      message:
        "Пользователь успешно вошел в систему. Перенаправление на панель мониторинга...",
    });

    setCookie("access_token", data.access_token);

    setTimeout(() => {
      window.location.href = "/dashboard";
    }, 2000);
  } catch {
    return iziToast.show({
      color: "red",
      position: "topRight",
      timeout: 2000,
      message: "Внутренняя ошибка сервера. Пожалуйста, повторите попытку позже",
    });
  }
});
