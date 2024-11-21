document.getElementById("check").addEventListener("click", () => {
  if (document.getElementById("check").checked) {
    document.getElementById("submit").disabled = false;
  } else {
    document.getElementById("submit").disabled = true;
  }
});

document.getElementById("eye").addEventListener("click", () => {
  let password = document.getElementById("password");

  if (password.type === "password") {
    password.type = "text";
  } else {
    password.type = "password";
  }
});

document.getElementById("submit").addEventListener("click", async (e) => {
  const email = document.getElementById("email").value;
  const name = document.getElementById("name").value;
  const password = document.getElementById("password").value;

  if (!email || !name || !password) {
    return iziToast.show({
      color: "yellow", // blue, red, green, yellow
      position: "topRight",
      timeout: 5000,
      message: "Все поля обязательны для заполнения!",
    });
  }

  const response = await fetch("/api/v1/auth/register", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      email,
      name,
      password,
    }),
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
      message: data.message,
    });

    setCookie("access_token", data.access_token);
    window.location.href = "/login";
  } catch {
    return iziToast.show({
      color: "red",
      position: "topRight",
      timeout: 5000,
      message: "Error during registeration",
    });
  }
});
