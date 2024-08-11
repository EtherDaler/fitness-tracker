document.getElementById("submit").addEventListener("click", async () => {
  const email = document.getElementById("email").value;

  let response = await fetch("/api/v1/auth/forget-password", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ email }),
  });

  try {
    const data = await response.json();

    if (response.status >= 400) {
      document.getElementById("submit").disabled = false;

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
        "Please check your email for the password reset link. Redirecting to the login page...",
    });
    setTimeout(() => {
      window.location.href = "/login";
    }, 5000);
  } catch {
    return iziToast.show({
      color: "red",
      position: "topRight",
      timeout: 2000,
      message: "Внутренняя ошибка сервера. Пожалуйста, повторите попытку позже",
    });
  }
});
