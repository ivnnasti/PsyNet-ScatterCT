(function () {
  const html = document.documentElement;
  const stored = localStorage.getItem("theme");
  if (stored) html.setAttribute("data-theme", stored);

  document.getElementById("themeToggle").addEventListener("click", function () {
    const next = html.getAttribute("data-theme") === "dark" ? "light" : "dark";
    html.setAttribute("data-theme", next);
    localStorage.setItem("theme", next);
  });

  fetch("/metrics")
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.rmse !== undefined) {
        document.getElementById("rmseVal").textContent = d.rmse.toFixed(4);
        document.getElementById("ssimVal").textContent = d.ssim.toFixed(4);
      }
    })
    .catch(function () {});

  var plotSection = document.getElementById("plotSection");
  fetch("/plot", { method: "HEAD" })
    .then(function (r) {
      if (r.ok) plotSection.style.display = "";
    })
    .catch(function () {});

  var fileInput = document.getElementById("fileInput");
  var fileName = document.getElementById("fileName");
  fileInput.addEventListener("change", function () {
    fileName.textContent = fileInput.files.length ? fileInput.files[0].name : "Файл не выбран";
  });

  document.getElementById("runBtn").addEventListener("click", function () {
    var file = fileInput.files[0];
    if (!file) {
      showError("Файл не выбран");
      return;
    }
    var spinner = document.getElementById("spinner");
    var resultGrid = document.getElementById("resultGrid");
    var btn = document.getElementById("runBtn");
    hideError();
    resultGrid.style.display = "none";
    spinner.style.display = "";
    btn.disabled = true;
    btn.textContent = "Загрузка...";

    var fd = new FormData();
    fd.append("file", file);
    fetch("/predict", { method: "POST", body: fd })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        spinner.style.display = "none";
        btn.disabled = false;
        btn.textContent = "Запустить";
        if (d.error) {
          showError(
            d.error === "модель не обучена"
              ? "Модель не обучена — сначала запустите train.py"
              : d.error
          );
          return;
        }
        document.getElementById("cbctImg").src = "data:image/png;base64," + d.cbct;
        document.getElementById("predImg").src = "data:image/png;base64," + d.pred;
        resultGrid.style.display = "";
      })
      .catch(function () {
        spinner.style.display = "none";
        btn.disabled = false;
        btn.textContent = "Запустить";
        showError("Ошибка запроса");
      });
  });

  function showError(msg) {
    var el = document.getElementById("errorMsg");
    el.textContent = msg;
    el.style.display = "";
  }

  function hideError() {
    document.getElementById("errorMsg").style.display = "none";
  }
})();
