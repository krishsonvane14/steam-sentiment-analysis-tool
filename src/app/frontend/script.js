async function predictSentiment() {
  const text = document.getElementById("single_input").value.trim();
  if (!text) return alert("Please enter a review.");

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    const data = await response.json();

    const logisticPct = (data.logistic_prob * 100).toFixed(2);
    const linearPct = (data.linear_prob * 100).toFixed(2);

    document.getElementById("single-result").innerHTML = `
      <table>
        <tr>
          <th></th>
          <th>Linear Model</th>
          <th>Logistic Model</th>
        </tr>
        <tr>
          <th>Label</th>
          <td>${data.linear_label}</th>
          <td>${data.logistic_label}</th>
        </tr>
        <tr>
          <th>Confidence</th>
          <td>${linearPct}%</th>
          <td>${logisticPct}%</th>
        <tr>
      </table>
    `;
  } catch (err) {
    document.getElementById("single-result").innerText =
      "Error: " + err.message;
  }
}

async function predictFromAppID() {
  const appidText = document.getElementById("appid-input").value.trim();
  if (!appidText) return alert("Please enter an identifier.");

  document.getElementById("appid-loading").innerText = "Fetching reviews...";
  document.getElementById("appid-result").innerHTML = "";

  try {
    const response = await fetch("http://localhost:8000/predict_from_appid", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ appid: parseInt(appidText) }),
    });

    const data = await response.json();
    document.getElementById("appid-loading").innerText = "";

    if (data.error) {
      document.getElementById("appid-result").innerText = data.error;
      return;
    }

    const lp = data.logistic_percentage.toFixed(2);
    const ln = data.linear_percentage.toFixed(2);

    let resultHTML = `
      <h3>${data.name} (AppID ${data.appid})</h3>

      <table>
        <tr>
          <th></th>
          <th>Linear Model</th>
          <th>Logistic Model</th>
        </tr>

        <tr>
          <th>Positive Reviews</th>
          <td>${data.linear_positive} / ${data.total_reviews}</td>
          <td>${data.logistic_positive} / ${data.total_reviews}</td>
        </tr>

        <tr>
          <th>Percentage</th>
          <td>${ln}%</td>
          <td>${lp}%</td>
        </tr>
      </table>

      <h4>Review List</h4>

      <table>
        <tr>
          <th>Review Text</th>
          <th>Linear</th>
          <th>Logistic</th>
        </tr>
    `;

    data.reviews.forEach((r) => {
      const lp = (r.logistic_prob * 100).toFixed(2);
      const ln = (r.linear_prob * 100).toFixed(2);

      resultHTML += `
        <tr>
          <td>${r.text}</td>
          <td>${r.linear_label}<br>${ln}%</td>
          <td>${r.logistic_label}<br>${lp}%</td>
        </tr>
      `;
    });

    resultHTML += "</table>";

    document.getElementById("appid-result").innerHTML = resultHTML;
  } catch (err) {
    document.getElementById("appid-loading").innerText = "";
    document.getElementById("appid-result").innerText = "Error: " + err.message;
  }
}

async function loadMetrics() {
  const res = await fetch("http://localhost:8000/metrics");
  const data = await res.json();

  const linear = data.linear;
  const logistic = data.logistic;

  // num_features
  document.getElementById("linear-features").textContent = linear.num_features;
  document.getElementById("logistic-features").textContent =
    logistic.num_features;

  // Epochs
  document.getElementById("linear-epochs").textContent = linear.epochs;
  document.getElementById("logistic-epochs").textContent = logistic.epochs;

  // learning_rate
  document.getElementById("linear-lr").textContent = linear.learning_rate;
  document.getElementById("logistic-lr").textContent = logistic.learning_rate;
  
  // train_loss
  document.getElementById("linear-tl").textContent = linear.train_loss.toFixed(3);
  document.getElementById("logistic-tl").textContent = logistic.train_loss.toFixed(3);

  // validation_loss
  document.getElementById("linear-vl").textContent = linear.validation_loss.toFixed(3);
  document.getElementById("logistic-vl").textContent = logistic.validation_loss.toFixed(3);
 
  // validation_accuracy
  document.getElementById("linear-va").textContent = (linear.validation_accuracy*100).toFixed(3) + "%";
  document.getElementById("logistic-va").textContent = (logistic.validation_accuracy*100).toFixed(3) + "%";


  // test_accuracy
  document.getElementById("linear-ta").textContent = (linear.test_accuracy*100).toFixed(3) + "%";
  document.getElementById("logistic-ta").textContent = (logistic.test_accuracy*100).toFixed(3) + "%";

}

loadMetrics();

window.onload = function () {
  const imgs = document.querySelectorAll("table img");

  const modal = document.getElementById("img-modal");
  const modalImg = document.getElementById("modal-img");
  const closeBtn = document.querySelector(".modal-close");

  imgs.forEach((img) => {
    img.addEventListener("click", () => {
      modal.style.display = "block";
      modalImg.src = img.src;
    });
  });

  closeBtn.onclick = function () {
    modal.style.display = "none";
  };

  modal.onclick = function (event) {
    if (event.target === modal) {
      modal.style.display = "none";
    }
  };

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      modal.style.display = "none";
    }
  });
};
