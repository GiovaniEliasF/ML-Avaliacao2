
document.getElementById('predict-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const features = [
    parseFloat(document.getElementById('sl').value),
    parseFloat(document.getElementById('sw').value),
    parseFloat(document.getElementById('pl').value),
    parseFloat(document.getElementById('pw').value)
  ];

  const response = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({features})
  });
  const data = await response.json();
  document.getElementById('result').innerText = 'Previsão: ' + data.prediction;
});
