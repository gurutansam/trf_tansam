setInterval(() => {
    fetch("/progress/status")
        .then(r => r.json())
        .then(data => {
            document.getElementById("statusText").innerText = data.message;
        });
}, 2000);
