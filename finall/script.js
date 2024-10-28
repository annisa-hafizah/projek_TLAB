function toggleGate() {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", "/toggle-gate", true);
    xhr.onload = function() {
        if (xhr.status == 200) {
            // Update the gate status on the page dynamically
            var response = xhr.responseText;
            document.getElementById("gate-state").innerText = response;
            document.getElementById("toggle-switch").checked = response.includes('OPEN');
        }
    };
    xhr.send();
}
