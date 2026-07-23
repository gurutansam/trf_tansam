document.addEventListener("DOMContentLoaded", function () {

    const phaseSelect = document.getElementById("phaseSelect");
    const phase1Box = document.getElementById("locationPhase1Box");
    const phase2Box = document.getElementById("locationPhase2Box");
    const location1 = document.getElementById("locationPhase1");
    const location2 = document.getElementById("locationPhase2");
    const hiddenLocation = document.getElementById("selectedLocation");

    phaseSelect.addEventListener("change", function () {
        if (this.value === "Phase 1") {
            phase1Box.style.display = "block";
            phase2Box.style.display = "none";
        } else if (this.value === "Phase 2") {
            phase1Box.style.display = "none";
            phase2Box.style.display = "block";
        }
    });

    location1.addEventListener("change", function () {
        hiddenLocation.value = this.value;
    });

    location2.addEventListener("change", function () {
        hiddenLocation.value = this.value;
    });

});
