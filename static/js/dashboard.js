// =========================
// MODAL CONTROL
// =========================

function openModal(id) {
    document.getElementById(id).style.display = "flex";
}

function closeModal(id) {
    document.getElementById(id).style.display = "none";
}

window.onclick = function (event) {
    const modals = document.querySelectorAll(".modal");
    modals.forEach(modal => {
        if (event.target === modal) {
            modal.style.display = "none";
        }
    });
};

// =========================
// LIVE DASHBOARD STATS
// =========================

async function loadDashboardStats() {
    try {
        const res = await fetch("/api/dashboard_stats");
        const data = await res.json();

        if (data.error) return;

        document.getElementById("trees").innerText = data.trees;
        document.getElementById("ripe").innerText = data.ripe;
        document.getElementById("unripe").innerText = data.unripe;
        document.getElementById("ripening").innerText = data.ripening;
        document.getElementById("total").innerText = data.total;
        document.getElementById("accuracy").innerText = data.accuracy + "%";

    } catch (err) {
        console.log("Dashboard live update failed:", err);
    }
}

// Run on page load
loadDashboardStats();

// Auto-refresh every 10 seconds
setInterval(loadDashboardStats, 10000);

// =========================
// CLEAR DASHBOARD
// =========================

async function clearDashboard() {
    if (!confirm("Are you sure you want to clear all dashboard data?")) {
        return;
    }

    try {
        const res = await fetch("/clear_dashboard", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            }
        });

        const data = await res.json();

        if (data.status === "cleared") {
            alert("Dashboard cleared successfully!");
            loadDashboardStats(); // refresh instantly
        } else {
            alert("Failed to clear dashboard.");
        }

    } catch (error) {
        alert("Error clearing dashboard.");
        console.error(error);
    }
}

// =========================
// HELP (SEND EMAIL)
// =========================

async function sendHelp() {
    const name = document.querySelector("#helpModal input[type='text']").value;
    const email = document.querySelector("#helpModal input[type='email']").value;
    const message = document.querySelector("#helpModal textarea").value;

    if (!name || !email || !message) {
        alert("Please fill in all fields.");
        return;
    }

    try {
        const res = await fetch("/send_help", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, email, message })
        });

        const data = await res.json();

        if (data.status === "sent") {
            alert("Message sent successfully!");
            closeModal("helpModal");
        } else {
            alert("Failed to send message.");
        }

    } catch (error) {
        alert("Error sending message.");
        console.error(error);
    }
}

// =========================
// SUPPORT (MOBILE MONEY READY)
// =========================

async function sendSupport() {
    const phone = document.querySelector("#supportModal input[type='text']").value;
    const amount = document.querySelector("#supportModal input[type='number']").value;

    if (!phone || !amount) {
        alert("Please fill in all fields.");
        return;
    }

    try {
        const res = await fetch("/support", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ phone, amount })
        });

        const data = await res.json();

        alert(data.message || "Support request sent!");
        closeModal("supportModal");

    } catch (error) {
        alert("Error processing support request.");
        console.error(error);
    }
}

// =========================
// LOGOUT CONFIRMATION
// =========================

function confirmLogout() {
    const ok = confirm("Are you sure you want to logout?");
    if (ok) {
        window.location.href = "/logout";
    }
}