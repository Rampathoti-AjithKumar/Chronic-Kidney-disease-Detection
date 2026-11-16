
document.addEventListener("DOMContentLoaded", () => {

    const email = document.getElementById("email");
    const remember = document.getElementById("remember");

    if (!email || !remember) return;

    // Restore saved email
    const savedEmail = localStorage.getItem("ckd_email");
    const savedRemember = localStorage.getItem("ckd_remember");

    if (savedRemember === "true" && savedEmail) {
        email.value = savedEmail;
        remember.checked = true;
    }

    // Save email preference on form submission
    const form = email.closest("form");

    form.addEventListener("submit", () => {

        if (remember.checked) {
            localStorage.setItem("ckd_email", email.value.trim());
            localStorage.setItem("ckd_remember", "true");
        } else {
            localStorage.removeItem("ckd_email");
            localStorage.removeItem("ckd_remember");
        }

    });

});
