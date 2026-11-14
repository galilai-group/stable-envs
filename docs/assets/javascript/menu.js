/* The theme toggles data-state on whatever was clicked and looks for
   #menu-button when the dialog closes. The button is #burger-button, and
   clicks land on the icon inside it, so the mark stays an X after the
   menu closes. */
(function () {
  const button = document.getElementById("burger-button");
  const dialog = document.getElementById("bottom-sidebar");
  if (!button || !dialog) return;

  function setOpen(open) {
    button.dataset.state = open ? "open" : "closed";
    if (open) {
      if (!dialog.open) dialog.showModal();
    } else if (dialog.open) {
      dialog.close();
    }
  }

  button.onclick = function (event) {
    event.preventDefault();
    setOpen(button.dataset.state !== "open");
  };

  dialog.addEventListener("close", function () {
    button.dataset.state = "closed";
  });

  dialog.addEventListener("click", function (event) {
    if (event.target.closest("a")) setOpen(false);
  });
})();
