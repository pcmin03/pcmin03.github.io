(function() {
  window.pageLoad.then(function() {
    var themeToggle = document.querySelector('.js-theme-toggle');
    if (!themeToggle) {
      return;
    }

    var icon = themeToggle.querySelector('i');
    var themeKey = 'resume-theme';
    var darkModeClass = 'dark-mode';

    // Get saved theme or default to light
    function getTheme() {
      return localStorage.getItem(themeKey) || 'light';
    }

    // Set theme
    function setTheme(theme) {
      if (theme === 'dark') {
        document.documentElement.classList.add(darkModeClass);
        if (icon) {
          icon.classList.remove('fa-moon');
          icon.classList.add('fa-sun');
        }
        localStorage.setItem(themeKey, 'dark');
      } else {
        document.documentElement.classList.remove(darkModeClass);
        if (icon) {
          icon.classList.remove('fa-sun');
          icon.classList.add('fa-moon');
        }
        localStorage.setItem(themeKey, 'light');
      }
    }

    // Initialize theme on page load
    var currentTheme = getTheme();
    setTheme(currentTheme);

    // Toggle theme on button click
    themeToggle.addEventListener('click', function(event) {
      event.preventDefault();
      var currentTheme = getTheme();
      var newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      setTheme(newTheme);
    });
  });
})();

