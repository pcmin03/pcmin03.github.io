(function() {
  window.pageLoad.then(function() {
    var menuToggle = document.querySelector('.js-menu-toggle');
    var navigation = document.querySelector('.js-navigation');
    var dropdownItems = document.querySelectorAll('.navigation__item--dropdown');

    if (!menuToggle || !navigation) {
      return;
    }

    var icon = menuToggle.querySelector('i');
    var toggleState = function(isOpen) {
      menuToggle.setAttribute('aria-expanded', isOpen);
      if (!icon) {
        return;
      }
      if (isOpen) {
        icon.classList.remove('fa-bars');
        icon.classList.add('fa-times');
      } else {
        icon.classList.remove('fa-times');
        icon.classList.add('fa-bars');
      }
    };

    menuToggle.addEventListener('click', function(event) {
      event.preventDefault();
      var willOpen = !navigation.classList.contains('navigation--open');
      navigation.classList.toggle('navigation--open', willOpen);
      toggleState(willOpen);
    });

    dropdownItems.forEach(function(item) {
      var trigger = item.querySelector('> a');
      if (!trigger) {
        return;
      }
      trigger.addEventListener('click', function(event) {
        if (window.innerWidth < 1024) {
          event.preventDefault();
          var dropdown = item.querySelector('.navigation__dropdown');
          if (dropdown) {
            dropdown.classList.toggle('navigation__dropdown--open');
          }
        }
      });
    });

    document.addEventListener('click', function(event) {
      if (!event.target.closest('.header')) {
        navigation.classList.remove('navigation--open');
        toggleState(false);
        dropdownItems.forEach(function(item) {
          var dropdown = item.querySelector('.navigation__dropdown');
          dropdown && dropdown.classList.remove('navigation__dropdown--open');
        });
      }
    });

    window.addEventListener('resize', function() {
      if (window.innerWidth >= 1024) {
        navigation.classList.remove('navigation--open');
        toggleState(false);
        dropdownItems.forEach(function(item) {
          var dropdown = item.querySelector('.navigation__dropdown');
          dropdown && dropdown.classList.remove('navigation__dropdown--open');
        });
      }
    });

    toggleState(false);
  });
})();
