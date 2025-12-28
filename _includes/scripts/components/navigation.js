(function() {
  // Bind as early as possible; waiting for full `load` makes navigation feel broken
  // (especially on pages with many images/resources).
  function init() {
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
      // Find direct child <a> element (more compatible than querySelector with >)
      var trigger = null;
      // Try to find direct child <a> element
      for (var i = 0; i < item.children.length; i++) {
        if (item.children[i].tagName === 'A') {
          trigger = item.children[i];
          break;
        }
      }
      // Fallback to any <a> element if direct child not found
      if (!trigger) {
        trigger = item.querySelector('a');
      }
      if (!trigger) {
        return;
      }
      trigger.addEventListener('click', function(event) {
        if (window.innerWidth < 1024) {
          var dropdown = item.querySelector('.navigation__dropdown');
          if (dropdown) {
            var willOpen = !dropdown.classList.contains('navigation__dropdown--open');
            // Mobile UX:
            // - First tap opens the dropdown
            // - Second tap (when already open) follows the link
            if (willOpen) {
              event.preventDefault();
            }
            dropdown.classList.toggle('navigation__dropdown--open', willOpen);
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
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
