(function() {
  var SOURCES = window.TEXT_VARIABLES.sources;

  window.Lazyload.js(SOURCES.jquery, function() {
    var $menuToggle = $('.js-menu-toggle');
    var $navigation = $('.js-navigation');
    var $dropdownItems = $('.navigation__item--dropdown');

    // Toggle mobile menu
    $menuToggle.on('click', function() {
      $navigation.toggleClass('navigation--open');
      $(this).find('i').toggleClass('fa-bars fa-times');
    });

    // Toggle dropdown on mobile
    $dropdownItems.on('click', '> a', function(e) {
      if (window.innerWidth < 1024) {
        e.preventDefault();
        var $dropdown = $(this).siblings('.navigation__dropdown');
        $dropdown.toggleClass('navigation__dropdown--open');
      }
    });

    // Close menu when clicking outside
    $(document).on('click', function(e) {
      if (!$(e.target).closest('.header').length) {
        $navigation.removeClass('navigation--open');
        $menuToggle.find('i').removeClass('fa-times').addClass('fa-bars');
        $dropdownItems.find('.navigation__dropdown').removeClass('navigation__dropdown--open');
      }
    });

    // Close menu on window resize if it becomes desktop size
    $(window).on('resize', function() {
      if (window.innerWidth >= 1024) {
        $navigation.removeClass('navigation--open');
        $menuToggle.find('i').removeClass('fa-times').addClass('fa-bars');
        $dropdownItems.find('.navigation__dropdown').removeClass('navigation__dropdown--open');
      }
    });
  });
})();

