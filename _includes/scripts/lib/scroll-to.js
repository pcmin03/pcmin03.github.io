(function() {
  var SOURCES = window.TEXT_VARIABLES.sources;
  window.Lazyload.js(SOURCES.jquery, function() {
    function scrollToAnchor(anchor, duration, callback) {
      var $root = this;
      var anchorStr = String(anchor || '');
      var targetEl = null;

      // Resolve element without relying on CSS selector parsing (ids can contain ':' '.' etc).
      if (anchorStr.charAt(0) === '#') {
        var rawId = anchorStr.slice(1);
        // href hashes can be encoded; decode safely.
        try {
          rawId = decodeURIComponent(rawId);
        } catch (e) {}
        targetEl = document.getElementById(rawId);
      }
      // Fallback: try jQuery (guard against invalid selectors).
      if (!targetEl) {
        try {
          targetEl = $(anchorStr)[0];
        } catch (e) {
          targetEl = null;
        }
      }
      if (!targetEl) {
        callback && callback();
        return;
      }

      var rootEl = $root && $root[0];
      var top;
      // If scrolling the document, use document coordinates.
      if (!rootEl || rootEl === document.body || rootEl === document.documentElement) {
        top = targetEl.getBoundingClientRect().top + (window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0);
      } else {
        // For a scroll container, compute position relative to the container.
        top = targetEl.getBoundingClientRect().top - rootEl.getBoundingClientRect().top + rootEl.scrollTop;
      }

      $root.animate({ scrollTop: top }, duration, function() {
        try {
          window.history.replaceState(null, '', window.location.href.split('#')[0] + anchorStr);
        } catch (e) {}
        callback && callback();
      });
    }
    $.fn.scrollToAnchor = scrollToAnchor;
  });
})();
