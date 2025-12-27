(function() {
  'use strict';
  
  // ===== UNIFIED NAVIGATION SYSTEM =====
  function initNavigation() {
    const NAV_OFFSET = 80;
    
    // Universal smooth scroll function
    function smoothScrollTo(targetId) {
      const target = document.querySelector(targetId);
      if (!target) {
        console.warn('Target not found:', targetId);
        return false;
      }
      
      const rect = target.getBoundingClientRect();
      const scrollTop = window.pageYOffset || document.documentElement.scrollTop || 0;
      const targetPosition = rect.top + scrollTop - NAV_OFFSET;
      
      window.scrollTo({
        top: Math.max(0, targetPosition),
        behavior: 'smooth'
      });
      return true;
    }
    
    // Direct event handlers for navigation links (more reliable)
    function setupNavLinks() {
      // Top navigation links
      const topNavLinks = document.querySelectorAll('.top-nav__link[href^="#"]');
      topNavLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          const href = this.getAttribute('href');
          smoothScrollTo(href);
        });
      });
      
      // Sidebar navigation links
      const sidebarNavLinks = document.querySelectorAll('.sidebar-nav__link[href^="#"]');
      sidebarNavLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          const href = this.getAttribute('href');
          smoothScrollTo(href);
        });
      });
      
      // Floating menu links
      const floatingNavLinks = document.querySelectorAll('.floating-nav__menu a[href^="#"]');
      floatingNavLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          const href = this.getAttribute('href');
          if (smoothScrollTo(href)) {
            // Close floating menu
            const floatingBtn = document.getElementById('floatingNavBtn');
            const floatingMenu = document.getElementById('floatingNavMenu');
            if (floatingBtn) floatingBtn.classList.remove('active');
            if (floatingMenu) floatingMenu.classList.remove('show');
          }
        });
      });
    }
    
    // Also keep event delegation as fallback
    document.addEventListener('click', function(e) {
      const link = e.target.closest('a[href^="#"]');
      if (!link) return;
      
      // Check if it's a navigation link
      const isNavLink = link.classList.contains('top-nav__link') || 
                       link.classList.contains('sidebar-nav__link') ||
                       link.closest('.floating-nav__menu');
      
      if (isNavLink) {
        e.preventDefault();
        e.stopPropagation();
        
        const href = link.getAttribute('href');
        if (smoothScrollTo(href)) {
          // Close floating menu if open
          const floatingMenu = link.closest('.floating-nav__menu');
          if (floatingMenu) {
            const floatingBtn = document.getElementById('floatingNavBtn');
            if (floatingBtn) {
              floatingBtn.classList.remove('active');
              floatingMenu.classList.remove('show');
            }
          }
        }
      }
    }, true);
    
    // Setup direct handlers
    setupNavLinks();
  }
  
  // ===== ACTIVE NAVIGATION STATE =====
  function initActiveNav() {
    const allNavLinks = document.querySelectorAll('.top-nav__link, .sidebar-nav__link, .floating-nav__menu a');
    const sections = [];
    
    // Build sections map
    allNavLinks.forEach(function(link) {
      const href = link.getAttribute('href');
      if (href && href.startsWith('#')) {
        const section = document.querySelector(href);
        if (section && !sections.find(s => s.id === href)) {
          sections.push({ id: href, element: section });
        }
      }
    });
    
    let scrollTimeout = null;
    
    function updateActiveNav() {
      const scrollPos = window.scrollY + 150;
      let currentSection = null;
      
      sections.forEach(function(section) {
        const top = section.element.offsetTop;
        const height = section.element.offsetHeight;
        if (scrollPos >= top && scrollPos < top + height) {
          currentSection = section;
        }
      });
      
      // Update all nav links
      allNavLinks.forEach(function(link) {
        link.classList.remove('active');
        if (currentSection && link.getAttribute('href') === currentSection.id) {
          link.classList.add('active');
        }
      });
      
      // Scroll top nav link into view if needed
      const activeTopLink = document.querySelector('.top-nav__link.active');
      if (activeTopLink) {
        const navList = activeTopLink.closest('.top-nav__list');
        if (navList) {
          const linkRect = activeTopLink.getBoundingClientRect();
          const navRect = navList.getBoundingClientRect();
          if (linkRect.left < navRect.left || linkRect.right > navRect.right) {
            const scrollLeft = activeTopLink.offsetLeft - (navList.offsetWidth / 2) + (activeTopLink.offsetWidth / 2);
            navList.scrollTo({ left: Math.max(0, scrollLeft), behavior: 'smooth' });
          }
        }
      }
    }
    
    function throttledUpdate() {
      if (scrollTimeout) return;
      scrollTimeout = requestAnimationFrame(function() {
        updateActiveNav();
        scrollTimeout = null;
      });
    }
    
    window.addEventListener('scroll', throttledUpdate, { passive: true });
    updateActiveNav();
  }
  
  // ===== FLOATING MENU TOGGLE =====
  function initFloatingMenu() {
    const floatingBtn = document.getElementById('floatingNavBtn');
    const floatingMenu = document.getElementById('floatingNavMenu');
    
    if (!floatingBtn || !floatingMenu) return;
    
    floatingBtn.addEventListener('click', function() {
      floatingBtn.classList.toggle('active');
      floatingMenu.classList.toggle('show');
    });
    
    document.addEventListener('click', function(e) {
      if (!floatingBtn.contains(e.target) && !floatingMenu.contains(e.target)) {
        floatingBtn.classList.remove('active');
        floatingMenu.classList.remove('show');
      }
    });
  }
  
  // ===== SCROLL ANIMATIONS =====
  function initScrollAnimations() {
    const observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('animated');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -50px 0px' });
    
    document.querySelectorAll('.scroll-animate').forEach(function(el) {
      observer.observe(el);
    });
    
    // Staggered delay for timeline items
    document.querySelectorAll('.timeline-item.scroll-animate').forEach(function(item, index) {
      item.style.transitionDelay = (index * 0.1) + 's';
    });
  }
  
  // ===== TAB SWITCHING =====
  function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(function(button) {
      button.addEventListener('click', function(e) {
        e.preventDefault();
        const targetTab = this.getAttribute('data-tab');
        
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(content => content.classList.remove('active'));
        
        this.classList.add('active');
        const targetContent = document.getElementById(targetTab + '-tab');
        if (targetContent) {
          targetContent.classList.add('active');
        }
      });
    });
  }
  
  // ===== INITIALIZE ALL =====
  function init() {
    console.log('Resume JS: Initializing...');
    initNavigation();
    initActiveNav();
    initFloatingMenu();
    initScrollAnimations();
    initTabs();
    console.log('Resume JS: Initialization complete');
  }
  
  // Ensure DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM already loaded, but wait a bit to ensure all elements are ready
    setTimeout(init, 100);
  }
})();

