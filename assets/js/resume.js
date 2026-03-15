(function() {
  'use strict';
  
  // ===== UNIFIED NAVIGATION SYSTEM =====
  function initNavigation() {
    const NAV_OFFSET = 96;
    
    // Universal smooth scroll function
    function smoothScrollTo(targetId) {
      const targetHash = String(targetId || '');
      let target = null;

      // Prefer id lookup (safe even if id contains special chars)
      if (targetHash.charAt(0) === '#') {
        let id = targetHash.slice(1);
        try { id = decodeURIComponent(id); } catch (_) {}
        target = document.getElementById(id);
      }
      // Fallback to querySelector for non-id selectors
      if (!target) {
        try {
          target = document.querySelector(targetHash);
        } catch (_) {
          target = null;
        }
      }
      if (!target) {
        return false;
      }

      // IMPORTANT: Avoid scrollIntoView() because it may also scroll horizontally on mobile/tablet,
      // which can make the whole page "shift" left/right. We scroll vertically only.
      function getScrollContainer(element) {
        // Walk up the DOM to find the nearest scrollable ancestor.
        let current = element;
        while (current && current !== document.body && current !== document.documentElement) {
          const style = window.getComputedStyle(current);
          const overflowY = style.overflowY || '';
          const overflow = style.overflow || '';
          const overflowAllowsScroll = /auto|scroll/i.test(overflowY) || /auto|scroll/i.test(overflow);
          const canScrollY = current.scrollHeight > current.clientHeight + 2;
          if (overflowAllowsScroll && canScrollY) return current;
          current = current.parentElement;
        }

        const main = document.querySelector('.js-page-main') || document.querySelector('.page__main');
        if (main) {
          const style = window.getComputedStyle(main);
          const overflowY = style.overflowY || '';
          const overflow = style.overflow || '';
          const overflowAllowsScroll = /auto|scroll/i.test(overflowY) || /auto|scroll/i.test(overflow);
          const canScrollY = main.scrollHeight > main.clientHeight + 2;
          if (overflowAllowsScroll && canScrollY) return main;
        }

        if (document.scrollingElement && document.scrollingElement.scrollHeight > document.scrollingElement.clientHeight + 2) {
          return window;
        }
        return window;
      }

      const scroller = getScrollContainer(target);
      const behavior = 'smooth';

      const startTop = scroller === window
        ? ((document.scrollingElement && document.scrollingElement.scrollTop) ||
          window.pageYOffset || document.documentElement.scrollTop || 0)
        : scroller.scrollTop;

      if (scroller === window) {
        const rect = target.getBoundingClientRect();
        const scrollTop = (document.scrollingElement && document.scrollingElement.scrollTop) ||
          window.pageYOffset || document.documentElement.scrollTop || 0;
        const targetTop = rect.top + scrollTop - NAV_OFFSET;
        window.scrollTo({ top: Math.max(0, targetTop), behavior });
        // Hard clamp horizontal scroll to avoid "shift"
        if (document.documentElement) document.documentElement.scrollLeft = 0;
        if (document.body) document.body.scrollLeft = 0;
      } else {
        const scrollerRect = scroller.getBoundingClientRect();
        const targetRect = target.getBoundingClientRect();
        const targetTop = (targetRect.top - scrollerRect.top) + scroller.scrollTop - NAV_OFFSET;
        if (typeof scroller.scrollTo === 'function') {
          scroller.scrollTo({ top: Math.max(0, targetTop), behavior });
        } else {
          scroller.scrollTop = Math.max(0, targetTop);
        }
        scroller.scrollLeft = 0;
      }

      // Keep URL hash in sync (so refresh/share works)
      try {
        window.history.replaceState(null, '', window.location.href.split('#')[0] + targetHash);
      } catch (_) {
        // ignore
      }

      // Fallback: if scroll didn't move, try native behavior.
      setTimeout(function() {
        const endTop = scroller === window
          ? ((document.scrollingElement && document.scrollingElement.scrollTop) ||
            window.pageYOffset || document.documentElement.scrollTop || 0)
          : scroller.scrollTop;
        if (Math.abs(endTop - startTop) < 2) {
          try {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
          } catch (_) {
            window.location.hash = targetHash;
          }
        }
      }, 60);
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
          if (!smoothScrollTo(href)) {
            window.location.hash = href;
          }
        });
      });
      
      // Sidebar navigation links
      const sidebarNavLinks = document.querySelectorAll('.sidebar-nav__link[href^="#"]');
      sidebarNavLinks.forEach(function(link) {
        link.addEventListener('click', function(e) {
          e.preventDefault();
          e.stopPropagation();
          const href = this.getAttribute('href');
          if (!smoothScrollTo(href)) {
            window.location.hash = href;
          }
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
          } else {
            window.location.hash = href;
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
        } else {
          window.location.hash = href;
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

  // ===== TOP NAV HORIZONTAL DRAG SCROLL (DESKTOP + TOUCH) =====
  function initTopNavDragScroll() {
    const navList = document.querySelector('.top-nav__list');
    if (!navList) return;

    // If it doesn't overflow, no need to attach handlers.
    const hasOverflow = () => navList.scrollWidth > navList.clientWidth + 2;
    if (!hasOverflow()) return;

    let isPointerDown = false;
    let startX = 0;
    let startScrollLeft = 0;
    let didDrag = false;
    let movedPx = 0;
    let hasPointerCapture = false;

    // Make it feel draggable on desktop.
    navList.style.cursor = 'grab';

    navList.addEventListener('pointerdown', function(e) {
      // Only primary button for mouse; allow touch/pen.
      if (e.pointerType === 'mouse' && e.button !== 0) return;
      isPointerDown = true;
      didDrag = false;
      movedPx = 0;
      hasPointerCapture = false;
      startX = e.clientX;
      startScrollLeft = navList.scrollLeft;
      navList.style.cursor = 'grabbing';
    });

    navList.addEventListener('pointermove', function(e) {
      if (!isPointerDown) return;
      const dx = e.clientX - startX;
      movedPx = Math.max(movedPx, Math.abs(dx));

      // Treat tiny jitter as a click, not a drag.
      // (On trackpads/touch, pointermove often fires with a few px movement.)
      if (movedPx < 8) {
        return;
      }

      // Only capture the pointer once we are sure it's a drag.
      // Capturing on pointerdown can retarget the subsequent click away from <a>,
      // making navigation links appear "dead" on some browsers.
      if (!hasPointerCapture && navList.setPointerCapture) {
        try {
          navList.setPointerCapture(e.pointerId);
          hasPointerCapture = true;
        } catch (_) {
          hasPointerCapture = false;
        }
      }

      didDrag = true;
      navList.scrollLeft = startScrollLeft - dx;
      // Prevent the page from selecting text / scrolling vertically while dragging.
      if (e.cancelable) e.preventDefault();
    }, { passive: false });

    function endDrag(e) {
      // Also consider actual scroll delta (more reliable than pointer jitter alone).
      if (Math.abs(navList.scrollLeft - startScrollLeft) > 6) {
        didDrag = true;
      }
      if (hasPointerCapture && navList.releasePointerCapture && e && e.pointerId !== undefined) {
        try {
          navList.releasePointerCapture(e.pointerId);
        } catch (_) {}
      }
      hasPointerCapture = false;
      isPointerDown = false;
      navList.style.cursor = 'grab';
    }

    navList.addEventListener('pointerup', endDrag);
    navList.addEventListener('pointercancel', endDrag);
    navList.addEventListener('pointerleave', endDrag);

    // If the user dragged, swallow the click so links don't accidentally open.
    navList.addEventListener('click', function(e) {
      if (didDrag) {
        e.preventDefault();
        e.stopPropagation();
        didDrag = false;
      }
    }, true);
  }

  // ===== TIMELINE LINK PREVIEW =====
  function initLinkPreviews() {
    const selector = '.timeline-link[href]';
    let activeLink = null;
    let tooltip = document.querySelector('.js-link-preview');
    const previewMetadata = {
      'https://kidd.co.kr/news/245144': {
        title: '포스코DX, 비전 AI로 철강 원료 항만 하역 무인화',
        image: 'https://pimg3.daara.co.kr/kidd/photo/2026/03/04/1772607405_93.jpg',
        domain: 'kidd.co.kr'
      },
      'https://www.digitaltoday.co.kr/news/articleView.html?idxno=500285&rf=toastPopup&utm_source=digitaltoday': {
        title: '포스코 그룹, 메타버스 기반 마케팅 디지털 전환 추진',
        image: 'https://cdn.digitaltoday.co.kr/news/photo/202401/500285_465879_5231.jpg',
        domain: 'digitaltoday.co.kr'
      },
      'https://fastcampus.co.kr/data_online_medicalai': {
        title: '딥러닝을 활용한 의료 영상 처리 & 모델 개발',
        image: '/assets/images/medical-ai-course.png',
        domain: 'fastcampus.co.kr'
      },
      'https://www.docdocdoc.co.kr/news/articleView.html?idxno=3013245': {
        title: '뷰노 "뷰노메드 흉부 CT AI, 일본서 보험급여 인정"',
        image: 'https://cdn.docdocdoc.co.kr/news/thumbnail/202401/3013245_3015112_2119_v150.jpg',
        domain: 'www.docdocdoc.co.kr'
      },
      'https://openaccess.thecvf.com/content/ACCV2024/html/Cho_CNG-SFDA_Clean-and-Noisy_Region_Guided_Online-Offline_Source-Free_Domain_Adaptation_ACCV_2024_paper.html': {
        title: 'CNG-SFDA: Clean-and-Noisy Region Guided Online-Offline Source-Free Domain Adaptation',
        image: '/assets/images/publications/cng-sfda.png',
        domain: 'openaccess.thecvf.com'
      },
      'https://tiger.grand-challenge.org/Home/': {
        title: 'TIGER Challenge',
        image: '/assets/images/posts/tiger-challenge/segmentation-patch-selection.png',
        domain: 'tiger.grand-challenge.org'
      },
      'https://www.vuno.co/news/view/810': {
        title: '뷰노, 디지털 병리 분석 AI 솔루션 뷰노메드 패스퀀트 식약처 인증 획득',
        image: 'https://www.vuno.co/data/files/2021-06/5be1819c1a924312dc242203946d8c06.jpg',
        domain: 'www.vuno.co'
      },
      'https://newsroom.posco.com/kr/%EC%9D%B8%ED%84%B0%EB%B7%B0-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%AC%EB%8A%94-%ED%98%81%EC%8B%A0-%EA%B8%B0%EC%88%A0-%EA%B0%9C%EB%B0%9C-2025-%ED%8F%AC%EC%8A%A4%EC%BD%94-%EA%B8%B0%EC%88%A0%EB%8C%80/': {
        title: '[인터뷰] 미래를 여는 혁신 기술 개발! 2025 포스코 기술대상 수상자들을 만나다',
        image: '',
        domain: 'newsroom.posco.com'
      }
    };

    if (!tooltip) {
      tooltip = document.createElement('div');
      tooltip.className = 'link-preview js-link-preview';
      tooltip.setAttribute('aria-hidden', 'true');
      tooltip.innerHTML = '' +
        '<img class="link-preview__image js-link-preview-image" alt=\"Link preview image\">' +
        '<div class="link-preview__body">' +
        '<span class="link-preview__label">Link Preview</span>' +
        '<span class="link-preview__title js-link-preview-title"></span>' +
        '<span class="link-preview__domain js-link-preview-domain"></span>' +
        '<span class="link-preview__url js-link-preview-url"></span>' +
        '</div>';
      document.body.appendChild(tooltip);
    }

    const tooltipUrl = tooltip.querySelector('.js-link-preview-url');
    const tooltipTitle = tooltip.querySelector('.js-link-preview-title');
    const tooltipDomain = tooltip.querySelector('.js-link-preview-domain');
    const tooltipImage = tooltip.querySelector('.js-link-preview-image');

    function getLinkPreviewText(link) {
      const href = link.getAttribute('href') || '';
      if (!href || href.indexOf('mailto:') === 0 || href.indexOf('tel:') === 0) {
        return '';
      }
      return href;
    }

    function getPreviewMetadata(href) {
      const normalizedHref = String(href || '');
      if (previewMetadata[normalizedHref]) {
        return previewMetadata[normalizedHref];
      }
      try {
        const url = new URL(normalizedHref, window.location.origin);
        return {
          title: '',
          image: '',
          domain: url.hostname
        };
      } catch (_) {
        return {
          title: '',
          image: '',
          domain: ''
        };
      }
    }

    function positionTooltip(event) {
      if (!activeLink) return;
      const offset = 18;
      const tooltipRect = tooltip.getBoundingClientRect();
      const maxLeft = Math.max(12, window.innerWidth - tooltipRect.width - 12);
      const maxTop = Math.max(12, window.innerHeight - tooltipRect.height - 12);
      const left = Math.min(maxLeft, Math.max(12, event.clientX + offset));
      const top = Math.min(maxTop, Math.max(12, event.clientY + offset));

      tooltip.style.left = left + 'px';
      tooltip.style.top = top + 'px';
    }

    function showTooltip(link, event) {
      const previewText = getLinkPreviewText(link);
      const metadata = getPreviewMetadata(previewText);
      if (!previewText) {
        hideTooltip();
        return;
      }

      activeLink = link;
      tooltipUrl.textContent = previewText;
      tooltipTitle.textContent = metadata.title || '';
      tooltipDomain.textContent = metadata.domain || '';
      tooltip.classList.toggle('link-preview--rich', Boolean(metadata.title || metadata.image));
      if (metadata.image) {
        tooltipImage.src = metadata.image;
        tooltipImage.alt = metadata.title || 'Link preview image';
      } else {
        tooltipImage.removeAttribute('src');
        tooltipImage.alt = 'Link preview image';
      }
      tooltip.classList.add('link-preview--visible');
      tooltip.setAttribute('aria-hidden', 'false');
      if (event) {
        positionTooltip(event);
      }
    }

    function hideTooltip() {
      activeLink = null;
      tooltip.classList.remove('link-preview--visible');
      tooltip.classList.remove('link-preview--rich');
      tooltip.setAttribute('aria-hidden', 'true');
    }

    document.addEventListener('mouseover', function(event) {
      const link = event.target.closest(selector);
      if (!link) {
        return;
      }
      showTooltip(link, event);
    });

    document.addEventListener('mousemove', function(event) {
      if (activeLink) {
        positionTooltip(event);
      }
    });

    document.addEventListener('mouseout', function(event) {
      if (!activeLink) return;
      const currentLink = event.target.closest(selector);
      if (!currentLink || currentLink !== activeLink) return;
      if (event.relatedTarget && activeLink.contains(event.relatedTarget)) return;
      hideTooltip();
    });

    document.addEventListener('focusin', function(event) {
      const link = event.target.closest(selector);
      if (!link) return;
      const rect = link.getBoundingClientRect();
      showTooltip(link, {
        clientX: rect.left,
        clientY: rect.bottom
      });
    });

    document.addEventListener('focusout', function(event) {
      const link = event.target.closest(selector);
      if (link && link === activeLink) {
        hideTooltip();
      }
    });
  }
  
  // ===== INITIALIZE ALL =====
  function init() {
    initNavigation();
    initActiveNav();
    initFloatingMenu();
    initScrollAnimations();
    initTabs();
    initTopNavDragScroll();
    initLinkPreviews();
  }
  
  // Ensure DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    // DOM already loaded, but wait a bit to ensure all elements are ready
    setTimeout(init, 100);
  }
})();
