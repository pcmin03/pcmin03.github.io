(function() {
  window.pageLoad.then(function() {
    var languageToggle = document.querySelector('.js-language-toggle');
    var languageToggleLabel = languageToggle ? languageToggle.querySelector('.js-language-toggle-label') : null;
    var themeToggle = document.querySelector('.js-theme-toggle');
    var languageKey = 'resume-language';
    var defaultLanguage = 'en';
    var translations = {
      en: {
        controls: {
          switchLanguage: 'Switch to Korean',
          themeToggle: 'Toggle morning/evening mode',
          floatingMenu: 'Navigation Menu'
        },
        home: {
          subtitle: 'Vision Research Engineer',
          quote: 'I build industrial and medical vision systems end to end: from Physical AI and domain adaptation to 3D detection, MLOps, and production deployment.',
          cards: {
            vision: {
              title: 'Physical AI · Spatial Intelligence',
              body: 'Build Physical AI, sim-to-real, and autonomous vision systems for manufacturing processes.',
              tag1: 'Physical AI',
              tag2: 'Spatial Intelligence',
              tag3: 'Vision Control'
            },
            product: {
              title: 'System Engineering · MLOps',
              body: 'Design end-to-end vision systems from data and model pipelines to packaging, deployment, and reliable operations.',
              tag1: 'System Engineering',
              tag2: 'MLOps',
              tag3: 'Deployment'
            },
            research: {
              title: 'Sim2Real · SSL · Domain Adaptation',
              body: 'Develop sim-to-real, self-supervised learning, and domain adaptation workflows that keep vision models stable under domain shift.',
              tag1: 'Sim2Real',
              tag2: 'Self-Supervised',
              tag3: 'Domain Adaptation'
            }
          },
          contact: {
            email: 'Email',
            cv: 'CV'
          },
          stats: {
            years: 'Years in CV & Industry',
            projects: 'Major projects',
            patents: 'Patents',
            papers: 'Vision papers'
          },
          quickNav: 'Quick Navigation',
          nav: {
            experience: 'Experience',
            education: 'Education',
            projects: 'Projects',
            seminar: 'Seminar',
            publications: 'Publications',
            skills: 'Expertise & Tools',
            awards: 'Awards',
            recentPosts: 'Recent Posts'
          },
          sections: {
            experience: 'Experience',
            education: 'Education',
            projects: 'Projects',
            seminar: 'Seminar & Teaching',
            publications: 'Publications & Research',
            patents: 'Patents',
            skills: 'Expertise & Tools',
            awards: 'Awards & Competitions',
            recentPosts: 'Recent Posts'
          },
          labels: {
            toolsUsed: 'Tools used:',
            thesis: 'Thesis:',
            domesticPatent: 'Domestic Patent:',
            awards: 'Awards:',
            onlineCourse: '[Online Course]',
            more: 'More'
          },
          seminar: {
            mentoringDesc: 'Provided guidance and mentorship to students in computer vision and machine learning projects.',
            gachonDesc: 'Taught machine learning fundamentals and applications to university students.'
          },
          tabs: {
            expertise: 'Expertise',
            tools: 'Tools & Technologies'
          },
          expertise: {
            item1: 'Object Detection & Segmentation',
            item2: '3D Vision · Depth · Pose',
            item3: 'Sensor Fusion & Robotics Perception',
            item4: 'Video Recognition & Tracking',
            item5: 'Self-Supervised Vision (DINO/MoCo/MAE)',
            item6: 'Embedding & Metric Learning',
            item7: 'Domain Adaptation & Continual Eval',
            item8: 'Large-Scale Data Pipelines',
            item9: 'Edge Deployment & MLOps'
          },
          awards: {
            codeLink: 'Code Link',
            gold: 'Gold',
            bronze: 'Bronze',
            excellence: 'Excellence'
          },
          experience: {
            posco: {
              title: 'AX Technology R&D Group · Spatial Intelligence Vision Engineer · POSCO DX',
              date: 'Pangyo, Korea · July 2024 - Present',
              bullet1Html: '<a href="https://kidd.co.kr/news/245144" target="_blank" rel="noopener" class="timeline-link">Isaac Sim·PLC·Sim2Real</a>: Built physical-AI training environments that mirrored steel-site conditions and iterated models against real deployment constraints.',
              bullet2Html: '<a href="https://www.digitaltoday.co.kr/news/articleView.html?idxno=500285&rf=toastPopup&utm_source=digitaltoday" target="_blank" rel="noopener" class="timeline-link">Dataiku·SDD·MLOps</a>: Advanced Digital Transformation (DX) from smart factory to intelligence factory by building SDD-driven image retrieval and automated MLOps pipelines for steel vision systems.',
              bullet3Html: '<a href="https://newsroom.posco.com/kr/%EC%9D%B8%ED%84%B0%EB%B7%B0-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%AC%EB%8A%94-%ED%98%81%EC%8B%A0-%EA%B8%B0%EC%88%A0-%EA%B0%9C%EB%B0%9C-2025-%ED%8F%AC%EC%8A%A4%EC%BD%94-%EA%B8%B0%EC%88%A0%EB%8C%80/" target="_blank" rel="noopener" class="timeline-link">Autonomous Steelmaking·L2 Integration·Vision AI</a>: Integrated L2 communications and control signals to apply vision-based anomaly detection in autonomous steelmaking operations.'
            },
            vuno: {
              title: 'AI Research Engineer · VUNO Inc.',
              date: 'Seoul, Korea · May 2021 - July 2024',
              bullet1Html: '<a href="https://openaccess.thecvf.com/content/ACCV2024/html/Cho_CNG-SFDA_Clean-and-Noisy_Region_Guided_Online-Offline_Source-Free_Domain_Adaptation_ACCV_2024_paper.html" target="_blank" rel="noopener" class="timeline-link">Test-Time Adaptation·Domain Adaptation·ACCV 2024</a>: Developed adaptation strategies for medical imaging domain shifts and contributed to the CNG-SFDA research line published at ACCV 2024.',
              bullet2Html: '<a href="https://tiger.grand-challenge.org/Home/" target="_blank" rel="noopener" class="timeline-link">Self-Supervised Learning·Universal Segmentation·TIGER Challenge</a>: Conducted segmentation-oriented SSL research, built pathology workflows around it, and applied the stack to achieve 1st place in the TIGER Challenge.',
              bullet3Html: '<a href="https://www.docdocdoc.co.kr/news/articleView.html?idxno=3013245" target="_blank" rel="noopener" class="timeline-link">Lung CT·3D Tiny Object Detection·MLOps</a>: Built and deployed lung nodule detection pipelines for 3D tiny-object modeling, with production-grade monitoring and release workflows in real service environments.',
              bullet4Html: '<a href="https://www.vuno.co/news/view/810" target="_blank" rel="noopener" class="timeline-link">PathQuant·End-to-End Delivery·Applied Engineering</a>: Owned the full cycle from data collection and model refinement to packaging and final deployment, demonstrating strong applied-engineer execution in medical imaging products.'
            }
          }
        }
      },
      ko: {
        controls: {
          switchLanguage: 'Switch to English',
          themeToggle: '아침/저녁 모드 전환',
          floatingMenu: '탐색 메뉴'
        },
        home: {
          subtitle: 'Vision Research Engineer',
          quote: 'Physical AI, 도메인 적응, 3D detection, MLOps, 운영 배포까지 산업과 의료 현장에서 동작하는 비전 시스템을 end-to-end로 구축합니다.',
          cards: {
            vision: {
              title: 'Physical AI · Spatial Intelligence',
              body: '제조공정을 위한 Physical AI, sim-to-real, autonomous vision 시스템을 구축합니다.',
              tag1: 'Physical AI',
              tag2: 'Spatial Intelligence',
              tag3: 'Vision Control'
            },
            product: {
              title: 'System Engineering · MLOps',
              body: '데이터와 모델 파이프라인부터 패키징, 배포, 운영 안정화까지 end-to-end 비전 시스템을 설계합니다.',
              tag1: 'System Engineering',
              tag2: 'MLOps',
              tag3: 'Deployment'
            },
            research: {
              title: 'Sim2Real · SSL · Domain Adaptation',
              body: 'sim-to-real, 자기지도학습, 도메인 적응 워크플로우를 통해 domain shift 환경에서도 안정적인 비전 모델을 만듭니다.',
              tag1: 'Sim2Real',
              tag2: '자기지도학습',
              tag3: '도메인 적응'
            }
          },
          contact: {
            email: '이메일',
            cv: '이력서'
          },
          stats: {
            years: 'CV 및 산업 경력',
            projects: '주요 프로젝트',
            patents: '특허',
            papers: '비전 논문'
          },
          quickNav: '빠른 이동',
          nav: {
            experience: '경력',
            education: '학력',
            projects: '프로젝트',
            seminar: '세미나',
            publications: '논문',
            skills: '전문성 & 도구',
            awards: '수상',
            recentPosts: '최근 글'
          },
          sections: {
            experience: '경력',
            education: '학력',
            projects: '프로젝트',
            seminar: '세미나 & 강의',
            publications: '논문 & 연구',
            patents: '특허',
            skills: '전문성 & 도구',
            awards: '수상 & 대회',
            recentPosts: '최근 글'
          },
          labels: {
            toolsUsed: '사용한 도구:',
            thesis: '학위논문:',
            domesticPatent: '국내 특허:',
            awards: '수상:',
            onlineCourse: '[온라인 강의]',
            more: '더보기'
          },
          seminar: {
            mentoringDesc: '컴퓨터 비전과 머신러닝 프로젝트를 수행하는 학생들에게 진로와 연구 방향을 멘토링했습니다.',
            gachonDesc: '대학생 대상 머신러닝 기초와 실제 응용 사례를 강의했습니다.'
          },
          tabs: {
            expertise: '전문성',
            tools: '도구 & 기술'
          },
          expertise: {
            item1: '객체 탐지 & 분할',
            item2: '3D 비전 · 깊이 · 자세 추정',
            item3: '센서 융합 & 로보틱스 퍼셉션',
            item4: '비디오 인식 & 추적',
            item5: '자기지도 비전 (DINO/MoCo/MAE)',
            item6: '임베딩 & 메트릭 러닝',
            item7: '도메인 적응 & 지속 평가',
            item8: '대규모 데이터 파이프라인',
            item9: '엣지 배포 & MLOps'
          },
          awards: {
            codeLink: '코드 링크',
            gold: '금상',
            bronze: '동상',
            excellence: '우수상'
          },
          experience: {
            posco: {
              title: 'AX 기술연구개발 그룹 · Spatial Intelligence Vision Engineer · POSCO DX',
              date: '판교, 대한민국 · 2024년 7월 - 현재',
              bullet1Html: '<a href="https://kidd.co.kr/news/245144" target="_blank" rel="noopener" class="timeline-link">Isaac Sim·PLC·Sim2Real</a>: 철강 현장과 유사한 물리 환경을 구성하고, 시뮬레이션 기반 학습 결과를 실제 적용 조건에 맞춰 검증·고도화했습니다.',
              bullet2Html: '<a href="https://www.digitaltoday.co.kr/news/articleView.html?idxno=500285&rf=toastPopup&utm_source=digitaltoday" target="_blank" rel="noopener" class="timeline-link">Dataiku·SDD·MLOps</a>: 스마트팩토리에서 인텔리전스 팩토리로 이어지는 Digital Transformation(DX) 맥락에서, SDD를 활용한 image retrieval 파이프라인과 자동화된 MLOps 체계를 구축했습니다.',
              bullet3Html: '<a href="https://newsroom.posco.com/kr/%EC%9D%B8%ED%84%B0%EB%B7%B0-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%AC%EB%8A%94-%ED%98%81%EC%8B%A0-%EA%B8%B0%EC%88%A0-%EA%B0%9C%EB%B0%9C-2025-%ED%8F%AC%EC%8A%A4%EC%BD%94-%EA%B8%B0%EC%88%A0%EB%8C%80/" target="_blank" rel="noopener" class="timeline-link">Autonomous 조업·L2 통신·Vision AI</a>: 2제강 autonomous 조업에 필요한 L2 레벨 통신과 제어 통합을 구축하고, vision 기반 anomaly detection을 적용했습니다.'
            },
            vuno: {
              title: 'AI 연구 엔지니어 · VUNO',
              date: '서울, 대한민국 · 2021년 5월 - 2024년 7월',
              bullet1Html: '<a href="https://openaccess.thecvf.com/content/ACCV2024/html/Cho_CNG-SFDA_Clean-and-Noisy_Region_Guided_Online-Offline_Source-Free_Domain_Adaptation_ACCV_2024_paper.html" target="_blank" rel="noopener" class="timeline-link">Test-Time Adaptation·Domain Adaptation·ACCV 2024</a>: 의료영상 도메인 변화에 대응하는 적응 기법을 연구하고, ACCV 2024의 CNG-SFDA 연구 라인으로 이어지는 기술을 개발했습니다.',
              bullet2Html: '<a href="https://tiger.grand-challenge.org/Home/" target="_blank" rel="noopener" class="timeline-link">Self-Supervised Learning·Universal Segmentation·TIGER Challenge</a>: segmentation 기반 SSL 연구와 병리 워크플로우를 구축해 TIGER Challenge에 적용했고, 1위를 달성했습니다.',
              bullet3Html: '<a href="https://www.docdocdoc.co.kr/news/articleView.html?idxno=3013245" target="_blank" rel="noopener" class="timeline-link">Lung CT·3D Tiny Object Detection·MLOps</a>: 폐 CT 결절 검출을 위한 3D tiny object detection 모델과 MLOps 파이프라인을 구축하고, 실제 서비스 환경에 운영·배포했습니다.',
              bullet4Html: '<a href="https://www.vuno.co/news/view/810" target="_blank" rel="noopener" class="timeline-link">PathQuant·End-to-End Delivery·Applied Engineering</a>: 데이터 수집부터 모델 고도화, 최종 패키징과 배포까지 전 과정을 단독으로 수행하며 applied engineer 역량을 입증했습니다.'
            }
          }
        }
      }
    };

    function normalizeLanguage(language) {
      return String(language || '').toLowerCase().indexOf('ko') === 0 ? 'ko' : 'en';
    }

    function getInitialLanguage() {
      var savedLanguage = null;
      try {
        savedLanguage = localStorage.getItem(languageKey);
      } catch (error) {
        savedLanguage = null;
      }
      if (savedLanguage) {
        return normalizeLanguage(savedLanguage);
      }
      if (document.documentElement && document.documentElement.lang) {
        return normalizeLanguage(document.documentElement.lang);
      }
      if (window.navigator) {
        return normalizeLanguage(window.navigator.language || window.navigator.userLanguage);
      }
      return defaultLanguage;
    }

    function getTranslation(language, key) {
      var path = String(key || '').split('.');
      var value = translations[language];
      var i;

      for (i = 0; i < path.length; i += 1) {
        if (!value || typeof value !== 'object') {
          value = null;
          break;
        }
        value = value[path[i]];
      }

      if (value == null && language !== defaultLanguage) {
        return getTranslation(defaultLanguage, key);
      }
      return value;
    }

    function updateI18nContent(language) {
      var textNodes = document.querySelectorAll('[data-i18n]');
      var htmlNodes = document.querySelectorAll('[data-i18n-html]');
      var titleNodes = document.querySelectorAll('[data-i18n-title]');
      var ariaNodes = document.querySelectorAll('[data-i18n-aria-label]');

      Array.prototype.forEach.call(textNodes, function(node) {
        var key = node.getAttribute('data-i18n');
        var value = getTranslation(language, key);
        if (typeof value === 'string') {
          node.textContent = value;
        }
      });

      Array.prototype.forEach.call(htmlNodes, function(node) {
        var key = node.getAttribute('data-i18n-html');
        var value = getTranslation(language, key);
        if (typeof value === 'string') {
          node.innerHTML = value;
        }
      });

      Array.prototype.forEach.call(titleNodes, function(node) {
        var key = node.getAttribute('data-i18n-title');
        var value = getTranslation(language, key);
        if (typeof value === 'string') {
          node.setAttribute('title', value);
        }
      });

      Array.prototype.forEach.call(ariaNodes, function(node) {
        var key = node.getAttribute('data-i18n-aria-label');
        var value = getTranslation(language, key);
        if (typeof value === 'string') {
          node.setAttribute('aria-label', value);
        }
      });
    }

    function updateNavigationLabels(language) {
      var navLabels = document.querySelectorAll('.js-language-nav-text');
      Array.prototype.forEach.call(navLabels, function(node) {
        var value = node.getAttribute('data-locale-' + language) || node.getAttribute('data-locale-en');
        if (value) {
          node.textContent = value;
        }
      });
    }

    function updateLanguageToggle(language) {
      if (!languageToggle) {
        return;
      }
      var nextLanguage = language === 'ko' ? 'EN' : 'KO';
      var label = language === 'ko' ? 'Switch to English' : 'Switch to Korean';
      if (languageToggleLabel) {
        languageToggleLabel.textContent = nextLanguage;
      }
      languageToggle.setAttribute('aria-label', label);
      languageToggle.setAttribute('title', label);
    }

    function updateThemeToggle(language) {
      if (!themeToggle) {
        return;
      }
      var label = getTranslation(language, 'controls.themeToggle');
      if (label) {
        themeToggle.setAttribute('aria-label', label);
        themeToggle.setAttribute('title', label);
      }
    }

    function persistLanguage(language) {
      try {
        localStorage.setItem(languageKey, language);
      } catch (error) {
        return;
      }
    }

    function applyLanguage(language) {
      var normalizedLanguage = normalizeLanguage(language);
      document.documentElement.lang = normalizedLanguage === 'ko' ? 'ko-KR' : 'en';
      document.documentElement.setAttribute('data-language', normalizedLanguage);
      persistLanguage(normalizedLanguage);
      updateLanguageToggle(normalizedLanguage);
      updateThemeToggle(normalizedLanguage);
      updateNavigationLabels(normalizedLanguage);
      updateI18nContent(normalizedLanguage);

      if (typeof window.CustomEvent === 'function') {
        document.dispatchEvent(new CustomEvent('resume:languagechange', {
          detail: {
            language: normalizedLanguage
          }
        }));
      }
    }

    if (languageToggle) {
      languageToggle.addEventListener('click', function(event) {
        event.preventDefault();
        var currentLanguage = normalizeLanguage(document.documentElement.getAttribute('data-language'));
        applyLanguage(currentLanguage === 'ko' ? 'en' : 'ko');
      });
    }

    applyLanguage(getInitialLanguage());
  });
})();
