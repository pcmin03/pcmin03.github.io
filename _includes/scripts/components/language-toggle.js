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
          quote: 'Building Physical AI for the real industrial world: spatial intelligence, vision foundation models, and sim-to-real systems that move factories from smart to intelligent.',
          cards: {
            physical: {
              title: 'Physical AI · Autonomous Operations',
              body: 'Perception-to-action AI that closes the loop with L2/PLC control, from autonomous steelmaking to unmanned material handling.'
            },
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
            },
            spatial: {
              title: 'Spatial Intelligence · Digital Twin',
              body: 'LiDAR, ToF, and RGB fusion for 3D scene understanding, trained in Isaac Sim digital twins and transferred with Sim2Real.'
            },
            vfm: {
              title: 'Vision Foundation Models',
              body: 'CLIP prompt tuning, PEFT, and self-supervised learning that adapt large models to new domains with minimal labels.'
            }
          },
          contact: {
            email: 'Email',
            cv: 'CV'
          },
          stats: {
            award: 'POSCO DX AI Tech Award 2025',
            years: 'Years in CV & Industry',
            projects: 'Major projects',
            patents: 'Patents',
            papers: 'Vision papers'
          },
          kickers: {
            highlight: 'Industrial AX',
            about: 'About me',
            experience: 'Experience',
            publications: 'My work',
            projects: 'Selected projects',
            education: 'Education',
            seminar: 'Teaching',
            skills: 'Toolbox',
            awards: 'Achievement',
            posts: 'From the blog'
          },
          about: {
            p1Html: 'I am a Spatial Intelligence Vision Engineer in the AX Technology R&D Group at POSCO DX. I design <strong>Physical AI</strong> systems that see the plant in 3D, learn in digital twins, and act through real control loops: autonomous operations, vision control, and AI-driven quality.',
            p2: 'On the research side I work on vision foundation models, parameter-efficient adaptation, and domain generalization. Before POSCO DX I spent three years at VUNO taking vision models from research to certified products.'
          },
          education: {
            unistScholarship: 'Academic excellence scholarship, 4 semesters (2019-2021)',
            injeScholarship: 'Academic excellence scholarship, 8 semesters (2014-2018)',
            injeAwards: 'Gold Prize, Inje Creative Comprehensive Design Contest · Bronze Prize, 2017 Engineering Festival'
          },
          tags: {
            physical: 'Physical AI',
            spatial: 'Spatial Intelligence',
            vfm: 'Vision Foundation Models',
            twin: 'Sim2Real · Digital Twin'
          },
          journey: {
            l1: 'Smart Factory',
            t1: 'Vision Inspection',
            d1: 'Steel defect detection and image retrieval turn camera streams into real-time quality signals.',
            l2: 'AI Factory',
            t2: 'MLOps · Continuous Training',
            d2: 'Data-centric pipelines and continuous training keep models current across lines and facilities.',
            l3: 'Intelligent Factory',
            t3: 'Physical AI · Autonomous Operations',
            d3: 'Spatial AI wired into L2/PLC lets equipment act on its own, from autonomous steelmaking to unmanned port cranes.'
          },
          projects: {
            craneTitle: 'Autonomous Port Unloading',
            steelmakingTitle: 'Autonomous Steelmaking'
          },
          quickNav: 'Quick Navigation',
          nav: {
            highlight: 'AX Roadmap',
            about: 'About',
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
            highlight: 'From Smart Factory to AI-native Intelligent Factory',
            experienceNote: 'Vision AI for industrial sites, from research prototypes to systems running on the plant floor.',
            projectsNote: 'Industrial AX projects first, then earlier vision research and products. Click a card for details.',
            educationNote: 'A compact academic timeline: degree, thesis, and research direction.',
            experience: 'Experience',
            education: 'Education',
            projects: 'Projects',
            seminar: 'Seminar & Teaching',
            publications: 'Publications & Research',
            patents: 'Patents',
            skills: 'Expertise & Tools',
            awards: 'Awards & Competitions',
            recentPosts: 'Recent Posts',
            highlightNote: 'How my work moves a steel plant from collecting data, to learning from it, to acting on its own.'
          },
          labels: {
            blog: 'Blog',
            toolsUsed: 'Tools',
            thesis: 'Thesis',
            domesticPatent: 'Domestic Patent:',
            awards: 'Awards:',
            onlineCourse: 'Online Course',
            more: 'More posts'
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
            item1: 'Vision Foundation Models · PEFT',
            item2: 'Vision-Language (CLIP) Adaptation',
            item3: 'Spatial Intelligence · 3D Perception',
            item4: 'Embodied Perception · Sensor Fusion',
            item5: 'Sim2Real · Digital Twin (Isaac Sim)',
            item6: 'Self-Supervised Learning (DINO/MAE)',
            item7: 'Domain Generalization · Test-Time Adaptation',
            item8: 'Data-centric AI · Large-scale Pipelines',
            item9: 'Edge AI Deployment · MLOps'
          },
          awards: {
            posco: 'AI Technology Excellence Award, POSCO DX (Sep 2025)',
            poscoNote: 'Steelmaking automation with vision AI',
            competitionsTitle: 'Awards & Competitions',
            academicTitle: 'Academic Awards',
            scholarships: 'Academic excellence scholarships for 12 semesters across UNIST and Inje University',
            codeLink: 'Code Link',
            gold: 'Gold',
            bronze: 'Bronze',
            excellence: 'Excellence'
          },
          experience: {
            posco: {
              title: 'Spatial Intelligence Vision Engineer',
              date: 'Pangyo, Korea · July 2024 - Present',
              bullet1Html: 'Physical AI·Digital Twin·Sim2Real: Built Isaac Sim digital twins of steel-site conditions and transferred perception models to the real plant.',
              bullet2Html: '<a href="https://kidd.co.kr/news/245144" target="_blank" rel="noopener" class="timeline-link">Spatial Intelligence·3D Sensor Fusion</a>: Fused LiDAR, ToF, and RGB for 3D scene understanding, applied to vision control such as unmanned port unloading.',
              bullet3Html: '<a href="https://newsroom.posco.com/kr/%EC%9D%B8%ED%84%B0%EB%B7%B0-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%AC%EB%8A%94-%ED%98%81%EC%8B%A0-%EA%B8%B0%EC%88%A0-%EA%B0%9C%EB%B0%9C-2025-%ED%8F%AC%EC%8A%A4%EC%BD%94-%EA%B8%B0%EC%88%A0%EB%8C%80/" target="_blank" rel="noopener" class="timeline-link">Autonomous Operations·L2/PLC</a>: Connected vision AI to L2 communications and control signals for autonomous steelmaking.',
              bullet4Html: 'Vision Foundation Models·PEFT·WACV 2027: Co-first-authored CAPT, which adapts a fully frozen CLIP with at most 1.35M trainable parameters and sets a new state of the art on DomainNet, Sketchy, and TU-Berlin.',
              bullet5Html: '<a href="https://www.digitaltoday.co.kr/news/articleView.html?idxno=500285&rf=toastPopup&utm_source=digitaltoday" target="_blank" rel="noopener" class="timeline-link">Steel Defect Detection·MLOps</a>: Built SDD models, image retrieval, and MLOps Level 1 with continuous training in Dataiku.'
            },
            vuno: {
              title: 'AI Research Engineer',
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
          quote: '현실 산업 현장을 위한 Physical AI를 만듭니다. Spatial Intelligence, Vision Foundation Model, Sim2Real로 스마트 팩토리를 인텔리전트 팩토리로 바꿉니다.',
          cards: {
            physical: {
              title: 'Physical AI · 자율 조업',
              body: '인지부터 행동까지 L2/PLC 제어와 연결된 AI로, 자율 조업부터 무인 물류·하역까지 적용합니다.'
            },
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
            },
            spatial: {
              title: 'Spatial Intelligence · 디지털 트윈',
              body: 'LiDAR·ToF·RGB 융합으로 3D 장면을 이해하고, Isaac Sim 디지털 트윈에서 학습해 Sim2Real로 옮깁니다.'
            },
            vfm: {
              title: 'Vision Foundation Model',
              body: 'CLIP 프롬프트 튜닝, PEFT, 자기지도학습으로 적은 라벨만으로 대형 모델을 새 도메인에 적응시킵니다.'
            }
          },
          contact: {
            email: '이메일',
            cv: '이력서'
          },
          stats: {
            award: '포스코DX AI 기술 우수상 2025',
            years: 'CV 및 산업 경력',
            projects: '주요 프로젝트',
            patents: '특허',
            papers: '비전 논문'
          },
          kickers: {
            highlight: 'Industrial AX',
            about: '소개',
            experience: '경력',
            publications: '연구 성과',
            projects: '주요 프로젝트',
            education: '학력',
            seminar: '강의',
            skills: '기술 스택',
            awards: '수상',
            posts: '블로그'
          },
          about: {
            p1Html: '포스코DX AX기술연구그룹의 Spatial Intelligence Vision Engineer입니다. 공장을 3D로 이해하고, 디지털 트윈에서 학습하며, 실제 제어 루프로 움직이는 <strong>Physical AI</strong> 시스템을 설계합니다. 자율 조업, Vision Control, AI 품질 관리가 주요 적용 분야입니다.',
            p2: '연구로는 Vision Foundation Model, 파라미터 효율적 적응(PEFT), 도메인 일반화를 다룹니다. 포스코DX 이전에는 VUNO에서 3년간 비전 모델을 연구부터 인허가 제품까지 완성했습니다.'
          },
          education: {
            unistScholarship: '성적우수 장학금 4학기 (2019-2021)',
            injeScholarship: '성적우수 장학금 8학기 (2014-2018)',
            injeAwards: '인제 창의종합설계 경진대회 금상 · 2017 공학페스티벌 동상'
          },
          tags: {
            physical: 'Physical AI',
            spatial: 'Spatial Intelligence',
            vfm: 'Vision Foundation Model',
            twin: 'Sim2Real · 디지털 트윈'
          },
          journey: {
            l1: '스마트 팩토리',
            t1: 'Vision Inspection',
            d1: '철강 결함 검출과 이미지 검색으로 카메라 영상을 실시간 품질 신호로 바꿉니다.',
            l2: 'AI 팩토리',
            t2: 'MLOps · 지속 학습',
            d2: '데이터 중심 파이프라인과 지속 학습으로 라인·설비별 모델을 계속 최신으로 유지합니다.',
            l3: '인텔리전트 팩토리',
            t3: 'Physical AI · 자율 조업',
            d3: 'L2/PLC와 연결된 Spatial AI로 설비가 스스로 움직입니다. 자율 조업부터 항만 무인 크레인까지 적용합니다.'
          },
          projects: {
            craneTitle: '항만 하역 자동화',
            steelmakingTitle: '자율 조업'
          },
          quickNav: '빠른 이동',
          nav: {
            highlight: 'AX 로드맵',
            about: '소개',
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
            highlight: '스마트 팩토리에서 AI-native 인텔리전트 팩토리로',
            experienceNote: '연구 프로토타입부터 실제 현장에서 돌아가는 시스템까지, 산업 현장 비전 AI 경력입니다.',
            projectsNote: '산업 AX 프로젝트를 먼저, 이전 비전 연구·제품을 그 다음에 정리했습니다. 카드를 누르면 상세 내용을 볼 수 있습니다.',
            educationNote: '학위, 논문, 연구 방향을 간략히 정리한 학력 사항입니다.',
            experience: '경력',
            education: '학력',
            projects: '프로젝트',
            seminar: '세미나 & 강의',
            publications: '논문 & 연구',
            patents: '특허',
            skills: '전문성 & 도구',
            awards: '수상 & 대회',
            recentPosts: '최근 글',
            highlightNote: '데이터를 모으고, 학습하고, 스스로 움직이는 제철소로 가는 과정에서 제가 하는 일입니다.'
          },
          labels: {
            blog: '블로그',
            toolsUsed: '사용 도구',
            thesis: '학위논문',
            domesticPatent: '국내 특허:',
            awards: '수상:',
            onlineCourse: '온라인 강의',
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
            item1: 'Vision Foundation Model · PEFT',
            item2: 'Vision-Language (CLIP) 적응',
            item3: 'Spatial Intelligence · 3D 인지',
            item4: 'Embodied Perception · 센서 퓨전',
            item5: 'Sim2Real · 디지털 트윈 (Isaac Sim)',
            item6: '자기지도학습 (DINO/MAE)',
            item7: '도메인 일반화 · Test-Time Adaptation',
            item8: 'Data-centric AI · 대규모 파이프라인',
            item9: 'Edge AI 배포 · MLOps'
          },
          awards: {
            posco: 'AI 기술 우수상, 포스코DX (2025년 9월)',
            poscoNote: '비전 AI 기반 제철 공정 자동화',
            competitionsTitle: '수상 & 대회',
            academicTitle: '교내 수상',
            scholarships: 'UNIST·인제대 통산 12학기 성적우수 장학금',
            codeLink: '코드 링크',
            gold: '금상',
            bronze: '동상',
            excellence: '우수상'
          },
          experience: {
            posco: {
              title: 'Spatial Intelligence Vision Engineer',
              date: '판교, 대한민국 · 2024년 7월 - 현재',
              bullet1Html: 'Physical AI·디지털 트윈·Sim2Real: 철강 현장 조건을 담은 Isaac Sim 디지털 트윈을 만들고, 인지 모델을 실제 현장으로 이전했습니다.',
              bullet2Html: '<a href="https://kidd.co.kr/news/245144" target="_blank" rel="noopener" class="timeline-link">Spatial Intelligence·3D 센서 퓨전</a>: LiDAR·ToF·RGB를 융합해 3D 장면을 이해하고, 항만 하역 무인화 같은 Vision Control에 적용했습니다.',
              bullet3Html: '<a href="https://newsroom.posco.com/kr/%EC%9D%B8%ED%84%B0%EB%B7%B0-%EB%AF%B8%EB%9E%98%EB%A5%BC-%EC%97%AC%EB%8A%94-%ED%98%81%EC%8B%A0-%EA%B8%B0%EC%88%A0-%EA%B0%9C%EB%B0%9C-2025-%ED%8F%AC%EC%8A%A4%EC%BD%94-%EA%B8%B0%EC%88%A0%EB%8C%80/" target="_blank" rel="noopener" class="timeline-link">자율 조업·L2/PLC 연동</a>: 비전 AI를 L2 통신·제어 신호와 연결해 자율 조업에 적용했습니다.',
              bullet4Html: 'Vision Foundation Model·PEFT·WACV 2027: 공동 1저자로 CAPT를 제안했습니다. CLIP을 완전히 동결한 채 1.35M 이하의 학습 파라미터만으로 DomainNet, Sketchy, TU-Berlin에서 SOTA를 달성했습니다.',
              bullet5Html: '<a href="https://www.digitaltoday.co.kr/news/articleView.html?idxno=500285&rf=toastPopup&utm_source=digitaltoday" target="_blank" rel="noopener" class="timeline-link">철강 결함 검출·MLOps</a>: SDD 모델과 이미지 검색을 개발하고, Dataiku 기반 지속 학습 MLOps Level 1을 구축했습니다.'
            },
            vuno: {
              title: 'AI 연구 엔지니어',
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
