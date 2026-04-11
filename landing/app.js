// ============================================================
// VolTradeAI Landing — Interactions
// ============================================================

(function () {
  // --- Theme Toggle ---
  const toggle = document.querySelector('[data-theme-toggle]');
  const root = document.documentElement;
  let theme = matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  root.setAttribute('data-theme', theme);
  updateToggleIcon();

  if (toggle) {
    toggle.addEventListener('click', () => {
      theme = theme === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', theme);
      toggle.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
      updateToggleIcon();
    });
  }

  function updateToggleIcon() {
    if (!toggle) return;
    toggle.innerHTML = theme === 'dark'
      ? '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>'
      : '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';
  }

  // --- Counter Animation ---
  const counters = document.querySelectorAll('.counter');
  let countersAnimated = false;

  function animateCounters() {
    if (countersAnimated) return;
    countersAnimated = true;

    counters.forEach(el => {
      const target = parseFloat(el.dataset.target);
      const suffix = el.dataset.suffix || '';
      const duration = 1200;
      const start = performance.now();
      const decimals = target % 1 !== 0 ? (target.toString().split('.')[1] || '').length : 0;

      function tick(now) {
        const elapsed = now - start;
        const progress = Math.min(elapsed / duration, 1);
        // Ease-out cubic
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = target * eased;
        el.textContent = current.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    });
  }

  // Use Intersection Observer for counters
  if (counters.length) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          animateCounters();
          observer.disconnect();
        }
      });
    }, { threshold: 0.3 });
    observer.observe(counters[0].closest('.hero__metrics'));
  }

  // --- Smooth scroll for nav links ---
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener('click', e => {
      const href = link.getAttribute('href');
      if (href === '#') return;
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // --- Animate progress bars on scroll ---
  const bars = document.querySelectorAll('.comp-card__bar-fill');
  if (bars.length) {
    // Set initial width to 0, then animate in
    bars.forEach(bar => {
      bar.dataset.width = bar.style.width;
      bar.style.width = '0%';
    });

    const barObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const fill = entry.target.querySelector('.comp-card__bar-fill');
          if (fill) {
            setTimeout(() => {
              fill.style.width = fill.dataset.width;
            }, 150);
          }
          barObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.2 });

    document.querySelectorAll('.comp-card').forEach(card => {
      barObserver.observe(card);
    });
  }

  // --- Equity Curve Chart ---
  function initEquityChart() {
    const canvas = document.getElementById('equityCurve');
    if (!canvas || typeof Chart === 'undefined') return;

    const labels = ['2014','2015','2016','2017','2018','2019','2020','2021','2022','2023','2024','2025'];
    const volData = [100000,127109,182757,304658,286378,513200,844814,1452322,1361831,2156674,3325554,4677519];
    const spyData = [100000,101230,113378,137992,131686,172798,204472,263217,215364,271746,339384,399522];

    const style = getComputedStyle(document.documentElement);
    const primary = style.getPropertyValue('--color-primary').trim();
    const muted = style.getPropertyValue('--color-text-faint').trim();
    const textColor = style.getPropertyValue('--color-text-muted').trim();
    const gridColor = style.getPropertyValue('--color-divider').trim();

    const ctx = canvas.getContext('2d');

    // Gradient fill for VolTradeAI
    const gradient = ctx.createLinearGradient(0, 0, 0, canvas.parentElement.clientHeight);
    gradient.addColorStop(0, primary.startsWith('#')
      ? primary + '20'
      : 'rgba(59, 130, 246, 0.12)');
    gradient.addColorStop(1, 'transparent');

    window.__equityChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'SPY',
            data: spyData,
            borderColor: muted,
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            borderDash: [6, 3],
            fill: false,
            tension: 0.35,
            pointRadius: 3,
            pointHoverRadius: 5,
            pointBackgroundColor: muted,
            pointBorderColor: 'transparent',
            pointHoverBorderColor: muted,
            pointHoverBorderWidth: 2,
          },
          {
            label: 'VolTradeAI',
            data: volData,
            borderColor: primary,
            backgroundColor: gradient,
            borderWidth: 2.5,
            fill: true,
            tension: 0.35,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: primary,
            pointBorderColor: 'transparent',
            pointHoverBorderColor: primary,
            pointHoverBorderWidth: 2,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: style.getPropertyValue('--color-surface-2').trim() || '#171b20',
            titleColor: style.getPropertyValue('--color-text').trim() || '#e0e4e8',
            bodyColor: style.getPropertyValue('--color-text-muted').trim() || '#8b929a',
            borderColor: style.getPropertyValue('--color-border').trim() || '#2d333b',
            borderWidth: 1,
            padding: 12,
            cornerRadius: 8,
            titleFont: { family: 'General Sans, sans-serif', weight: '600', size: 13 },
            bodyFont: { family: 'General Sans, sans-serif', size: 12 },
            callbacks: {
              title: function(items) {
                return items[0].label === '2014' ? 'Start (End of 2014)' : 'End of ' + items[0].label;
              },
              label: function(context) {
                const val = context.parsed.y;
                if (val >= 1000000) {
                  return ' ' + context.dataset.label + ': $' + (val / 1000000).toFixed(2) + 'M';
                }
                return ' ' + context.dataset.label + ': $' + val.toLocaleString();
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor, drawBorder: false },
            ticks: {
              color: textColor,
              font: { family: 'General Sans, sans-serif', size: 11 },
              callback: function(value, index) {
                return labels[index] === '2014' ? 'Start' : labels[index];
              }
            },
            border: { display: false },
          },
          y: {
            type: 'logarithmic',
            grid: { color: gridColor, drawBorder: false },
            ticks: {
              color: textColor,
              font: { family: 'General Sans, sans-serif', size: 11 },
              callback: function(value) {
                if (value >= 1000000) return '$' + (value / 1000000).toFixed(0) + 'M';
                if (value >= 1000) return '$' + (value / 1000).toFixed(0) + 'K';
                return '$' + value;
              },
              maxTicksLimit: 6,
            },
            border: { display: false },
          }
        }
      }
    });
  }

  // Initialize chart when visible
  const chartWrap = document.querySelector('.equity-chart-wrap');
  if (chartWrap) {
    const chartObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          // Small delay for Chart.js to be loaded
          if (typeof Chart !== 'undefined') {
            initEquityChart();
          } else {
            // Poll for Chart.js
            const poll = setInterval(() => {
              if (typeof Chart !== 'undefined') {
                clearInterval(poll);
                initEquityChart();
              }
            }, 100);
          }
          chartObserver.disconnect();
        }
      });
    }, { threshold: 0.1 });
    chartObserver.observe(chartWrap);
  }

  // Re-init chart on theme change to update colors
  if (toggle) {
    toggle.addEventListener('click', () => {
      if (window.__equityChart) {
        window.__equityChart.destroy();
        window.__equityChart = null;
        setTimeout(initEquityChart, 50);
      }
    });
  }

  // --- Fallback fade-in for browsers without animation-timeline ---
  if (!CSS.supports('animation-timeline', 'scroll()')) {
    const fadeEls = document.querySelectorAll('.fade-in');
    if (fadeEls.length) {
      fadeEls.forEach(el => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.6s cubic-bezier(0.16, 1, 0.3, 1)';
      });

      const fadeObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
          if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            fadeObserver.unobserve(entry.target);
          }
        });
      }, { threshold: 0.1 });

      fadeEls.forEach(el => fadeObserver.observe(el));
    }
  }
})();
