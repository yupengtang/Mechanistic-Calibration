const header = document.querySelector('[data-header]');
const progress = document.querySelector('[data-progress]');
const menuButton = document.querySelector('[data-menu]');
const nav = document.querySelector('[data-nav]');

function updateHeader() {
  header.classList.toggle('is-scrolled', window.scrollY > 16);
  if (window.scrollY < 80) nav.querySelectorAll('a').forEach(link => {
    link.classList.remove('is-active');
    link.removeAttribute('aria-current');
  });
  const distance = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.transform = `scaleX(${distance > 0 ? Math.min(1, window.scrollY / distance) : 0})`;
}
updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });
window.addEventListener('resize', updateHeader, { passive: true });
function closeMenu() {
  nav.classList.remove('is-open');
  menuButton.setAttribute('aria-expanded', 'false');
}
menuButton.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') === 'true';
  menuButton.setAttribute('aria-expanded', String(!open));
  nav.classList.toggle('is-open', !open);
});
nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && nav.classList.contains('is-open')) {
    closeMenu();
    menuButton.focus();
  }
});

// Content is visible without JavaScript; motion is an optional enhancement.
if ('IntersectionObserver' in window) {
  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
        revealObserver.unobserve(entry.target);
      }
    });
  }, { threshold: 0.08 });
  document.querySelectorAll('.reveal').forEach(node => revealObserver.observe(node));
  const links = [...nav.querySelectorAll('a')];
  const sectionObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) links.forEach(link => {
        const active = link.hash === `#${entry.target.id}`;
        link.classList.toggle('is-active', active);
        if (active) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-15% 0px -50%', threshold: 0 });
  links.forEach(link => sectionObserver.observe(document.querySelector(link.hash)));
}

const branchNotes = {
  neutral: 'Reflection measures revision without an opponent. The initial answer and evidence stay fixed. Illustrative outcomes only.',
  anonymous: 'An unnamed assistant gives the opposing answer. Comparing A₁ with A₀ measures the effect of disagreement. Illustrative outcomes only.',
  expert: 'The same opposition is attributed to a domain expert. Comparing A₃ with A₁ measures the source-label effect. Illustrative outcomes only.'
};
const branches = document.querySelectorAll('[data-branch]');
branches.forEach(button => button.addEventListener('click', () => {
  branches.forEach(item => {
    item.classList.toggle('is-active', item === button);
    item.setAttribute('aria-pressed', String(item === button));
  });
  document.querySelector('[data-demo-note]').textContent = branchNotes[button.dataset.branch];
}));

const resultTabs = [...document.querySelectorAll('[data-result]')];
function selectResult(tab) {
  resultTabs.forEach(item => {
    const active = item === tab;
    item.setAttribute('aria-selected', String(active));
    item.tabIndex = active ? 0 : -1;
    const panel = document.getElementById(item.dataset.result);
    panel.hidden = !active;
    panel.setAttribute('role', 'tabpanel');
    panel.tabIndex = 0;
  });
  updateHeader();
}
document.querySelector('.result-tabs').hidden = false;
selectResult(resultTabs[0]);
resultTabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectResult(tab));
  tab.addEventListener('keydown', event => {
    if (!['ArrowRight', 'ArrowLeft', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? resultTabs.length - 1 :
      (index + (event.key === 'ArrowRight' ? 1 : -1) + resultTabs.length) % resultTabs.length;
    selectResult(resultTabs[next]);
    resultTabs[next].focus();
  });
});

const dialog = document.querySelector('.figure-dialog');
if (typeof dialog.showModal === 'function') {
  const fullImage = document.querySelector('[data-figure-image]');
  const fullVector = document.querySelector('[data-figure-vector]');
  fullVector.addEventListener('load', () => {
    // Keyboard events inside an embedded SVG do not bubble to the page.
    fullVector.contentDocument?.addEventListener('keydown', event => {
      if (event.key === 'Escape' && dialog.open) {
        event.preventDefault();
        dialog.close();
      }
    });
  });
  function openFigure(figure, source, title, selectable = false) {
    document.querySelector('#figure-dialog-title').textContent = title;
    fullImage.hidden = selectable;
    fullVector.hidden = !selectable;
    if (selectable) {
      fullVector.data = source;
    } else {
      fullImage.src = source;
      fullImage.alt = title;
    }
    document.querySelector('[data-figure-original]').href = source;
    document.querySelector('[data-figure-caption]').textContent = figure.querySelector('figcaption')?.textContent || '';
    dialog.showModal();
    document.body.classList.add('dialog-open');
    document.querySelector('[data-close-figure]').focus();
  }
  // Keep the SVG document outside a button so native text selection works.
  const methodologyButton = document.querySelector('[data-enlarge-methodology]');
  methodologyButton.hidden = false;
  methodologyButton.addEventListener('click', () => {
    const figure = methodologyButton.closest('figure');
    openFigure(figure, figure.querySelector('object').data, 'Two-pass branch-paired audit protocol', true);
  });
  document.querySelectorAll('figure > img').forEach(img => {
    const figure = img.closest('figure');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'figure-zoom';
    button.setAttribute('aria-label', `Enlarge: ${img.alt}`);
    img.before(button);
    button.append(img);
    button.addEventListener('click', () => {
      openFigure(figure, img.src, figure.querySelector('h3')?.textContent || img.alt);
    });
  });
  document.querySelector('[data-close-figure]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));
}

document.querySelectorAll('[data-copy-target]').forEach(button => {
  button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copyTarget);
    const label = button.textContent;
    try {
      await navigator.clipboard.writeText(source.textContent.trim());
      button.textContent = 'Copied';
      document.querySelector('[data-copy-status]').textContent = `${label} copied to clipboard.`;
      window.setTimeout(() => { button.textContent = label; }, 1800);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(source);
      const selection = window.getSelection();
      selection.removeAllRanges();
      selection.addRange(range);
      document.querySelector('[data-copy-status]').textContent = 'Text selected. Use your browser’s copy command.';
    }
  });
});
