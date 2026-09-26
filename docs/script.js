const header = document.querySelector('[data-header]');
const menuButton = document.querySelector('[data-menu]');
const nav = document.querySelector('[data-nav]');

function updateHeader() {
  header?.classList.toggle('is-scrolled', window.scrollY > 16);
}
updateHeader();
window.addEventListener('scroll', updateHeader, { passive: true });

menuButton?.addEventListener('click', () => {
  const open = menuButton.getAttribute('aria-expanded') === 'true';
  menuButton.setAttribute('aria-expanded', String(!open));
  nav?.classList.toggle('is-open', !open);
});

nav?.querySelectorAll('a').forEach((link) => {
  link.addEventListener('click', () => {
    nav.classList.remove('is-open');
    menuButton?.setAttribute('aria-expanded', 'false');
  });
});

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (entry.isIntersecting) {
      entry.target.classList.add('is-visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.12 });
document.querySelectorAll('.reveal').forEach((node) => revealObserver.observe(node));

const sectionLinks = [...document.querySelectorAll('.site-nav a')];
const sections = sectionLinks
  .map((link) => document.querySelector(link.getAttribute('href')))
  .filter(Boolean);
const sectionObserver = new IntersectionObserver((entries) => {
  entries.forEach((entry) => {
    if (!entry.isIntersecting) return;
    sectionLinks.forEach((link) => {
      link.classList.toggle('is-active', link.getAttribute('href') === `#${entry.target.id}`);
    });
  });
}, { rootMargin: '-25% 0px -68%', threshold: 0 });
sections.forEach((section) => sectionObserver.observe(section));

const branchNotes = {
  neutral: 'Neutral reconsideration estimates ordinary prompt drift without an opponent.',
  anonymous: 'Anonymous disagreement adds an opposing answer without an expertise cue.',
  expert: 'Expert-labeled disagreement changes only the source description relative to the anonymous branch.'
};
const branchButtons = document.querySelectorAll('[data-branch]');
const demoNote = document.querySelector('[data-demo-note]');
branchButtons.forEach((button) => {
  button.addEventListener('click', () => {
    branchButtons.forEach((item) => item.classList.remove('is-active'));
    button.classList.add('is-active');
    if (demoNote) demoNote.textContent = branchNotes[button.dataset.branch];
  });
});

document.querySelector('[data-copy]')?.addEventListener('click', async (event) => {
  const text = document.querySelector('#bibtex code')?.textContent ?? '';
  try {
    await navigator.clipboard.writeText(text);
    event.currentTarget.textContent = 'Copied';
    window.setTimeout(() => { event.currentTarget.textContent = 'Copy BibTeX'; }, 1800);
  } catch {
    event.currentTarget.textContent = 'Select and copy';
  }
});
