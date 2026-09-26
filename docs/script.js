const dialog = document.querySelector('.figure-dialog');

if (dialog && typeof dialog.showModal === 'function') {
  const fullImage = dialog.querySelector('[data-figure-image]');
  const fullVector = dialog.querySelector('[data-figure-vector]');
  let opener = null;

  fullVector.addEventListener('load', () => {
    // Keyboard events in the embedded SVG do not bubble to the parent page.
    fullVector.contentDocument?.addEventListener('keydown', event => {
      if (event.key === 'Escape' && dialog.open) {
        event.preventDefault();
        dialog.close();
      }
    });
  });

  function openFigure(button, figure, source, title, selectable = false) {
    opener = button;
    dialog.querySelector('#figure-dialog-title').textContent = title;
    fullImage.hidden = selectable;
    fullVector.hidden = !selectable;
    if (selectable) {
      fullVector.data = source;
    } else {
      fullImage.src = source;
      fullImage.alt = title;
    }
    dialog.querySelector('[data-figure-original]').href = source;
    dialog.querySelector('[data-figure-caption]').textContent = figure.querySelector('figcaption')?.textContent || '';
    dialog.showModal();
    document.body.classList.add('dialog-open');
    dialog.querySelector('[data-close-figure]').focus();
  }

  // An object, not an image or button, preserves native SVG text selection.
  const methodologyButton = document.querySelector('[data-enlarge-methodology]');
  if (methodologyButton) {
    methodologyButton.hidden = false;
    methodologyButton.addEventListener('click', () => {
      const figure = methodologyButton.closest('figure');
      openFigure(methodologyButton, figure, figure.querySelector('object').data, 'Branch-paired audit protocol', true);
    });
  }

  document.querySelectorAll('figure > img').forEach(img => {
    const figure = img.closest('figure');
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'figure-zoom';
    button.setAttribute('aria-label', `Enlarge: ${img.alt}`);
    img.before(button);
    button.append(img);
    button.addEventListener('click', () => {
      openFigure(button, figure, img.src, figure.querySelector('h3')?.textContent || img.alt);
    });
  });

  dialog.querySelector('[data-close-figure]').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    if (event.target !== dialog) return;
    const bounds = dialog.getBoundingClientRect();
    if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
  });
  dialog.addEventListener('close', () => {
    document.body.classList.remove('dialog-open');
    opener?.focus({ preventScroll: true });
  });
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
