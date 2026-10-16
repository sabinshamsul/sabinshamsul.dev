// Clickjacking guard. GitHub Pages cannot send frame-ancestors or X-Frame-Options, and
// browsers ignore frame-ancestors in a <meta> CSP, so another site could load this page in
// an invisible frame and trick a visitor into using the contact form. This loads in <head>
// so the page is hidden before any of it renders: inside any frame, the CSS in index.html
// hides everything but a notice with the address to open directly. The address is shown as
// text too, since a sandboxed frame may block the link from opening.
if (window.top !== window.self) {
  document.documentElement.classList.add('framed');
  document.addEventListener('DOMContentLoaded', () => {
    const url = location.href.split('#')[0];
    const notice = Object.assign(document.createElement('p'), { className: 'framed-notice', textContent: 'This site cannot be shown inside another page. Open it directly: ' });
    notice.append(Object.assign(document.createElement('a'), { href: url, target: '_blank', rel: 'noopener', textContent: url }));
    document.body.append(notice);
  });
}
