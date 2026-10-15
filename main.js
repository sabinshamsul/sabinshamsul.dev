// Theme
const root = document.documentElement;
const themeToggle = document.getElementById('themeToggle');
const saved = localStorage.getItem('theme');
const systemLight = window.matchMedia('(prefers-color-scheme: light)').matches;
if (saved === 'light' || (!saved && systemLight)) root.setAttribute('data-theme', 'light');
themeToggle.addEventListener('click', () => {
  const next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
  if (next === 'dark') root.removeAttribute('data-theme'); else root.setAttribute('data-theme', 'light');
  localStorage.setItem('theme', next);
});

// Mobile menu
const hamburger = document.getElementById('navHamburger');
const navLinks = document.getElementById('navLinks');
const closeMenu = () => { hamburger.setAttribute('aria-expanded', 'false'); navLinks.classList.remove('open'); };
hamburger.addEventListener('click', () => {
  const open = hamburger.getAttribute('aria-expanded') === 'true';
  hamburger.setAttribute('aria-expanded', String(!open));
  navLinks.classList.toggle('open', !open);
});
navLinks.querySelectorAll('a').forEach(a => a.addEventListener('click', closeMenu));
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeMenu(); });

// Scroll reveal (staggered children get an index)
document.querySelectorAll('.stagger').forEach(g => [...g.children].forEach((c, i) => c.style.setProperty('--i', i)));
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('visible'); revealObserver.unobserve(entry.target); } });
}, { threshold: 0.08, rootMargin: '0px 0px -8% 0px' });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

// Scroll progress + nav state (rAF-throttled)
const progress = document.getElementById('progress');
const navEl = document.querySelector('nav');
let ticking = false;
const onScroll = () => {
  if (ticking) return; ticking = true;
  requestAnimationFrame(() => {
    const max = document.documentElement.scrollHeight - innerHeight;
    progress.style.transform = `scaleX(${max > 0 ? Math.min(1, scrollY / max) : 0})`;
    navEl.classList.toggle('scrolled', scrollY > 8);
    ticking = false;
  });
};
addEventListener('scroll', onScroll, { passive: true }); onScroll();

// Drifting dust background
(() => {
  const canvas = document.getElementById('dust');
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const ctx = canvas.getContext('2d');
  let w = 0, h = 0, dpr = 1, dots = [], raf = 0, last = 0;
  const isLight = () => root.getAttribute('data-theme') === 'light';
  let rgb = isLight() ? '29,78,216' : '147,197,253';
  const rand = (a, b) => a + Math.random() * (b - a);
  const make = (fresh) => ({
    x: rand(0, w), y: fresh ? h + rand(0, 40) : rand(0, h),
    r: rand(0.6, 1.9), vy: rand(6, 18), vx: rand(-4, 4),
    a: rand(0.25, 1), tw: rand(0.4, 1.4), ph: rand(0, Math.PI * 2)
  });
  const resize = () => {
    dpr = Math.min(2, devicePixelRatio || 1);
    w = innerWidth; h = innerHeight;
    canvas.width = w * dpr; canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(110, Math.max(30, Math.round((w * h) / 16000)));
    dots = Array.from({ length: n }, () => make(false));
  };
  const tick = (t) => {
    const dt = Math.min(0.05, (t - last) / 1000 || 0); last = t;
    ctx.clearRect(0, 0, w, h);
    const base = isLight() ? 0.32 : 0.5;
    for (const d of dots) {
      d.y -= d.vy * dt; d.x += d.vx * dt + Math.sin(t / 1400 + d.ph) * 0.15;
      if (d.y < -10) Object.assign(d, make(true));
      if (d.x < -10) d.x = w + 10; else if (d.x > w + 10) d.x = -10;
      const tw = 0.55 + 0.45 * Math.sin(t / 1000 * d.tw + d.ph);
      ctx.beginPath(); ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${rgb},${(base * d.a * tw).toFixed(3)})`; ctx.fill();
    }
    raf = requestAnimationFrame(tick);
  };
  const start = () => { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };
  resize(); start();
  addEventListener('resize', resize, { passive: true });
  document.addEventListener('visibilitychange', () => document.hidden ? stop() : start());
  new MutationObserver(() => { rgb = isLight() ? '29,78,216' : '147,197,253'; }).observe(root, { attributes: true, attributeFilter: ['data-theme'] });
})();

// Project carousel: cards tilt away from the centre like flipping through records
(() => {
  const track = document.getElementById('projectTrack');
  const cards = [...track.querySelectorAll('.card')];
  const prev = document.querySelector('.car-prev'), next = document.querySelector('.car-next');
  const flat = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const step = () => cards[0].offsetWidth + 8;
  let raf = 0;
  const update = () => {
    raf = 0;
    const mid = track.scrollLeft + track.clientWidth / 2;
    cards.forEach(c => {
      const o = flat ? 0 : Math.max(-1.5, Math.min(1.5, (c.offsetLeft + c.offsetWidth / 2 - mid) / step()));
      c.style.setProperty('--o', o.toFixed(3));
      c.style.setProperty('--a', Math.abs(o).toFixed(3));
      c.style.zIndex = 10 - Math.round(Math.abs(o) * 4);
    });
    prev.disabled = track.scrollLeft < 4;
    next.disabled = track.scrollLeft + track.clientWidth > track.scrollWidth - 4;
  };
  const queue = () => { if (!raf) raf = requestAnimationFrame(update); };
  prev.addEventListener('click', () => track.scrollBy({ left: -step() }));
  next.addEventListener('click', () => track.scrollBy({ left: step() }));
  track.addEventListener('scroll', queue, { passive: true });
  addEventListener('resize', queue);
  update();
})();

// Count-up for dialog stats
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
function countUp(el) {
  const text = el.textContent.trim();
  const m = text.match(/^([~]?)([\d,]*\.?\d+)(.*)$/);
  if (!m || reduceMotion) return;
  const prefix = m[1], suffix = m[3], target = parseFloat(m[2].replace(/,/g, ''));
  const decimals = (m[2].split('.')[1] || '').length;
  const useComma = m[2].includes(',');
  const t0 = performance.now(), dur = 900;
  const fmt = v => { let n = v.toFixed(decimals); if (useComma) n = Number(n).toLocaleString('en-US', { minimumFractionDigits: decimals }); return prefix + n + suffix; };
  const step = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(target * e);
    if (k < 1) requestAnimationFrame(step); else el.textContent = text;
  };
  requestAnimationFrame(step);
  setTimeout(() => { el.textContent = text; }, dur + 80);
}

// ─── Project detail dialog ───
const GH = 'https://raw.githubusercontent.com/sabinshamsul/';
const PROJECTS = {
  medivision: {
    title: 'MediVision: Smart Patient Flow Platform',
    links: [
      { label: 'Open Live Demo', href: 'https://medivision-dashboard.vercel.app/', primary: true },
      { label: 'Watch Walkthrough', href: 'https://youtu.be/a-wKaxb9S9o' },
      { label: 'GitHub', href: 'https://github.com/sabinshamsul/medivision-dashboard' }
    ],
    demo: { src: 'https://medivision-dashboard.vercel.app/', note: 'Hosted on Vercel. If it does not load here, open it in a new tab.' },
    problem: [
      'Emergency departments triage patients under pressure, with paper handovers and no shared view of where each patient is. Severity gets misjudged at intake, staff cannot see bed availability or wait times in real time, and department-wide metrics are only visible after the fact.',
      'Our capstone brief was to build a working platform that fixes that flow end to end, as a team of five. I was project manager and owned the backend and the ML service integration.'
    ],
    solution: [
      '<strong>Triage recommendation from data.</strong> An XGBoost classifier predicts severity from vital signs and symptoms at intake. Nurses review and can override every prediction, and overrides are logged, so the model assists rather than decides.',
      '<strong>One live view of the department.</strong> Patient status updates push to every screen over Socket.io, so staff see the pathway without refreshing.',
      '<strong>Metrics while they still matter.</strong> Department load, wait times and patient flow are charted with Recharts on a dashboard, not in a monthly report.',
      '<strong>Right access for each role.</strong> JWT logins with role-based permissions for clinical staff.',
      '<strong>Model that can evolve.</strong> The classifier runs in its own FastAPI microservice, so it can be retrained and redeployed without touching the main app.'
    ],
    shots: [
      ['medivision-dashboard/main/screenshots/dashboard.png', 'Dashboard'],
      ['medivision-dashboard/main/screenshots/triage.png', 'AI triage with nurse review'],
      ['medivision-dashboard/main/screenshots/tracking.png', 'Patient tracking'],
      ['medivision-dashboard/main/screenshots/analytics.png', 'Analytics']
    ]
  },
  bone: {
    title: 'Bone Fracture Detection With Grad-CAM',
    links: [
      { label: 'Open Live Demo', href: 'https://huggingface.co/spaces/sabinshamsul/bone-fracture-detection', primary: true },
      { label: 'GitHub', href: 'https://github.com/sabinshamsul/Bone-Fracture-Detection' }
    ],
    demo: { src: 'https://sabinshamsul-bone-fracture-detection.hf.space', allow: 'camera; clipboard-read', note: 'Upload an X-ray of a hand, leg, hip or shoulder to get a prediction and a Grad-CAM heatmap. The Space may take a minute to wake up.' },
    stats: [['86%', 'Overall accuracy'], ['74%', 'Fracture recall'], ['0.89', 'Validation AUC'], ['4,083', 'Hospital X-rays']],
    problem: [
      'Fractures are easy to miss on X-rays, especially subtle ones, and a second reader is not always available. A classifier alone is not enough in a clinical setting: if it cannot show why it flagged an image, no one can check it.',
      'The data adds a second problem. In the FracAtlas dataset only 17.6% of the 4,083 hospital X-rays are fractured, so a naive model learns to say "no fracture" and still looks accurate.'
    ],
    solution: [
      '<strong>Transfer learning instead of training from scratch.</strong> EfficientNetB0 pretrained on ImageNet, fine-tuned in two stages in TensorFlow on a Colab GPU.',
      '<strong>Better input before the model sees it.</strong> CLAHE contrast enhancement in OpenCV makes bone edges and fracture lines clearer.',
      '<strong>Class weighting for the imbalance.</strong> Fractured cases are weighted up during training, lifting fracture recall to 74% with 86% overall accuracy and 0.89 AUC.',
      '<strong>Explanations, not just answers.</strong> Grad-CAM heatmaps show which regions drove each prediction. Reviewing them confirmed the model attends to bone shafts, joints and surgical hardware.',
      '<strong>A weakness found and documented.</strong> In 2 of 10 samples the model focused on image-border artefacts from CLAHE. Recorded as shortcut learning, with border cropping and Integrated Gradients as next steps.',
      '<strong>Usable by anyone.</strong> Deployed as a public Gradio app on Hugging Face Spaces, plus a CLI for batch inference.'
    ],
    shots: [
      ['Bone-Fracture-Detection/main/results/gradcam_batch.png', 'Grad-CAM heatmaps on sample X-rays', 'wide'],
      ['Bone-Fracture-Detection/main/results/training_history.png', 'Training history'],
      ['Bone-Fracture-Detection/main/results/confusion_matrix.png', 'Confusion matrix']
    ]
  },
  rag: {
    title: 'RAG Over AI-Governance Documents',
    links: [
      { label: 'Open Live Demo', href: 'https://huggingface.co/spaces/sabinshamsul/rag-ai-governance', primary: true },
      { label: 'GitHub', href: 'https://github.com/sabinshamsul/rag-ai' }
    ],
    demo: { src: 'https://sabinshamsul-rag-ai-governance.hf.space', note: 'Ask a question about AI laws or policies. Answers are grounded in the AGORA corpus with citations. Free-tier rate limits apply.' },
    stats: [['~650', 'Laws and policies'], ['5,203', 'Indexed chunks'], ['384', 'Embedding dimensions']],
    problem: [
      'Ask a language model about AI regulation and it answers from memory, confidently, and sometimes wrongly. For legal and policy questions that is worse than no answer, because a fabricated citation looks exactly like a real one.',
      'The AGORA corpus has roughly 650 AI laws, bills and policy frameworks split into 5,400 sections. The task was to answer questions using only that text, cite every claim, and refuse when the corpus does not cover the question.'
    ],
    solution: [
      '<strong>Retrieve first, then answer.</strong> Each question is embedded with all-MiniLM-L6-v2 and matched against a FAISS index of 5,203 cleaned chunks, so the model only ever sees relevant source passages.',
      '<strong>Grounding enforced in the prompt.</strong> Gemini is instructed to answer only from the retrieved context, cite sources by number, and synthesise across jurisdictions where they differ.',
      '<strong>Refusal by design.</strong> If the best match scores below 0.3, the system says it does not have enough information and never calls the model. The threshold came from the observed gap between relevant and irrelevant queries.',
      '<strong>Respecting the data.</strong> The dataset\'s expert segmentation is kept instead of re-chunking, because each segment is a self-contained legal provision. Non-operative segments are filtered out before indexing.',
      '<strong>Built without a framework.</strong> No LangChain or LlamaIndex, so every step from loading to citation is understood and can be swapped.',
      '<strong>Limits stated openly.</strong> General-purpose embeddings, truncated long segments, manual evaluation and no conversation memory are all documented.'
    ],
    shots: [
      ['rag-ai/main/screenshots/06_generation_output.png', 'Grounded answer with citations across jurisdictions', 'wide'],
      ['rag-ai/main/screenshots/04_retrieval_output.png', 'Semantic retrieval matching on meaning'],
      ['rag-ai/main/screenshots/07_generation_output.png', 'Declining when the corpus cannot answer']
    ]
  },
  loan: {
    title: 'Loan Repayment Risk Prediction',
    links: [
      { label: 'GitHub', href: 'https://github.com/sabinshamsul/loan-repayment', primary: true, note: 'private repo' }
    ],
    stats: [['0.814', 'AUC-ROC'], ['80%', 'Bad-loan recall'], ['74%', 'Accuracy'], ['~525k', 'Payment records']],
    problem: [
      'A lender sees nearly 60% of funded loans default. Every bad loan approved is a direct loss, so the question is which applicants to flag before money goes out.',
      'The data made that harder than it sounds: three separate tables (applications, per-loan payment history, third-party underwriting checks), about 525,000 payment records, and no target column to predict.'
    ],
    solution: [
      '<strong>Defined the target.</strong> Engineered a binary repayment label from loan status, and excluded loans whose outcome was still unknown so they would not add noise.',
      '<strong>Turned payment history into features.</strong> Payment success rate, total payment attempts and first-payment status were built from the payments table and merged onto each loan. Two of the top four predictors came from this set.',
      '<strong>Compared a baseline against an ensemble.</strong> Random Forest beat Logistic Regression on every metric and was selected: 0.814 AUC, 80% recall on bad loans, 74% accuracy.',
      '<strong>Found what actually predicts default.</strong> Scheduled payment amount, payment success rate, fraud score and a failed first payment. Returning customers repaid far more often than cold leads; weekly payers were the riskiest group.',
      '<strong>Made it actionable.</strong> Recommendations: prioritise applicants with payment history, send failed first payments to manual review, price weekly-payment borrowers as higher risk.'
    ],
    shots: [
      ['assets/loan/top_15_features.png', 'Top 15 features by importance. Scheduled payment amount leads, followed by payment success rate and fraud score', 'wide', true],
      ['assets/loan/roc_curve.png', 'ROC curve. Random Forest at 0.814 AUC beats Logistic Regression at 0.786 across all thresholds', '', true],
      ['assets/loan/model_comparison.png', 'Random Forest versus Logistic Regression across every metric', '', true],
      ['assets/loan/confusion_matrix_random_forest.png', 'Confusion matrix. 2,947 of 3,682 bad loans correctly flagged', '', true],
      ['assets/loan/corr_matrix_top_15.png', 'Correlation matrix for the top 15 features', '', true],
      ['assets/loan/loan_repayment_distribution.png', 'Repayment outcomes. Nearly 60% of funded loans defaulted', '', true],
      ['assets/loan/loan_amount_distribution.png', 'Loan amount by outcome. Defaults concentrate in the $200 to $600 range', '', true],
      ['assets/loan/repayment_rate_leadtype.png', 'Repayment rate by lead type. Returning customers repay at 100%, cold leads at 34%', '', true],
      ['assets/loan/repayment_rate.png', 'Repayment rate by payment frequency. Weekly payers repay least often at 37%', '', true],
      ['assets/loan/top10states.png', 'Top 10 states by loan volume, with repayment rates from 28% to 54%', '', true],
      ['assets/loan/payment_status_breakdown.png', 'Payment status across 525,000 records. 270,000 cancelled, 32,000 rejected', '', true],
      ['assets/loan/top_10_reasons_rejected.png', 'Reasons for rejected payments. Insufficient funds dwarfs every other cause', '', true],
      ['assets/loan/distribution_num_payments.png', 'Payments per loan. Most average 17, with a tail beyond 40', '', true],
      ['assets/loan/normal_vs_collection_plan.png', 'Only 2% of payments went through a collection plan', '', true]
    ]
  },
  kadku: {
    title: 'kadku: A Digital Business Card',
    links: [
      { label: 'Open Live Site', href: 'https://kadku.sabinshamsul.dev/', primary: true }
    ],
    demo: { src: 'https://kadku.sabinshamsul.dev/', note: 'This is the real card. It is designed for a phone screen, so it looks best in the narrow frame.' },
    problem: [
      'At career fairs the same thing kept happening. Spelling out an email address letter by letter. Helping a recruiter search for the right LinkedIn profile. Having only a paper copy of a resume. Every one of those moments is friction right when it matters most.',
      'The card had to work on a stranger\'s phone in seconds, with nothing to install and nothing to type.'
    ],
    solution: [
      '<strong>One scan, everything saved.</strong> A vCard QR code writes contact details straight into the recruiter\'s phone book.',
      '<strong>No searching.</strong> Direct QR links to the LinkedIn profile and the latest PDF resume.',
      '<strong>Built for a phone screen.</strong> A mobile-first accordion: each section opens on tap and scrolls the full QR code into view.',
      '<strong>Bilingual.</strong> A toggle switches the whole card between Bahasa Melayu and English.',
      '<strong>Zero maintenance hosting.</strong> Static HTML and Tailwind on GitHub Pages, deployed by GitHub Actions on every push, served from a custom subdomain.'
    ]
  },
  plants: {
    title: 'Smart Monitoring For Houseplants',
    links: [
      { label: 'Open Wokwi Simulation', href: 'https://wokwi.com/projects/436298272500748289', primary: true }
    ],
    problem: [
      'Houseplants die from inconsistent care: overwatered one week, forgotten the next. Soil moisture, temperature, humidity and pH all matter, but none of them are visible without checking by hand.',
      'The coursework brief was a working IoT prototype that senses those conditions and acts on them without a person in the loop.'
    ],
    solution: [
      '<strong>Sense the plant, not the calendar.</strong> An ESP32 reads soil moisture, temperature, humidity and soil pH sensors on a schedule.',
      '<strong>Water only when needed.</strong> A relay-driven pump runs when moisture drops below a threshold, so the plant is watered by its actual condition.',
      '<strong>See it in real time.</strong> Readings stream to a Raspberry Pi for logging and monitoring.',
      '<strong>Reproducible.</strong> The circuit and firmware run in the Wokwi simulator, so the build can be inspected and tested without the hardware.'
    ]
  }
};

const dialog = document.getElementById('projectDialog');
const pdBody = document.getElementById('pdBody');
let lastFocus = null;
const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');

function renderProject(key) {
  const p = PROJECTS[key];
  if (!p) return;
  document.getElementById('pdTitle').textContent = p.title;

  let html = '<div class="pd-actions">' + p.links.map(l =>
    `<a class="btn ${l.primary ? 'btn-primary' : 'btn-secondary'}" href="${l.href}" target="_blank" rel="noopener">${esc(l.label)} ↗</a>`
  ).join('') + '</div>';

  if (p.stats) html += '<div class="pd-stats">' + p.stats.map(([n, l]) =>
    `<div><div class="pd-stat-num">${esc(n)}</div><div class="pd-stat-label">${esc(l)}</div></div>`).join('') + '</div>';

  html += '<div class="pd-section"><h3>Problem</h3>' + p.problem.map(t => `<p>${t}</p>`).join('') + '</div>';
  html += '<div class="pd-section"><h3>Solution</h3><ol>' + p.solution.map(t => `<li>${t}</li>`).join('') + '</ol></div>';

  if (p.shots) html += '<div class="pd-section"><h3>Screenshots</h3><div class="pd-shots">' + p.shots.map(([src, cap, cls, local]) =>
    `<figure class="${cls || ''}" tabindex="0" role="button" aria-label="Open ${esc(cap)} in viewer"><img loading="lazy" src="${local ? src : GH + src}" alt="${esc(cap)}"><figcaption>${esc(cap)}</figcaption></figure>`).join('') + '</div></div>';

  if (p.demo) html += `<div class="pd-section pd-demo"><h3>Try it here</h3>
    <div class="pd-demo-frame" data-src="${p.demo.src}" data-allow="${p.demo.allow || ''}"><button type="button" class="btn btn-secondary load-demo">Load demo</button></div>
    <p class="pd-demo-note">${esc(p.demo.note)}</p></div>`;


  pdBody.innerHTML = html;
  pdBody.scrollTop = 0;
  pdBody.querySelectorAll('.pd-stat-num').forEach(countUp);
  pdBody.querySelectorAll('.pd-shots figure').forEach(f => f.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); f.click(); }
  }));
}

function openProject(key) {
  if (!PROJECTS[key]) return;
  lastFocus = document.activeElement;
  renderProject(key);
  dialog.showModal();
  document.body.style.overflow = 'hidden';
  history.replaceState(null, '', '#project-' + key);
}
dialog.addEventListener('close', () => {
  if (dialog.open) return; // reopened before this fired
  pdBody.innerHTML = '';
  document.body.style.overflow = '';
  if (location.hash.startsWith('#project-')) history.replaceState(null, '', '#projects');
  if (lastFocus) lastFocus.focus();
});
document.getElementById('pdClose').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', e => { if (e.target === dialog) dialog.close(); });

pdBody.addEventListener('click', e => {
  const fig = e.target.closest('.pd-shots figure');
  if (fig) {
    const figs = [...pdBody.querySelectorAll('.pd-shots figure')];
    LB.open(figs.map(f => ({ src: f.querySelector('img').src, caption: f.querySelector('figcaption').textContent })), figs.indexOf(fig), document.getElementById('pdTitle').textContent);
    return;
  }
  const btn = e.target.closest('.load-demo');
  if (!btn) return;
  const frame = btn.closest('.pd-demo-frame');
  const iframe = document.createElement('iframe');
  // Demos are third-party apps, so sandbox them. They keep what they use (scripts, their
  // own storage, form submits, links opening in a new tab, downloads), but never get
  // allow-top-navigation, so a compromised demo cannot redirect this tab.
  // allow-same-origin keeps each demo on its own origin (vercel.app, hf.space,
  // kadku.sabinshamsul.dev) instead of an opaque one. All of those are cross-origin to this
  // page, which is why allow-scripts + allow-same-origin cannot strip the sandbox here.
  iframe.sandbox = 'allow-scripts allow-same-origin allow-forms allow-popups allow-popups-to-escape-sandbox allow-downloads';
  iframe.referrerPolicy = 'no-referrer';
  // Browser features are granted per demo: copy buttons everywhere, plus whatever a demo's
  // inputs need (the X-ray demo can take a photo or a pasted image).
  iframe.allow = ['clipboard-write', frame.dataset.allow].filter(Boolean).join('; ');
  iframe.src = frame.dataset.src;
  iframe.title = 'Live demo';
  frame.replaceChildren(iframe);
});

document.querySelectorAll('.card').forEach(card => {
  card.addEventListener('click', () => openProject(card.dataset.project));
  card.addEventListener('keydown', e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openProject(card.dataset.project); }
  });
});

if (location.hash.startsWith('#project-')) openProject(location.hash.slice(9));


// Contact form (Web3Forms)
(() => {
  const form = document.getElementById('contactForm');
  const sendBtn = document.getElementById('sendBtn');
  const clearBtn = document.getElementById('clearBtn');
  const toast = document.getElementById('toast');
  let toastTimer = 0;
  const notify = (msg, ok = true) => {
    toast.querySelector('.msg').textContent = msg;
    toast.classList.toggle('ok', ok); toast.classList.toggle('bad', !ok);
    toast.querySelector('.ico svg path').setAttribute('d', ok ? 'M5 13l4 4L19 7' : 'M6 6l12 12M18 6L6 18');
    toast.classList.add('show');
    clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('show'), ok ? 5000 : 7000);
  };
  const fields = [...form.querySelectorAll('.field')];
  const validate = () => {
    let first = null;
    fields.forEach(f => {
      const c = f.querySelector('input, textarea, select');
      const bad = c && !c.checkValidity();
      f.classList.toggle('invalid', !!bad);
      if (bad && !first) first = c;
    });
    if (first) first.focus();
    return !first;
  };
  form.addEventListener('input', e => { const f = e.target.closest('.field'); if (f && f.classList.contains('invalid') && e.target.checkValidity()) f.classList.remove('invalid'); });
  // Message placeholder follows the chosen topic
  const topic = form.querySelector('#f-topic'), msg = form.querySelector('#f-msg');
  const syncTopic = () => { msg.placeholder = topic.selectedOptions[0].dataset.placeholder; };
  topic.addEventListener('change', syncTopic);

  // hCaptcha via Web3Forms' free shared sitekey. Web3Forms checks the token server
  // side once hCaptcha is switched on in its dashboard, which stops scripts that post
  // straight to the API. Loaded only when the form is near, so most visitors never
  // fetch it, and redrawn to match the site theme until it is solved.
  const captchaBox = document.getElementById('captcha'), captchaErr = document.getElementById('captchaErr');
  const CAPTCHA_MSG = captchaErr.textContent, EXPIRED_MSG = 'The spam check expired. Please tick it again.';
  let widget = null, widgetSize = '', captchaFailed = false, challengeOpen = false, captchaScript = null;
  // The normal widget is a fixed 303px wide; use the compact one where that would overflow.
  const fitSize = () => (captchaBox.clientWidth < 304 ? 'compact' : 'normal');
  const captchaToken = () => (widget !== null && window.hcaptcha ? hcaptcha.getResponse(widget) : '');
  const resetCaptcha = () => { if (widget !== null && window.hcaptcha) hcaptcha.reset(widget); };
  // Replacing the message's content each time makes role="alert" announce it again.
  const captchaError = (...parts) => { captchaErr.replaceChildren(...parts); captchaBox.classList.add('invalid'); };
  const failCaptcha = () => {
    captchaFailed = true; challengeOpen = false;
    const mail = Object.assign(document.createElement('a'), { href: 'mailto:sabin.shamsul@gmail.com', className: 'inline-link', textContent: 'sabin.shamsul@gmail.com' });
    captchaError('The spam check could not load (an ad blocker may be stopping it). Press Send to try again, reload the page, or email me at ', mail, '.');
  };
  const renderCaptcha = () => {
    const el = document.getElementById('captchaWidget');
    try {
      if (widget !== null) hcaptcha.remove(widget);
      widgetSize = fitSize();
      widget = hcaptcha.render(el, {
        sitekey: '50b2fe65-b00b-4b9e-ad62-3ba471098be2',
        theme: document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark',
        size: widgetSize,
        callback: () => { challengeOpen = false; captchaBox.classList.remove('invalid'); },
        'expired-callback': () => captchaError(EXPIRED_MSG),
        'chalexpired-callback': () => { challengeOpen = false; captchaError(EXPIRED_MSG); },
        'open-callback': () => { challengeOpen = true; },
        'close-callback': () => { challengeOpen = false; },
        'error-callback': failCaptcha
      });
      // A fresh widget replaces any earlier "could not load" text with the plain prompt.
      if (captchaBox.classList.contains('invalid')) captchaError(CAPTCHA_MSG);
    } catch (err) { widget = null; failCaptcha(); }
  };
  window.onHcaptchaLoad = renderCaptcha;
  const loadCaptcha = () => {
    if (loadCaptcha.done) return; loadCaptcha.done = true;
    captchaScript = document.createElement('script');
    captchaScript.src = 'https://js.hcaptcha.com/1/api.js?render=explicit&onload=onHcaptchaLoad&recaptchacompat=off';
    captchaScript.async = true;
    captchaScript.onerror = failCaptcha;
    document.head.appendChild(captchaScript);
  };
  // After a failure, Send tries again: redraw if the API is here, otherwise fetch it afresh.
  const retryCaptcha = () => {
    captchaFailed = false;
    if (window.hcaptcha) renderCaptcha();
    else { if (captchaScript) captchaScript.remove(); loadCaptcha.done = false; loadCaptcha(); }
  };
  new IntersectionObserver((entries, io) => {
    if (entries.some(en => en.isIntersecting)) { loadCaptcha(); io.disconnect(); }
  }, { rootMargin: '600px 0px' }).observe(form);
  form.addEventListener('focusin', loadCaptcha);
  // Redraw for a new theme or width, but never throw away a solved or open check.
  const redraw = () => { if (window.hcaptcha && widget !== null && !challengeOpen && !captchaToken()) renderCaptcha(); };
  new MutationObserver(redraw).observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  new ResizeObserver(() => { if (widget !== null && fitSize() !== widgetSize) redraw(); }).observe(captchaBox);
  form.addEventListener('reset', () => setTimeout(syncTopic));
  clearBtn.addEventListener('click', () => { form.reset(); fields.forEach(f => f.classList.remove('invalid')); if (!captchaFailed) captchaBox.classList.remove('invalid'); resetCaptcha(); form.querySelector('#f-title').focus(); });

  form.addEventListener('submit', async e => {
    e.preventDefault();
    if (!validate()) return;
    const data = Object.fromEntries(new FormData(form).entries());
    const sent = (name, mail) => notify(`Thanks ${name}, your message has been sent. I will reply to ${mail}.`);
    // Honeypot: people never see this box, bots that fill every field tick it.
    // Show the normal success message so the bot learns nothing, but send nothing.
    if ('botcheck' in data) {
      sent(data.first_name, data.email);
      form.reset();
      return;
    }
    if (!data.access_key || data.access_key.startsWith('__')) { notify('The form is not connected yet. Please email me directly instead.', false); return; }
    // Clean what leaves the browser: dropdowns must be one of their own options,
    // one-line fields lose line breaks and control characters (they end up in email
    // headers), and every field is length-capped. Web3Forms still sees whatever a
    // script sends it directly, so this guards the page, not the API.
    // Caps count characters, not UTF-16 units, so an emoji is never cut in half.
    const cap = (s, n) => { const a = [...s]; return a.length > n ? a.slice(0, n).join('') : s; };
    // One-line fields: control characters, line/paragraph separators, and invisible or
    // bidi characters that can spoof a subject all become a space. ZWJ and ZWNJ
    // (U+200C/U+200D) stay, since emoji sequences and some names need them.
    const oneLine = (v, max) => cap(String(v || '').replace(/[\p{Cc}\p{Zl}\p{Zp}\u00ad\u061c\u200b\u200e\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]+/gu, ' ').replace(/\s+/g, ' ').trim(), max).trim();
    // Message: keep tabs and line breaks, drop other control and bidi characters.
    const msgClean = v => cap(String(v || '').replace(/(?![\t\n\r])[\p{Cc}\u202a-\u202e\u2066-\u2069\ufeff]/gu, '').trim(), 5000).trim();
    const pick = (sel, v, fallback) => [...form.querySelector(sel).options].some(o => o.value === v) ? v : fallback;
    const title = pick('#f-title', data.title, '');
    const topicVal = pick('#f-topic', data.topic, 'Something else');
    const code = pick('#f-code', data.country_code, '');
    const first = oneLine(data.first_name, 50), last = oneLine(data.last_name, 50);
    const email = oneLine(data.email, 254), phoneNo = oneLine(data.phone_number, 20);
    const message = msgClean(data.message);
    // Validation saw the raw values; a name of only spaces or a message padded with
    // control characters can still come out empty or too short after cleaning.
    const empty = [['#f-first', first], ['#f-last', last], ['#f-msg', message.length >= 10]].find(([, ok]) => !ok);
    if (empty) { const el = form.querySelector(empty[0]); el.closest('.field').classList.add('invalid'); el.focus(); return; }
    // Spam check must be solved; its token is single-use, so it is reset after every attempt.
    const token = captchaToken();
    if (!token) {
      if (captchaFailed) retryCaptcha(); else loadCaptcha();
      if (!captchaFailed) captchaError(CAPTCHA_MSG);
      captchaBox.scrollIntoView({ block: 'center', behavior: reduceMotion ? 'auto' : 'smooth' });
      (captchaBox.querySelector('iframe') || captchaErr).focus({ preventScroll: true });
      return;
    }
    const fullName = [title, first, last].filter(Boolean).join(' ');
    const phone = phoneNo ? `${code} ${phoneNo}`.trim() : 'not provided';
    const payload = {
      access_key: data.access_key,
      subject: `[${topicVal}] Portfolio message from ${fullName}`,
      from_name: fullName,
      name: fullName,
      email,
      replyto: email,
      phone,
      topic: topicVal,
      message,
      'h-captcha-response': token
    };
    sendBtn.disabled = true; sendBtn.classList.add('sending'); sendBtn.querySelector('.lbl').textContent = 'Sending';
    try {
      const res = await fetch('https://api.web3forms.com/submit', { method: 'POST', headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' }, body: JSON.stringify(payload) });
      const out = await res.json().catch(() => ({}));
      if (res.ok && out.success) {
        sent(first, email);
        form.reset();
      } else {
        notify(out.message || 'Something went wrong sending your message. Please try again or email me directly.', false);
      }
    } catch (err) {
      notify('Could not reach the mail service. Check your connection or email me directly.', false);
    } finally {
      sendBtn.disabled = false; sendBtn.classList.remove('sending'); sendBtn.querySelector('.lbl').textContent = 'Send message';
      resetCaptcha();
    }
  });
})();

// ─── Image lightbox ───
const LB = (() => {
  const dlg = document.getElementById('lightbox');
  const stage = document.getElementById('lbStage');
  const img = document.getElementById('lbImg');
  const strip = document.getElementById('lbStrip');
  const capEl = document.getElementById('lbCaption');
  const subEl = document.getElementById('lbSub');
  const countEl = document.getElementById('lbCount');
  const zoomEl = document.getElementById('lbZoom');
  const playBtn = document.getElementById('lbPlay');
  const frame = document.querySelector('.lb-frame');
  const toggleBtn = document.getElementById('lbToggle');
  const prevBtn = document.getElementById('lbPrev');
  const nextBtn = document.getElementById('lbNext');
  const SLIDE_MS = 3500;

  let items = [], idx = 0, scale = 1, tx = 0, ty = 0, timer = 0, opener = null;
  const MIN = 0.25, MAX = 6;

  const apply = () => {
    img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
    zoomEl.textContent = Math.round(scale * 100) + '%';
    stage.classList.toggle('zoomed', scale > 1);
    document.getElementById('lbOut').disabled = scale <= MIN + 0.001;
    document.getElementById('lbIn').disabled = scale >= MAX;
  };
  const resetZoom = () => { scale = 1; tx = 0; ty = 0; apply(); };
  const clampPan = () => {
    const r = img.getBoundingClientRect(), s = stage.getBoundingClientRect();
    const ox = Math.max(0, (r.width - s.width) / 2 + 24), oy = Math.max(0, (r.height - s.height) / 2 + 24);
    tx = Math.min(ox, Math.max(-ox, tx)); ty = Math.min(oy, Math.max(-oy, ty));
  };
  const zoomTo = (next, cx, cy) => {
    const prev = scale;
    scale = Math.min(MAX, Math.max(MIN, next));
    if (scale <= 1) { tx = 0; ty = 0; }
    else if (cx != null) {
      const r = stage.getBoundingClientRect();
      const dx = cx - r.left - r.width / 2, dy = cy - r.top - r.height / 2;
      const k = scale / prev;
      tx = dx - (dx - tx) * k; ty = dy - (dy - ty) * k;
      clampPan();
    }
    apply();
  };

  const show = (i, instant) => {
    idx = (i + items.length) % items.length;
    const it = items[idx];
    img.src = it.src; img.alt = it.caption || '';
    capEl.textContent = it.caption || '';
    subEl.textContent = it.project || '';
    countEl.textContent = `${idx + 1} / ${items.length}`;
    resetZoom();
    [...strip.children].forEach((t, n) => {
      const on = n === idx;
      t.setAttribute('aria-current', on ? 'true' : 'false');
      if (on) t.scrollIntoView({ inline: 'center', block: 'nearest', behavior: instant ? 'auto' : 'smooth' });
    });
    const solo = items.length < 2;
    prevBtn.hidden = nextBtn.hidden = solo;
    playBtn.hidden = solo;
    if (playBtn.classList.contains('playing')) restartBar();
  };
  const restartBar = () => {
    const bar = playBtn.querySelector('.lb-progress');
    bar.style.animation = 'none'; void bar.offsetWidth; bar.style.animation = '';
  };
  const stop = () => { clearInterval(timer); timer = 0; playBtn.classList.remove('playing'); playBtn.setAttribute('aria-label', 'Start slideshow'); };
  const play = () => {
    if (items.length < 2) return;
    stop(); playBtn.classList.add('playing'); playBtn.setAttribute('aria-label', 'Pause slideshow');
    restartBar();
    timer = setInterval(() => show(idx + 1), SLIDE_MS);
  };

  const setInfo = (visible, remember) => {
    frame.classList.toggle('info-hidden', !visible);
    toggleBtn.setAttribute('aria-expanded', String(visible));
    toggleBtn.setAttribute('aria-label', visible ? 'Hide caption and thumbnails' : 'Show caption and thumbnails');
    toggleBtn.title = (visible ? 'Hide' : 'Show') + ' caption (C)';
    if (remember) { try { localStorage.setItem('lbInfo', visible ? '1' : '0'); } catch (e) {} }
    requestAnimationFrame(() => { clampPan(); apply(); });
  };
  toggleBtn.addEventListener('click', () => setInfo(frame.classList.contains('info-hidden'), true));

  const open = (list, start, projectTitle) => {
    items = list.map(x => ({ ...x, project: projectTitle }));
    strip.innerHTML = items.map((it, n) =>
      `<button type="button" class="lb-thumb" data-n="${n}" aria-label="Go to image ${n + 1}"><img src="${it.src}" alt=""></button>`).join('');
    opener = document.activeElement;
    dlg.showModal();
    document.documentElement.style.setProperty('--slide-ms', SLIDE_MS + 'ms');
    let pref = '1';
    try { pref = localStorage.getItem('lbInfo') ?? '1'; } catch (e) {}
    setInfo(pref !== '0', false);
    show(start, true);
  };

  strip.addEventListener('click', e => { const b = e.target.closest('.lb-thumb'); if (b) { stop(); show(+b.dataset.n); } });
  prevBtn.addEventListener('click', () => { stop(); show(idx - 1); });
  nextBtn.addEventListener('click', () => { stop(); show(idx + 1); });
  playBtn.addEventListener('click', () => playBtn.classList.contains('playing') ? stop() : play());
  document.getElementById('lbIn').addEventListener('click', () => zoomTo(scale * 1.25));
  document.getElementById('lbOut').addEventListener('click', () => zoomTo(scale / 1.25));
  document.getElementById('lbReset').addEventListener('click', resetZoom);
  document.getElementById('lbClose').addEventListener('click', () => dlg.close());

  dlg.addEventListener('close', () => { stop(); img.removeAttribute('src'); if (opener) opener.focus(); });
  dlg.addEventListener('click', e => { if (e.target === dlg || e.target === stage) dlg.close(); });

  img.addEventListener('dblclick', e => { stop(); zoomTo(scale > 1 ? 1 : 2.5, e.clientX, e.clientY); });
  img.addEventListener('click', e => { if (scale === 1) { stop(); zoomTo(2.5, e.clientX, e.clientY); } });
  stage.addEventListener('wheel', e => {
    if (!dlg.open) return;
    e.preventDefault(); stop();
    zoomTo(scale * (e.deltaY < 0 ? 1.18 : 1 / 1.18), e.clientX, e.clientY);
  }, { passive: false });

  // Pointer drag to pan, swipe to change image
  let down = null, pts = new Map(), startDist = 0, startScale = 1;
  img.addEventListener('pointerdown', e => {
    pts.set(e.pointerId, e);
    if (pts.size === 2) { const [a, b] = [...pts.values()]; startDist = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); startScale = scale; return; }
    down = { x: e.clientX, y: e.clientY, tx, ty, t: Date.now() };
    img.setPointerCapture(e.pointerId); img.classList.add('dragging');
  });
  img.addEventListener('pointermove', e => {
    if (!pts.has(e.pointerId)) return;
    pts.set(e.pointerId, e);
    if (pts.size === 2) {
      const [a, b] = [...pts.values()];
      const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
      if (startDist) zoomTo(startScale * (d / startDist), (a.clientX + b.clientX) / 2, (a.clientY + b.clientY) / 2);
      return;
    }
    if (!down || scale === 1) return;
    tx = down.tx + (e.clientX - down.x); ty = down.ty + (e.clientY - down.y);
    clampPan(); apply();
  });
  const endPointer = e => {
    pts.delete(e.pointerId);
    if (pts.size < 2) startDist = 0;
    img.classList.remove('dragging');
    if (down && scale === 1 && items.length > 1) {
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) { stop(); show(idx + (dx < 0 ? 1 : -1)); }
    }
    down = null;
  };
  img.addEventListener('pointerup', endPointer);
  img.addEventListener('pointercancel', endPointer);

  dlg.addEventListener('keydown', e => {
    const k = e.key;
    if (k === 'ArrowRight') { e.preventDefault(); stop(); show(idx + 1); }
    else if (k === 'ArrowLeft') { e.preventDefault(); stop(); show(idx - 1); }
    else if (k === '+' || k === '=') { e.preventDefault(); zoomTo(scale * 1.25); }
    else if (k === '-') { e.preventDefault(); zoomTo(scale / 1.25); }
    else if (k === '0') { e.preventDefault(); resetZoom(); }
    else if (k === ' ') { e.preventDefault(); playBtn.click(); }
    else if (k === 'c' || k === 'C') { e.preventDefault(); toggleBtn.click(); }
  });
  addEventListener('resize', () => { if (dlg.open) { clampPan(); apply(); } });
  return { open };
})();

// Active nav link
const navItems = document.querySelectorAll('.nav-links a');
const navObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    navItems.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + entry.target.id));
  });
}, { rootMargin: '-40% 0px -55% 0px' });
document.querySelectorAll('main section[id]').forEach(s => navObserver.observe(s));
