DocETL Reveal.js Slides — Style Guide

Purpose
- Ensure the DocETL VLDB deck is readable at conference viewing distances while staying on-brand with UC Berkeley colors and Inter typography.

Typography
- Font family: Inter (loaded via rsms.me).
- Body size: 34px target (30–34pt minimum).
- Titles: H1 56px, H2 44px, H3 38px.
- Line height: ~1.2–1.3 for paragraphs/lists.
- Avoid text smaller than 28px except footnotes.

Colors (CSS variables in styles.css)
- --berkeley-blue: #003262 (primary headings, accents)
- --berkeley-gold: #FDB515 (accents, highlights)
- --founders-rock: #3B7EA1 (links)
- --medalist: #C4820E (subtitles, callouts)
- Background: --off-white #FAFAFA for slides; white for print.

Layout
- Title slides: Large title, short subtitle, authors, and logos.
- Body slides: 3–6 bullets max; break content across slides.
- Use the persistent footer logos as-is; don’t crowd the bottom edge.
- Images: prefer 60–80% width, centered or aligned to grid.
- Callouts: use the `.callout` class for emphasized statements.

Logos & Branding
- Use logos on the title and closing slides; avoid a persistent footer.
- Keep logos on light background; avoid recoloring.

Accessibility & Readability
- High contrast: headings in Berkeley Blue; avoid light-on-light.
- Bullet density: 6–8 words per line, 3–6 bullets per slide.
- Use visuals (figures/tables) over text when possible.

Authoring Notes
- Edit slide content in index.html. Use standard Reveal.js sections.
- Keep lists concise; split complex concepts across multiple slides.
- Prefer verbs first in bullets; avoid full sentences.
- For code, keep to 8–12 lines per slide; bump font size if needed.

Reveal.js Usage
- Keyboard: arrows to navigate; ‘s’ for speaker notes.
- Slide numbers enabled; URL hash reflects current slide.
- Print to PDF: append `?print-pdf` to the URL, then print.

Content Checklist per Slide
- One clear idea (title says the idea).
- 3–6 concise bullets or one figure.
- No more than ~30 words total.
- Sufficient whitespace; consistent alignment.

How to Add Figures
- Place assets in `images/` and reference with `<img>`.
- Add `alt` text. Keep height consistent across related slides.

Custom Classes
- `.callout` for emphasized blocks.
- `.small` or `<small>` for brief footnotes (≥ 24–28px).

Attribution
- Paper: arXiv:2410.12189 “DocETL: Agentic Query Rewriting and Evaluation for Complex Document Processing”.
