// Config for md-to-pdf: clean A4 styling, Mermaid (built-in) + KaTeX (injected).
module.exports = {
  stylesheet: [
    'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
  ],
  css: `
    body {
      font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      font-size: 11pt;
      line-height: 1.5;
      color: #1a1a1a;
      max-width: 100%;
    }
    h1 { font-size: 22pt; color: #0b3d5c; border-bottom: 3px solid #0b3d5c; padding-bottom: 6px; }
    h2 { font-size: 15pt; color: #0b3d5c; margin-top: 22px; border-bottom: 1px solid #cbd5e0; padding-bottom: 3px; }
    h3 { font-size: 12.5pt; color: #14507a; margin-top: 16px; }
    h4 { font-size: 11pt; color: #14507a; }
    code {
      font-family: "Cascadia Code", "Fira Code", Consolas, monospace;
      font-size: 9.5pt;
      background: #f0f3f6;
      padding: 1px 4px;
      border-radius: 3px;
    }
    pre code { background: none; padding: 0; }
    pre {
      background: #f6f8fa;
      border: 1px solid #e1e4e8;
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 9pt;
      overflow-x: auto;
    }
    table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 10px 0; }
    th, td { border: 1px solid #cbd5e0; padding: 5px 8px; text-align: left; vertical-align: top; }
    th { background: #0b3d5c; color: #fff; }
    tr:nth-child(even) td { background: #f4f7fa; }
    blockquote { border-left: 4px solid #0b3d5c; margin: 10px 0; padding: 4px 14px; color: #444; background: #f7f9fb; }
    hr { border: none; border-top: 1px solid #cbd5e0; margin: 18px 0; }
    a { color: #14507a; text-decoration: none; }
    .mermaid { text-align: center; margin: 14px 0; }
    .katex-display { margin: 14px 0; }
  `,
  body_class: ['markdown-body'],
  marked_options: { headerIds: true, smartypants: true },
  script: [
    { url: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js' },
    { url: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js' },
    {
      content: `
        window.addEventListener('load', function () {
          if (window.renderMathInElement) {
            renderMathInElement(document.body, {
              delimiters: [
                { left: '$$', right: '$$', display: true },
                { left: '$', right: '$', display: false }
              ],
              throwOnError: false
            });
          }
        });
      `,
    },
  ],
  pdf_options: {
    format: 'A4',
    margin: { top: '18mm', bottom: '18mm', left: '16mm', right: '16mm' },
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate:
      '<div style="width:100%; font-size:8pt; color:#888; text-align:center;">Young Families Engagement Model — <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
  },
  launch_options: { args: ['--no-sandbox'] },
};
