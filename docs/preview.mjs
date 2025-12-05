#!/usr/bin/env node
/**
 * Live Markdown Preview with KaTeX
 * 
 * Usage:
 *   npx --yes -p markdown-it -p markdown-it-katex node docs/preview.mjs [file.md]
 * 
 * Or install dependencies and run:
 *   npm install markdown-it markdown-it-katex
 *   node docs/preview.mjs [file.md]
 */

import http from 'http';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { watch } from 'fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const DOCS_DIR = __dirname;

// Try to import markdown-it (will fail if not installed, that's ok)
let MarkdownIt, markdownItKatex;
try {
  const mdModule = await import('markdown-it');
  MarkdownIt = mdModule.default;
  const katexModule = await import('markdown-it-katex');
  markdownItKatex = katexModule.default;
} catch (e) {
  console.error('❌ Missing dependencies. Install with:');
  console.error('   npm install markdown-it markdown-it-katex');
  console.error('   or run: npx --yes -p markdown-it -p markdown-it-katex node docs/preview.mjs');
  process.exit(1);
}

const PORT = process.env.PORT || 8000;
const targetFile = process.argv[2] || null;

// Find all markdown files
const markdownFiles = fs.readdirSync(DOCS_DIR)
  .filter(f => f.endsWith('.md') && f !== 'README_KATEX.md' && f !== 'README_PREVIEW.md')
  .sort();

const md = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: true,
  breaks: true
}).use(markdownItKatex, { throwOnError: false, errorColor: '#cc0000' });

const HTML_TEMPLATE = (content, files, currentFile) => `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${currentFile || 'Markdown Preview'} - KaTeX</title>
    
    <!-- KaTeX CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
    
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            background: #f6f8fa;
            color: #24292e;
        }
        
        .container {
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        .header h1 {
            margin-bottom: 15px;
            font-size: 24px;
        }
        
        .controls {
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        
        .file-selector {
            flex: 1;
            min-width: 200px;
        }
        
        .file-selector select {
            width: 100%;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #d1d5da;
            border-radius: 4px;
            background: white;
        }
        
        .status {
            padding: 8px 12px;
            background: #28a745;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .content {
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 40px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            min-height: 400px;
        }
        
        /* KaTeX styling */
        .katex {
            font-size: 1.1em;
        }
        
        .katex-display {
            margin: 1.5em 0;
            overflow-x: auto;
            overflow-y: hidden;
        }
        
        /* Markdown styling */
        h1, h2, h3, h4, h5, h6 {
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }
        
        h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
        h3 { font-size: 1.25em; }
        
        code {
            background: rgba(27,31,35,0.05);
            border-radius: 3px;
            padding: 0.2em 0.4em;
            font-size: 85%;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }
        
        pre {
            background: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
            line-height: 1.45;
        }
        
        pre code {
            background: none;
            padding: 0;
        }
        
        blockquote {
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }
        
        table th, table td {
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }
        
        table th {
            background: #f6f8fa;
            font-weight: 600;
        }
        
        a {
            color: #0366d6;
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Live Markdown Preview with KaTeX</h1>
            <div class="controls">
                <div class="file-selector">
                    <select id="doc-select" onchange="window.location.href='/?file=' + this.value">
                        <option value="">-- Select a document --</option>
                        ${files.map(f => `<option value="${f}" ${f === currentFile ? 'selected' : ''}>${f}</option>`).join('\n                        ')}
                    </select>
                </div>
                <div class="status" id="status">Live</div>
            </div>
        </div>
        
        <div class="content">
            ${content}
        </div>
    </div>
    
    <script>
        // Auto-reload every 2 seconds
        let lastModified = ${Date.now()};
        setInterval(async () => {
            try {
                const response = await fetch('/api/check?file=${currentFile || ''}');
                const data = await response.json();
                if (data.modified > lastModified) {
                    location.reload();
                }
            } catch (e) {
                // Ignore errors
            }
        }, 2000);
    </script>
</body>
</html>`;

function renderMarkdown(filePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    return md.render(content);
  } catch (e) {
    return `<div style="color: #cc0000; padding: 20px; background: #ffeef0; border-radius: 6px;">
      <strong>Error loading file:</strong><br>${e.message}
    </div>`;
  }
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  
  if (url.pathname === '/api/check') {
    const file = url.searchParams.get('file');
    if (file) {
      const filePath = path.join(DOCS_DIR, file);
      try {
        const stats = fs.statSync(filePath);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ modified: stats.mtimeMs }));
        return;
      } catch (e) {
        res.writeHead(404);
        res.end(JSON.stringify({ error: 'File not found' }));
        return;
      }
    }
    res.writeHead(400);
    res.end(JSON.stringify({ error: 'No file specified' }));
    return;
  }
  
  if (url.pathname === '/' || url.pathname === '') {
    const file = url.searchParams.get('file') || targetFile;
    const currentFile = file && markdownFiles.includes(file) ? file : null;
    
    let content = '<div style="text-align: center; padding: 40px; color: #6a737d;">Select a document from the dropdown above to preview it with KaTeX math rendering.</div>';
    
    if (currentFile) {
      const filePath = path.join(DOCS_DIR, currentFile);
      content = renderMarkdown(filePath);
    }
    
    const html = HTML_TEMPLATE(content, markdownFiles, currentFile);
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end(html);
    return;
  }
  
  // Serve static files (if any)
  res.writeHead(404);
  res.end('Not found');
});

server.listen(PORT, () => {
  console.log(`🚀 Live Preview Server running at http://localhost:${PORT}`);
  console.log(`📁 Serving from: ${DOCS_DIR}`);
  if (targetFile) {
    console.log(`📄 Initial file: ${targetFile}`);
  }
  console.log(`\n💡 Tip: Edit markdown files and they'll auto-reload every 2 seconds`);
  console.log(`   Press Ctrl+C to stop\n`);
  
  // Watch for file changes
  if (targetFile) {
    const filePath = path.join(DOCS_DIR, targetFile);
    watch(filePath, (eventType) => {
      if (eventType === 'change') {
        console.log(`📝 File changed: ${targetFile}`);
      }
    });
  }
});

process.on('SIGINT', () => {
  console.log('\n👋 Server stopped');
  process.exit(0);
});

