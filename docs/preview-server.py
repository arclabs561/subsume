#!/usr/bin/env python3
"""
Live Markdown Preview Server with KaTeX Math Rendering

Usage:
    python3 preview-server.py [port] [file.md]

Examples:
    python3 preview-server.py                    # Preview on port 8000, shows file selector
    python3 preview-server.py 3000               # Preview on port 3000
    python3 preview-server.py 8000 MATHEMATICAL_FOUNDATIONS.md  # Preview specific file
"""

import http.server
import socketserver
import urllib.parse
import os
import sys
import json
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
INITIAL_FILE = sys.argv[2] if len(sys.argv) > 2 else None

DOCS_DIR = Path(__file__).parent
MARKDOWN_FILES = sorted([f.name for f in DOCS_DIR.glob("*.md") if f.name != "README_KATEX.md"])

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} - Live Preview</title>
    
    <!-- KaTeX CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
    
    <!-- Marked.js for markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
    
    <!-- KaTeX JavaScript -->
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlG8jCC0C9WXZB6XH0jZqKk7xK3I2VqQnBk3xSB7f4F0v3f5f5f5f5f5f" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
    
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            background: #f6f8fa;
            color: #24292e;
        }}
        
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        .header {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin-bottom: 15px;
            font-size: 24px;
        }}
        
        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .file-selector {{
            flex: 1;
            min-width: 200px;
        }}
        
        .file-selector select {{
            width: 100%;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #d1d5da;
            border-radius: 4px;
            background: white;
        }}
        
        .status {{
            padding: 8px 12px;
            background: #28a745;
            color: white;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 500;
        }}
        
        .status.reloading {{
            background: #ffc107;
        }}
        
        .content {{
            background: white;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 40px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            min-height: 400px;
        }}
        
        /* KaTeX styling */
        .katex {{
            font-size: 1.1em;
        }}
        
        .katex-display {{
            margin: 1.5em 0;
            overflow-x: auto;
            overflow-y: hidden;
        }}
        
        /* Markdown styling */
        h1, h2, h3, h4, h5, h6 {{
            margin-top: 24px;
            margin-bottom: 16px;
            font-weight: 600;
            line-height: 1.25;
        }}
        
        h1 {{ font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h2 {{ font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }}
        h3 {{ font-size: 1.25em; }}
        
        code {{
            background: rgba(27,31,35,0.05);
            border-radius: 3px;
            padding: 0.2em 0.4em;
            font-size: 85%;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }}
        
        pre {{
            background: #f6f8fa;
            border-radius: 6px;
            padding: 16px;
            overflow: auto;
            line-height: 1.45;
        }}
        
        pre code {{
            background: none;
            padding: 0;
        }}
        
        blockquote {{
            padding: 0 1em;
            color: #6a737d;
            border-left: 0.25em solid #dfe2e5;
            margin: 0;
        }}
        
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
        }}
        
        table th, table td {{
            border: 1px solid #dfe2e5;
            padding: 6px 13px;
        }}
        
        table th {{
            background: #f6f8fa;
            font-weight: 600;
        }}
        
        a {{
            color: #0366d6;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        .loading {{
            text-align: center;
            padding: 40px;
            color: #6a737d;
        }}
        
        .error {{
            background: #ffeef0;
            border: 1px solid #f97583;
            border-radius: 6px;
            padding: 16px;
            color: #86181d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Live Markdown Preview with KaTeX</h1>
            <div class="controls">
                <div class="file-selector">
                    <select id="doc-select">
                        <option value="">-- Select a document --</option>
                        {file_options}
                    </select>
                </div>
                <div class="status" id="status">Ready</div>
            </div>
        </div>
        
        <div class="content" id="content">
            <div class="loading">Select a document from the dropdown above to preview it with KaTeX math rendering.</div>
        </div>
    </div>
    
    <script>
        const docSelect = document.getElementById('doc-select');
        const content = document.getElementById('content');
        const status = document.getElementById('status');
        let currentFile = null;
        let reloadInterval = null;
        
        // Configure KaTeX options
        const katexOptions = {{
            delimiters: [
                {{left: "\\\\[", right: "\\\\]", display: true}},
                {{left: "\\\\(", right: "\\\\)", display: false}},
                {{left: "$$", right: "$$", display: true}},
                {{left: "$", right: "$", display: false}}
            ],
            throwOnError: false,
            strict: false,
            trust: true
        }};
        
        // Configure Marked options
        marked.setOptions({{
            breaks: true,
            gfm: true,
            headerIds: true,
            mangle: false
        }});
        
        async function loadDocument(filename) {{
            if (!filename) {{
                content.innerHTML = '<div class="loading">Select a document from the dropdown above to preview it with KaTeX math rendering.</div>';
                currentFile = null;
                if (reloadInterval) {{
                    clearInterval(reloadInterval);
                    reloadInterval = null;
                }}
                return;
            }}
            
            status.textContent = 'Loading...';
            status.className = 'status reloading';
            content.innerHTML = '<div class="loading">Loading...</div>';
            
            try {{
                const response = await fetch(`/api/file?name=${{encodeURIComponent(filename)}}`);
                if (!response.ok) {{
                    throw new Error(`Failed to load ${{filename}}: ${{response.statusText}}`);
                }}
                
                const data = await response.json();
                const markdown = data.content;
                
                // Convert markdown to HTML
                const html = marked.parse(markdown);
                
                // Update content
                content.innerHTML = html;
                
                // Render math with KaTeX
                renderMathInElement(content, katexOptions);
                
                currentFile = filename;
                status.textContent = 'Live';
                status.className = 'status';
                
                // Start auto-reload if not already running
                if (!reloadInterval) {{
                    reloadInterval = setInterval(() => {{
                        if (currentFile) {{
                            loadDocument(currentFile);
                        }}
                    }}, 2000); // Reload every 2 seconds
                }}
                
            }} catch (error) {{
                content.innerHTML = `
                    <div class="error">
                        <strong>Error loading document:</strong><br>
                        ${{error.message}}
                    </div>
                `;
                status.textContent = 'Error';
                status.className = 'status';
            }}
        }}
        
        docSelect.addEventListener('change', (e) => {{
            loadDocument(e.target.value);
        }});
        
        // Load initial file if specified
        {initial_load}
    </script>
</body>
</html>"""

class DocsHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS_DIR), **kwargs)
    
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        if parsed_path.path == '/':
            # Serve main preview page
            file_options = '\n'.join([f'                        <option value="{f}">{f}</option>' for f in MARKDOWN_FILES])
            initial_load = f'docSelect.value = "{INITIAL_FILE}"; loadDocument("{INITIAL_FILE}");' if INITIAL_FILE else ''
            
            html = HTML_TEMPLATE.format(
                title="Documentation Preview",
                file_options=file_options,
                initial_load=initial_load
            )
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
            return
        
        elif parsed_path.path == '/api/file':
            # API endpoint to get file content
            query = urllib.parse.parse_qs(parsed_path.query)
            filename = query.get('name', [None])[0]
            
            if not filename or filename not in MARKDOWN_FILES:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'File not found'}).encode())
                return
            
            file_path = DOCS_DIR / filename
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'content': content}).encode())
                return
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                return
        
        # Fall back to serving files normally
        return super().do_GET()

def main():
    with socketserver.TCPServer(("", PORT), DocsHandler) as httpd:
        print(f"🚀 Live Preview Server running at http://localhost:{PORT}")
        print(f"📁 Serving from: {DOCS_DIR}")
        if INITIAL_FILE:
            print(f"📄 Initial file: {INITIAL_FILE}")
        print(f"\n💡 Tip: Edit markdown files and they'll auto-reload every 2 seconds")
        print(f"   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")

if __name__ == "__main__":
    main()

