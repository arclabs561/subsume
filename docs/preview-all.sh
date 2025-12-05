#!/bin/bash
# Preview all markdown documentation files with GitHub-style rendering
# Similar to gh-markdown-preview: https://github.com/yusukebe/gh-markdown-preview
#
# Usage:
#   ./preview-all.sh                    # Preview on default port 3333
#   ./preview-all.sh 8000              # Preview on port 8000
#   ./preview-all.sh --file README.md  # Preview specific file

set -e

PORT=3333
TARGET_FILE=""
AUTO_OPEN=true
LIVE_RELOAD=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --file|-f)
            TARGET_FILE="$2"
            shift 2
            ;;
        --port|-p)
            PORT="$2"
            shift 2
            ;;
        --no-open)
            AUTO_OPEN=false
            shift
            ;;
        --no-reload)
            LIVE_RELOAD=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --file, -f FILE    Preview specific file"
            echo "  --port, -p PORT    Port number (default: 3333)"
            echo "  --no-open          Don't auto-open browser"
            echo "  --no-reload        Disable live reload"
            echo "  --help, -h         Show this help"
            exit 0
            ;;
        *)
            if [[ "$1" =~ ^[0-9]+$ ]]; then
                PORT="$1"
            else
                TARGET_FILE="$1"
            fi
            shift
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$SCRIPT_DIR"

# Check if gh CLI is available
USE_GH=false
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    echo "✅ Using GitHub API for rendering (via gh CLI)"
    USE_GH=true
else
    echo "⚠️  GitHub CLI not authenticated, using local rendering with KaTeX"
    echo "   Install 'gh' CLI and run 'gh auth login' for GitHub-style rendering"
fi

# Find all markdown files
MARKDOWN_FILES=($(find "$DOCS_DIR" -maxdepth 1 -name "*.md" -type f | sort | xargs -n1 basename))

if [ ${#MARKDOWN_FILES[@]} -eq 0 ]; then
    echo "❌ No markdown files found in $DOCS_DIR"
    exit 1
fi

echo "📚 Found ${#MARKDOWN_FILES[@]} markdown files"
echo "🚀 Starting preview server on port $PORT..."

# Create preview server
cat > /tmp/md-preview-server-$$.py << PYEOF
import http.server
import socketserver
import urllib.parse
import os
import sys
import json
import subprocess
import time
from pathlib import Path
from threading import Thread

PORT = int(sys.argv[1])
DOCS_DIR = Path(sys.argv[2])
USE_GH = sys.argv[3] == "true"
LIVE_RELOAD = sys.argv[4] == "true"
TARGET_FILE = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None

def render_with_gh(markdown_content):
    """Render markdown using GitHub API via gh CLI"""
    try:
        result = subprocess.run(
            ['gh', 'api', 'markdown', '-f', f'text={markdown_content}', '-f', 'mode=gfm'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
    except Exception as e:
        print(f"GitHub API error: {e}")
    return None

def get_file_mtime(filepath):
    """Get file modification time"""
    try:
        return os.path.getmtime(filepath)
    except:
        return 0

# Track file modification times for live reload
file_mtimes = {}

class DocsHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DOCS_DIR), **kwargs)
    
    def log_message(self, format, *args):
        # Suppress default logging
        pass
    
    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        
        # API endpoint for file content
        if parsed_path.path == '/api/file':
            query = urllib.parse.parse_qs(parsed_path.query)
            filename = query.get('name', [None])[0]
            
            if not filename:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'Missing file parameter'}).encode())
                return
            
            file_path = Path(DOCS_DIR) / filename
            if not file_path.exists():
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': 'File not found'}).encode())
                return
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                mtime = get_file_mtime(file_path)
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'content': content,
                    'modified': mtime
                }).encode())
                return
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
                return
        
        # API endpoint for checking file changes
        elif parsed_path.path == '/api/check':
            query = urllib.parse.parse_qs(parsed_path.query)
            filename = query.get('file', [None])[0]
            
            if filename:
                file_path = Path(DOCS_DIR) / filename
                if file_path.exists():
                    mtime = get_file_mtime(file_path)
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'modified': mtime}).encode())
                    return
            
            self.send_response(400)
            self.end_headers()
            return
        
        # Main preview page
        elif parsed_path.path == '/' or parsed_path.path == '/index.html':
            query = urllib.parse.parse_qs(parsed_path.query)
            current_file = query.get('file', [TARGET_FILE])[0] if TARGET_FILE or query.get('file') else None
            
            files_html = '\n'.join([
                f'                        <option value="{f}" {"selected" if f == current_file else ""}>{f}</option>'
                for f in sorted(Path(DOCS_DIR).glob("*.md"))
                if f.name not in ['README_KATEX.md', 'README_PREVIEW.md', 'README_RUSTDOC.md']
            ])
            
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation Preview</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/5.2.0/github-markdown.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #fff;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
        }}
        .header {{
            background: #f6f8fa;
            border: 1px solid #e1e4e8;
            border-radius: 6px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin-bottom: 15px;
            font-size: 24px;
        }}
        .controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .file-selector {{
            flex: 1;
        }}
        .file-selector select {{
            width: 100%;
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #d1d5da;
            border-radius: 4px;
        }}
        .status {{
            padding: 8px 12px;
            background: #28a745;
            color: white;
            border-radius: 4px;
            font-size: 12px;
        }}
        .markdown-body {{
            box-sizing: border-box;
            min-width: 200px;
            max-width: 980px;
            margin: 0 auto;
            padding: 45px;
            font-size: 16px;
            line-height: 1.6;
        }}
        .katex {{ font-size: 1.1em; }}
        .katex-display {{ margin: 1.5em 0; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📚 Documentation Preview</h1>
            <div class="controls">
                <div class="file-selector">
                    <select id="doc-select" onchange="loadDocument(this.value)">
                        <option value="">-- Select a document --</option>
                        {files_html}
                    </select>
                </div>
                <div class="status" id="status">Ready</div>
            </div>
        </div>
        <div class="markdown-body" id="content">
            <p style="text-align: center; color: #6a737d; padding: 40px;">
                Select a document from the dropdown above to preview it.
            </p>
        </div>
    </div>
    
    <script>
        const USE_GH = {str(USE_GH).lower()};
        const LIVE_RELOAD = {str(LIVE_RELOAD).lower()};
        let currentFile = null;
        let reloadInterval = null;
        let lastModified = 0;
        
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
        
        async function loadDocument(filename) {{
            if (!filename) {{
                document.getElementById('content').innerHTML = 
                    '<p style="text-align: center; color: #6a737d; padding: 40px;">Select a document from the dropdown above to preview it.</p>';
                currentFile = null;
                if (reloadInterval) {{
                    clearInterval(reloadInterval);
                    reloadInterval = null;
                }}
                return;
            }}
            
            const status = document.getElementById('status');
            status.textContent = 'Loading...';
            status.style.background = '#ffc107';
            
            try {{
                const response = await fetch(`/api/file?name=${{encodeURIComponent(filename)}}`);
                if (!response.ok) {{
                    throw new Error(`Failed to load: ${{response.statusText}}`);
                }}
                
                const data = await response.json();
                const markdown = data.content;
                lastModified = data.modified;
                
                let html;
                if (USE_GH) {{
                    // Use GitHub API for rendering
                    const ghResponse = await fetch('https://api.github.com/markdown', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{
                            text: markdown,
                            mode: 'gfm'
                        }})
                    }});
                    if (ghResponse.ok) {{
                        html = await ghResponse.text();
                    }} else {{
                        throw new Error('GitHub API failed');
                    }}
                }} else {{
                    // Local rendering with marked.js
                    if (typeof marked === 'undefined') {{
                        // Load marked.js if not available
                        const script = document.createElement('script');
                        script.src = 'https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js';
                        script.onload = () => loadDocument(filename);
                        document.head.appendChild(script);
                        return;
                    }}
                    html = marked.parse(markdown);
                }}
                
                document.getElementById('content').innerHTML = html;
                
                // Render math with KaTeX
                if (typeof renderMathInElement !== 'undefined') {{
                    renderMathInElement(document.body, katexOptions);
                }}
                
                currentFile = filename;
                status.textContent = 'Live';
                status.style.background = '#28a745';
                
                // Start auto-reload if enabled
                if (LIVE_RELOAD && !reloadInterval) {{
                    reloadInterval = setInterval(async () => {{
                        if (currentFile) {{
                            try {{
                                const checkResponse = await fetch(`/api/check?file=${{encodeURIComponent(currentFile)}}`);
                                if (checkResponse.ok) {{
                                    const checkData = await checkResponse.json();
                                    if (checkData.modified > lastModified) {{
                                        lastModified = checkData.modified;
                                        loadDocument(currentFile);
                                    }}
                                }}
                            }} catch (e) {{
                                // Ignore errors
                            }}
                        }}
                    }}, 2000);
                }}
                
            }} catch (error) {{
                document.getElementById('content').innerHTML = 
                    `<div style="background: #ffeef0; border: 1px solid #f97583; border-radius: 6px; padding: 16px; color: #86181d;">
                        <strong>Error:</strong> ${{error.message}}
                    </div>`;
                status.textContent = 'Error';
                status.style.background = '#dc3545';
            }}
        }}
        
        // Load initial file if specified
        {f'loadDocument("{TARGET_FILE}");' if TARGET_FILE else ''}
    </script>
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
</body>
</html>"""
            
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(html.encode())
            return
        
        # Fall back to serving files
        return super().do_GET()

if __name__ == "__main__":
    with socketserver.TCPServer(("", PORT), DocsHandler) as httpd:
        print(f"🚀 Preview server running at http://localhost:{PORT}")
        print(f"📁 Serving from: {DOCS_DIR}")
        if USE_GH:
            print("✅ Using GitHub API for rendering")
        else:
            print("⚠️  Using local rendering with KaTeX")
        if TARGET_FILE:
            print(f"📄 Initial file: {TARGET_FILE}")
        print(f"\n💡 Open http://localhost:{PORT} in your browser")
        if LIVE_RELOAD:
            print(f"   Files auto-reload every 2 seconds")
        print(f"   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
PYEOF

# Start server
python3 /tmp/md-preview-server-$$.py "$PORT" "$DOCS_DIR" "$USE_GH" "$LIVE_RELOAD" "$TARGET_FILE" &
SERVER_PID=$!

# Cleanup on exit
trap "kill $SERVER_PID 2>/dev/null; rm -f /tmp/md-preview-server-$$.py" EXIT

# Wait a moment for server to start
sleep 1

# Auto-open browser
if [ "$AUTO_OPEN" = true ]; then
    URL="http://localhost:$PORT"
    if [ -n "$TARGET_FILE" ]; then
        URL="$URL?file=$(echo "$TARGET_FILE" | sed 's|.*/||')"
    fi
    
    if command -v open >/dev/null 2>&1; then
        open "$URL"  # macOS
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL"  # Linux
    elif command -v start >/dev/null 2>&1; then
        start "$URL"  # Windows
    fi
fi

# Wait for server
wait $SERVER_PID
