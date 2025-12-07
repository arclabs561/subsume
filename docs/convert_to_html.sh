#!/bin/bash
# Convert markdown documentation to HTML with KaTeX rendering
# Usage: ./convert_to_html.sh [input.md] [output.html]

set -e

INPUT="${1:-MATHEMATICAL_FOUNDATIONS.md}"
OUTPUT="${2:-${INPUT%.md}.html}"

if [ ! -f "$INPUT" ]; then
    echo "Error: File $INPUT not found"
    exit 1
fi

cat > "$OUTPUT" << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Documentation - KaTeX Rendering</title>
    
    <!-- KaTeX CSS -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css" integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV" crossorigin="anonymous">
    
    <!-- Marked.js for markdown parsing -->
    <script src="https://cdn.jsdelivr.net/npm/marked@11.1.1/marked.min.js"></script>
    
    <!-- KaTeX JavaScript -->
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js" integrity="sha384-XjKyOOlG8jCC0C9WXZB6XH0jZqKk7xK3I2VqQnBk3xSB7f4F0v3f5q3f5f5f5f5f" crossorigin="anonymous"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"></script>
    
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #fff;
            color: #333;
        }
        
        .katex {
            font-size: 1.1em;
        }
        
        .katex-display {
            margin: 1.5em 0;
            overflow-x: auto;
            overflow-y: hidden;
        }
        
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
    <div id="content"></div>
    
    <script>
        const markdown = `
EOF

# Escape backticks and dollar signs for JavaScript
sed 's/`/\\`/g; s/\$/\\$/g' "$INPUT" >> "$OUTPUT"

cat >> "$OUTPUT" << 'EOF'
        `;
        
        // Configure KaTeX options
        const katexOptions = {
            delimiters: [
                {left: "\\[", right: "\\]", display: true},
                {left: "\\(", right: "\\)", display: false},
                {left: "$$", right: "$$", display: true},
                {left: "$", right: "$", display: false}
            ],
            throwOnError: false,
            strict: false,
            trust: true
        };
        
        // Configure Marked options
        marked.setOptions({
            breaks: true,
            gfm: true,
            headerIds: true,
            mangle: false
        });
        
        // Convert markdown to HTML
        const html = marked.parse(markdown);
        document.getElementById('content').innerHTML = html;
        
        // Render math with KaTeX after DOM is ready
        document.addEventListener('DOMContentLoaded', function() {
            renderMathInElement(document.body, katexOptions);
        });
    </script>
</body>
</html>
EOF

echo "✅ Converted $INPUT to $OUTPUT"
echo "   Open $OUTPUT in a browser to view with KaTeX rendering"

