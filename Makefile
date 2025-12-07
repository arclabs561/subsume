.PHONY: docs docs-open docs-watch typst-docs typst-preview help

help:
	@echo "Documentation targets:"
	@echo "  make docs        - Generate rustdoc with KaTeX"
	@echo "  make docs-open   - Generate and open in browser"
	@echo "  make docs-watch  - Watch for changes and regenerate (requires cargo-watch)"
	@echo "  make typst-docs  - Build all Typst math documentation (PDFs)"
	@echo "  make typst-preview FILE=<name> - Preview Typst document with auto-reload"

docs:
	@echo "📚 Generating rustdoc with KaTeX support..."
	@RUSTDOCFLAGS="--html-in-header docs/katex-header.html" cargo doc --no-deps
	@echo "✅ Docs generated at: target/doc/subsume_core/index.html"

docs-open: docs
	@echo "🌐 Opening in browser..."
	@if command -v open >/dev/null 2>&1; then \
		open target/doc/subsume_core/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open target/doc/subsume_core/index.html; \
	elif command -v start >/dev/null 2>&1; then \
		start target/doc/subsume_core/index.html; \
	else \
		echo "📖 Open: target/doc/subsume_core/index.html"; \
	fi

docs-watch:
	@echo "👀 Watching for changes (requires: cargo install cargo-watch)..."
	@cargo watch -x "doc --no-deps --html-in-header docs/katex-header.html"

typst-docs:
	@echo "📐 Building Typst math documentation..."
	@./docs/typst/build.sh

typst-preview:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make typst-preview FILE=<filename>"; \
		echo "Example: make typst-preview FILE=gumbel-box-volume"; \
		exit 1; \
	fi
	@./docs/typst/preview.sh $(FILE)

