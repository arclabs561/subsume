.PHONY: docs docs-open docs-watch help

help:
	@echo "Documentation targets:"
	@echo "  make docs        - Generate rustdoc with KaTeX"
	@echo "  make docs-open   - Generate and open in browser"
	@echo "  make docs-watch  - Watch for changes and regenerate (requires cargo-watch)"

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

