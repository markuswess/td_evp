DEPENDS= intro.md _config.yml _toc.yml logo.png installation.md examples.md
.PHONY: clean

_build: $(DEPENDS)
	jupyter-book build --all .
