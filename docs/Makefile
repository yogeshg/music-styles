
pandoc_beamer=pandoc -t beamer -V theme:Warsaw # --toc 

gan-presentation.pdf: gan-presentation.md
	${pandoc_beamer} gan-presentation.md -o gan-presentation.pdf

progress.pdf: progress.md
	${pandoc_beamer} progress.md -o progress.pdf

.PHONY: clean
clean:
	rm -rf gan-presentation.pdf

.PHONY: all
all: clean gan-presentation.pdf progress.pdf

.PHONY: progress
progress: progress.pdf
