
pandoc_beamer=pandoc -t beamer -V theme:Warsaw # --toc 

gan-presentation.pdf: gan-presentation.md
	${pandoc_beamer} gan-presentation.md -o gan-presentation.pdf

clean:
	rm -rf gan-presentation.pdf

