FILE = filename

all:
	latexmk -pdf -enable-write18 -pdflatex="pdflatex \input{%S}'" ${FILE}