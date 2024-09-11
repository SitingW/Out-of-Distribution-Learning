FILE = filename
all:
	latexmk -pdf -enable-write18 -pdflatex="pdflatex -shell-escap %O '\PassOptionsToPackage{disable}{todonotes}\input{%S}'" ${FILE}
final:
	latexmk -pdf -enable-write18 -pdflatex="pdflatex -shell-escap %O '\PassOptionsToPackage{disable}{todonotes}\input{%S}'" ${FILE}
#	pdflatex '\PassOptionsToPackage{disable}{todonotes}\input{paper}�
# 	[ -d ~/Dropbox/ ] && cp paper.pdf ~/Dropbox/paper.pdf
clean:
	latexmk -C
	rm tikz/*