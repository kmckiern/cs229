comp_btex() {
    ARGS=$1
    pdflatex ${ARGS}.tex && bibtex ${ARGS} && pdflatex ${ARGS}.tex && pdflatex ${ARGS}.tex
}

comp_btex milestone
