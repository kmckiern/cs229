comp_btex() {
    ARGS=$1
    pdflatex ${ARGS}.tex && biber ${ARGS} && pdflatex ${ARGS}.tex && pdflatex ${ARGS}.tex
}

comp_btex milestone
