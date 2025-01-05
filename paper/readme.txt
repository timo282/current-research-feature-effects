You find five files in this zip file:

- readme.txt: the file you're reading right now; available purely FYI

- paper.tex: a LaTeX template for your paper; just fill out the blanks!

- llncs.cls: a LaTeX class that is required for compiling your paper;

- splncs04.bst: a BibTeX style that is required for compiling your paper if you use BibTeX

In order to work on and compile your LaTeX paper, make sure that the last three files are all in the same directory, and run 
pdflatex paper.tex
from that directory. If you use figure formats that are incompatible with pdflatex, please use the appropriate command chain instead. Alternatively, you can upload these three files to an empty Overleaf repository, and tell Overleaf that paper.tex is your main file.

When preparing your paper submission, please note that the Springer style files prescribe certain markup choices; authors may not deviate from such choices. Please do not change fonts, font sizes*, margins, spacing around headers, and so on, and so forth. The following publication contains an overview of things that are _not_ allowed:
https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.1361
Since this paper was published, LaTeX development has not stopped completely, so this paper is by necessity not exhaustive. If you intend to use LaTeX packages and/or commands that are not explicitly listed in this paper, but have similar effects to ones that are listed, please think again.

If you find yourself with a paper that spills over the maximum number of pages, we understand that you do want to reduce it back to within the page limit. In such cases, rather than changing fonts or margins, we recommend reducing sizes of figures, and reformulating your sentences/paragraphs. For instance, the opening line of this very paragraph could instead have started with "If your paper surpasses the maximum number of pages", and that would have saved half a line in a compiled PDF.

If you have any questions regarding the formatting requirements, please contact the ECML PKDD 2024 Proceedings Chairs.

*notice that resizing or scaling an entire table also changes the font size. Consider transposing your oversized table, or splitting it into multiple tables.