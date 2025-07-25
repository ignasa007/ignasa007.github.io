1. Write a `main.tex` document with a `refs.bib` bibliography.

2. Convert the LaTeX document to a Markdown

    ```bash
    pandoc main.tex -M link-citations=true --citeproc --bibliography=refs.bib --csl=custom.csl --mathjax --wrap=preserve -s -o main.md
    ```

    This will look fine, except the references section will look weird, example

    ```markdown
    ::: {#ref-akaho2007kcca .csl-entry}
    Akaho, S. (2007). A kernel method for canonical correlation analysis.
    Retrieved from <https://arxiv.org/abs/cs/0609071>
    :::
    ```

    Moreover, the in-text citations will not be click-able, i.e. they won't redirect to the corresponding reference.

3. Convert the LaTeX file to HTML

    ```bash
    pandoc main.tex -M link-citations=true --citeproc --bibliography=refs.bib --csl=custom.csl --mathjax --wrap=preserve -s -o main.html
    ```

4. Replace the references section in `main.md` with the one in `main.html`. (Add a 'References' section title.)

5. Add permalink and tags.

6. Equation formatting notes:
    - Wrap the in-line equations in `\\(...\\)` instead of `$...$` &ndash; find `\$([^$]+?)\$` and replace with `\\\\($1\\\\)`.
    - Replace `_` with `\_` ONLY in in-line equations, not in math mode or citations &ndash; find `_` and replace with `\\_`.
    - Replace `\{` and `\}` with `\\{` and `\\}`, respectively &ndash; find `\\([{}])` and replace with `\\\\$1`.
    - Replace `\coloneqq` with `&#x2254;` &ndash; ONLY works in in-line equations, not math mode.