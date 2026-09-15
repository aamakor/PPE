"""Refresh download assets without a frontend build tool."""
from pathlib import Path
from html import escape
import re
import shutil

root = Path(__file__).resolve().parents[1]
for source, name in (("examples/custom_dataset.py", "custom_dataset.py"),
                     ("examples/plot_navigation.py", "plot_navigation.py"),
                     ("CITATION.bib", "citation.bib")):
    shutil.copyfile(root / source, root / "docs/assets" / name)

# Both displayed references and the download come from the same BibTeX source.
citation = (root / "CITATION.bib").read_text().strip()
fields = dict(re.findall(r'^\s*(\w+)\s*=\s*\{([^\n]*)\}', citation, re.MULTILINE))
authors = []
for author in fields['author'].split(' and '):
    family, given = author.split(',', 1)
    authors.append(f'{given.strip()} {family.strip()}')
reference = (
    f'<h3>{escape(fields["title"])}</h3>'
    f'<p>{escape(" · ".join(authors))}</p>'
    f'<p>{escape(fields["booktitle"])}. {escape(fields["publisher"])}, '
    f'{escape(fields["year"])}, pp. {escape(fields["pages"])}.</p>'
    f'<p><a href="https://doi.org/{escape(fields["DOI"], quote=True)}">'
    f'DOI: {escape(fields["DOI"])}</a></p>'
)
index = root / 'docs/index.html'
content, count = re.subn(
    r'(<section class="wrap citation-block">).*?(</section>)',
    lambda m: m[1] + '<p class="eyebrow">CITE THE METHOD</p>' + reference
    + '<a href="assets/citation.bib" download>Download BibTeX ↓</a>' + m[2],
    index.read_text(), flags=re.DOTALL,
)
if count != 1:
    raise ValueError('Expected one homepage citation section.')
index.write_text(content)

guide = root / 'docs/guide.html'
content, count = re.subn(
    r'(<h2 id="citation">Citation</h2>).*?(</pre>)',
    lambda m: m[1] + reference + '<pre><code id="citation-bib">'
    + escape(citation) + '</code>' + m[2],
    guide.read_text(), flags=re.DOTALL,
)
if count != 1:
    raise ValueError('Expected one user-guide citation section.')
guide.write_text(content)
