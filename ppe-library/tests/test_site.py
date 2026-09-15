"""Static-site integrity checks without browser or JavaScript dependencies."""
from html.parser import HTMLParser
from html import unescape
from pathlib import Path
import re
from urllib.parse import urlsplit, unquote
import unittest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


class Page(HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.links = []
        self.ids = []
        self.language = None
        self.feed(path.read_text())

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "html":
            self.language = attrs.get("lang")
        for attr in ("href", "src"):
            if attrs.get(attr):
                self.links.append(attrs[attr])


class SiteIntegrity(unittest.TestCase):
    def test_displayed_citation_matches_download(self):
        citation = (ROOT / 'CITATION.bib').read_text().strip()
        guide = (DOCS / 'guide.html').read_text()
        match = re.search(r'<code id="citation-bib">(.*?)</code>', guide, re.DOTALL)
        self.assertIsNotNone(match)
        self.assertEqual(unescape(match[1]), citation)
        doi = re.search(r'DOI\s*=\s*\{([^}]+)\}', citation)[1]
        for page in ('index.html', 'guide.html'):
            self.assertIn('https://doi.org/' + doi, (DOCS / page).read_text())

    def test_local_links_and_fragments(self):
        for source in DOCS.glob("*.html"):
            page = Page(source)
            self.assertEqual(page.language, "en")
            self.assertEqual(len(page.ids), len(set(page.ids)), f"Duplicate IDs in {source}")
            for link in page.links:
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                target = (source.parent / unquote(parts.path)).resolve() if parts.path else source
                self.assertTrue(target.is_relative_to(DOCS.resolve()), f"Public link escapes site: {link}")
                self.assertTrue(target.is_file(), f"Missing {link} in {source}")
                if parts.fragment and target.suffix == ".html":
                    self.assertIn(parts.fragment, Page(target).ids, f"Missing fragment {link}")

    def test_downloads_match_sources(self):
        self.assertEqual((DOCS / "assets/custom_dataset.py").read_bytes(),
                         (ROOT / "examples/custom_dataset.py").read_bytes())
        self.assertEqual((DOCS / "assets/citation.bib").read_bytes(),
                         (ROOT / "CITATION.bib").read_bytes())
        self.assertEqual((DOCS / "assets/plot_navigation.py").read_bytes(),
                         (ROOT / "examples/plot_navigation.py").read_bytes())

    def test_creator_material_is_outside_public_tree(self):
        creator_files = {"hosting.md", "logic.md", "validation.md", "plotting-operations.md"}
        for path in DOCS.rglob("*"):
            self.assertTrue(path.resolve().is_relative_to(DOCS.resolve()), f"Escaping symlink: {path}")
            self.assertNotIn(path.name, creator_files)
            if path.suffix in (".html", ".md"):
                content = path.read_text()
                for private_reference in (*creator_files, "maintainer/", "http.server", "ssh -N", "ssh -J"):
                    self.assertNotIn(private_reference, content, f"Creator material in {path}")

    def test_markdown_links_stay_within_public_site(self):
        for source in DOCS.rglob("*.md"):
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", source.read_text()):
                parts = urlsplit(link)
                if parts.scheme or parts.netloc:
                    continue
                target = (source.parent / unquote(parts.path)).resolve() if parts.path else source
                self.assertTrue(target.is_relative_to(DOCS.resolve()), f"Public link escapes site: {link}")
                self.assertTrue(target.is_file(), f"Missing {link} in {source}")
                if parts.fragment and target.suffix == ".html":
                    self.assertIn(parts.fragment, Page(target).ids)


if __name__ == "__main__":
    unittest.main()
