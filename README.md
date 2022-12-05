# ITOHI

- itohi.com static web page.
- API backend

Dockerized and deployable into DockerHub as well as Github Pages.

## Web page

Web page is a blog created using Hugo with Clarity theme.
I am using Hugo modules.
Simetimes the site fails to render locally when running `hugo serve` with `found no layout file for "JSON" for kind "home": You should create a template file which matches Hugo Layouts Lookup Rules for this combination.` error messages all over the place.
In that case I need to simply run `hugo mod clean`, and then `hugo serve` again.

### Hugo GoAT diagrams

Native support from Hugo - use `goat` code block.

### Mermaid diagrams support

Just use `mermaid` code block.

### WaveDrom waves/circuits support

Just use `wave` code block. Tutorials are [here](https://wavedrom.com/tutorial.html).

### GraphViz DOT support

Just use `viz-dot` code block.
.DOT graph diagrams are included using a deprecated and abandoned [GraphViz](https://github.com/mdaines/viz.js) js library.
.DOT language [documentation](https://graphviz.org/doc/info/lang.html).

### Charts support

Just use `chart` code block.
This uses Chart.js library.

## API Backend

API Backend contains GraphQL and GRPC servers as well as proxies for certain sub-projects.