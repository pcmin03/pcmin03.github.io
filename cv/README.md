# CV

LaTeX source of `/MY_CV_final.pdf`, moved here from Overleaf. Jekyll skips this folder (`_config.yml` → `exclude`).

Build the PDF and replace the published copy:

```bash
tectonic -X compile cv/main.tex --outdir cv
mv cv/main.pdf MY_CV_final.pdf
```

`main.tex` still compiles with pdfLaTeX on Overleaf; the `\ifXeTeX` lines only set fonts for local XeTeX (tectonic) builds.
