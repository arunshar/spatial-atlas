# Spatial Atlas Poster Print Preflight

Preflight date: 2026-07-30

Print authority: `poster/spatial_atlas_poster.pdf`

## Result

**PASS with one non-blocking raster caution.**

No clipping, overlap, missing-font, malformed-PDF, broken-QR, or broken-link
defect was found. The PDF is suitable for a 48 by 36 inch landscape poster
order. The exact Berkeley RDI wordmark is a 93 ppi raster image at its current
placement. It should be acceptable at normal poster viewing distance, but it
can look mildly soft under close inspection.

## Measured PDF properties

| Check | Result |
| --- | --- |
| Pages | 1 |
| Orientation | Landscape |
| MediaBox | 3456 by 2592 pt |
| Physical size | 48 by 36 in |
| PDF version | 1.7 |
| Encryption | None |
| JavaScript | None |
| Ghostscript parse | Pass |
| Overfull boxes | 0 |
| Underfull boxes | 0 |

The final source was rebuilt with XeLaTeX after the footer margin correction.

## Fonts

- 11 font resources are embedded.
- All 11 are subsetted.
- No Type 3 fonts are present.
- The build log contains only the known EB Garamond Initials missing-space
  warning and an `inputenc` notice from XeLaTeX.
- No visible text loss was found in the final rendered preview.

## Images

| Asset | Form in PDF | Effective resolution |
| --- | --- | --- |
| University of Minnesota seal | Vector PDF | Resolution independent |
| NVIDIA logo | Vector PDF | Resolution independent |
| Berkeley RDI wordmark | 720 by 148 pixel RGB image | 93 by 93 ppi |
| Figures, equations, tables, and QR codes | Vector LaTeX and TikZ content | Resolution independent |

The Berkeley wordmark is byte-identical to Berkeley RDI's official website
asset. No exact higher-resolution or vector version of this newer wordmark was
found in the official repository, its history, or its press kit. A larger
official press-kit logo exists, but it uses a different blue-and-gold design
and was not substituted.

## Content margins

The poster was rendered at 72 dpi for a pixel-level edge scan.

| Margin | Measured clear content margin |
| --- | --- |
| Left | 0.389 in |
| Right | 0.389 in |
| Top | 0.264 in |
| Bottom | 0.264 in |

The scan excludes two intentional full-width decorative rules at the header
and footer. Those rules reach the trim edge by design. A printer may trim a
small part of those rules without affecting text, logos, tables, equations,
QR codes, or contact information.

## QR codes

All four QR codes were detected and decoded from the complete poster rendered
at 90 dpi:

1. <https://github.com/arunshar/spatial-atlas>
2. <https://huggingface.co/spaces/Arun0808/spatial-atlas>
3. <https://arxiv.org/abs/2604.12102v2>
4. <https://raw.githubusercontent.com/arunshar/spatial-atlas/main/poster/spatial_atlas_poster.pdf>

The codes retain white quiet zones and adequate separation in the two-by-two
matrix.

## Clickable PDF contacts

Seven invisible-border PDF annotations were verified:

1. Phone
2. Email
3. LinkedIn
4. Personal website
5. GitHub
6. Hugging Face
7. X

The visible text remains black.

## Visual review

The 90 dpi full-page render was inspected for:

- header-logo cropping
- title and subtitle collisions
- column-boundary overlap
- figure-arrow visibility
- table alignment
- equation clipping
- QR-code interference
- footer wrapping
- edge clipping

No blocking visual defect was found. The footer remains on one line.

## Reproduction checks

Run from the repository root:

```bash
cd poster
xelatex -interaction=nonstopmode -halt-on-error -file-line-error spatial_atlas_poster.tex
xelatex -interaction=nonstopmode -halt-on-error -file-line-error spatial_atlas_poster.tex
pdfinfo -box spatial_atlas_poster.pdf
pdffonts spatial_atlas_poster.pdf
pdfimages -list spatial_atlas_poster.pdf
pdfinfo -url spatial_atlas_poster.pdf
gs -q -dNOPAUSE -dBATCH -sDEVICE=nullpage spatial_atlas_poster.pdf
```

`qpdf` was not installed in the preflight environment. Ghostscript parsed the
complete PDF successfully.

## Print instruction

Submit `poster/spatial_atlas_poster.pdf` at its native 48 by 36 inch landscape
size. Do not use the PNG preview as the print source.
