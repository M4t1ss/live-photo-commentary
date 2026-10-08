# Notices

The app (this directory) is licensed under the MIT licence (`../LICENSE`).

Parts of it are ported from the projects below. Their licence texts are in
`LICENSES/`. Each ported module says at the top what it was ported from and
how it differs.

## Ported code

| Rust | Ported from | Licence |
|---|---|---|
| `src/difference.rs`: `average_hash`, `phash`, `phash_simple`, `dhash`, `dhash_vertical`, `whash`, `colorhash` | `imagehash/__init__.py` from [ImageHash](https://github.com/JohannesBuchner/imagehash) 4.3.2, Copyright (c) 2013-2022 Johannes Buchner | BSD-2-Clause (`LICENSES/imagehash-BSD-2-Clause.txt`) |
| `src/difference.rs`: `ssim` | `skimage/metrics/_structural_similarity.py` from [scikit-image](https://github.com/scikit-image/scikit-image) 0.26, Copyright (C) 2009-2022 the scikit-image team | BSD-3-Clause (`LICENSES/scikit-image-BSD-3-Clause.txt`) |

The DCT (for `phash`/`phash_simple`) and the Haar wavelet transform (for
`whash`) are implemented from the maths, not ported from scipy or PyWavelets.
`phash` is ported only from ImageHash, never from the original pHash library
(phash.org), which is GPL-3.0.
