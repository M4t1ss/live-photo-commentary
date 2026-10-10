//! Frame difference measures (the nine measures and the dispatch by name),
//! used by the pipeline to decide whether a new screenshot
//! differs enough from the previous one to describe.
//!
//! `average_hash`, `phash`, `phash_simple`, `dhash`, `dhash_vertical`,
//! `whash` and `colorhash` are ported from ImageHash 4.3.2
//! (BSD-2-Clause, Copyright (c) 2013-2022 Johannes Buchner); `ssim` is ported
//! from scikit-image 0.26's `structural_similarity`
//! (BSD-3-Clause, Copyright (C) 2009-2022 the scikit-image team). See
//! `../NOTICE.md` and `../LICENSES/`. The DCT (`phash`/`phash_simple`) and
//! the Haar wavelet transform (`whash`) are implemented from the maths, not
//! ported from scipy or PyWavelets.

use image::{DynamicImage, GenericImageView, GrayImage, Luma, imageops::FilterType};

/// How different two images are under the measure with this name (0 for
/// identical images).
pub fn difference(a: &DynamicImage, b: &DynamicImage, measure: &str) -> Result<f64, String> {
    match measure {
        "mse" => Ok(mse(a, b)),
        "ssim" => Ok(ssim(a, b)),
        "average_hash" => Ok(hash_diff(a, b, average_hash)),
        "phash" => Ok(hash_diff(a, b, phash)),
        "phash_simple" => Ok(hash_diff(a, b, phash_simple)),
        "dhash" => Ok(hash_diff(a, b, dhash)),
        "dhash_vertical" => Ok(hash_diff(a, b, dhash_vertical)),
        "whash" => Ok(hash_diff(a, b, whash)),
        "colorhash" => Ok(hash_diff(a, b, colorhash)),
        other => Err(format!("Unsupported similarity measure: {other}")),
    }
}

// ── mse ───────────────────────────────────────────────────────────────────

fn mse(a: &DynamicImage, b: &DynamicImage) -> f64 {
    let (a, b) = to_rgb_pair(a, b, FilterType::CatmullRom);
    let n = (a.width() as f64) * (a.height() as f64) * 3.0;
    let sum_sq: f64 = a
        .pixels()
        .zip(b.pixels())
        .map(|(p, q)| {
            p.0.iter()
                .zip(q.0.iter())
                .map(|(&x, &y)| {
                    let d = x as f64 - y as f64;
                    d * d
                })
                .sum::<f64>()
        })
        .sum();
    (sum_sq / n) / (256.0 * 256.0)
}

// ── ssim ──────────────────────────────────────────────────────────────────

/// `(1 - mean structural similarity) / 2`, averaged over the R, G, B
/// channels, each treated as its own 2D image.
///
/// Uses `win_size=7`, a uniform (box) filter, `K1=0.01`, `K2=0.03`,
/// `use_sample_covariance=True`, `data_range=255` (scikit-image's defaults,
/// with the data range given).
///
/// `structural_similarity` filters with `scipy.ndimage.uniform_filter`'s
/// default `mode='reflect'` boundary handling, then crops exactly
/// `(win_size - 1) / 2` pixels off every edge before averaging. Since that
/// crop width equals the filter's own half-width, every window that
/// survives the crop lies entirely inside the real image — it never reaches
/// into the reflected padding. So the boundary mode never affects the
/// result, and box sums via a summed-area table over the valid region are
/// enough; no reflect-padding needs porting.
fn ssim(a: &DynamicImage, b: &DynamicImage) -> f64 {
    let (a, b) = to_rgb_pair(a, b, FilterType::CatmullRom);
    let (w, h) = (a.width() as usize, a.height() as usize);

    let channel = |img: &image::RgbImage, c: usize| -> Vec<f64> {
        img.pixels().map(|p| p[c] as f64).collect()
    };

    let mssim: f64 = (0..3)
        .map(|c| ssim_channel(&channel(&a, c), &channel(&b, c), w, h, 7, 255.0, 0.01, 0.03))
        .sum::<f64>()
        / 3.0;
    (1.0 - mssim) / 2.0
}

fn ssim_channel(im1: &[f64], im2: &[f64], w: usize, h: usize, win_size: usize, data_range: f64, k1: f64, k2: f64) -> f64 {
    let pad = (win_size - 1) / 2;
    let n = (win_size * win_size) as f64;
    let cov_norm = n / (n - 1.0); // use_sample_covariance=True
    let c1 = (k1 * data_range).powi(2);
    let c2 = (k2 * data_range).powi(2);

    let sq1: Vec<f64> = im1.iter().map(|&x| x * x).collect();
    let sq2: Vec<f64> = im2.iter().map(|&x| x * x).collect();
    let cross: Vec<f64> = im1.iter().zip(im2).map(|(&x, &y)| x * y).collect();

    let i1 = integral_image(im1, w, h);
    let i2 = integral_image(im2, w, h);
    let i11 = integral_image(&sq1, w, h);
    let i22 = integral_image(&sq2, w, h);
    let i12 = integral_image(&cross, w, h);

    let mut total = 0.0;
    let mut count = 0usize;
    for y in pad..h - pad {
        for x in pad..w - pad {
            let (x0, y0, x1, y1) = (x - pad, y - pad, x + pad, y + pad);
            let ux = box_sum(&i1, w, x0, y0, x1, y1) / n;
            let uy = box_sum(&i2, w, x0, y0, x1, y1) / n;
            let uxx = box_sum(&i11, w, x0, y0, x1, y1) / n;
            let uyy = box_sum(&i22, w, x0, y0, x1, y1) / n;
            let uxy = box_sum(&i12, w, x0, y0, x1, y1) / n;
            let vx = cov_norm * (uxx - ux * ux);
            let vy = cov_norm * (uyy - uy * uy);
            let vxy = cov_norm * (uxy - ux * uy);
            let a1 = 2.0 * ux * uy + c1;
            let a2 = 2.0 * vxy + c2;
            let b1 = ux * ux + uy * uy + c1;
            let b2 = vx + vy + c2;
            total += (a1 * a2) / (b1 * b2);
            count += 1;
        }
    }
    total / count as f64
}

/// A summed-area table (size `(w+1) * (h+1)`, row-major): `integral[(y+1)*(w+1)+(x+1)]`
/// is the sum of `data[0..=y][0..=x]`.
fn integral_image(data: &[f64], w: usize, h: usize) -> Vec<f64> {
    let stride = w + 1;
    let mut integral = vec![0.0; stride * (h + 1)];
    for y in 0..h {
        for x in 0..w {
            integral[(y + 1) * stride + (x + 1)] =
                data[y * w + x] + integral[y * stride + (x + 1)] + integral[(y + 1) * stride + x] - integral[y * stride + x];
        }
    }
    integral
}

/// Sum of `data[y0..=y1][x0..=x1]`, from an `integral_image(data, w, _)`.
fn box_sum(integral: &[f64], w: usize, x0: usize, y0: usize, x1: usize, y1: usize) -> f64 {
    let stride = w + 1;
    let at = |x: usize, y: usize| integral[y * stride + x];
    at(x1 + 1, y1 + 1) - at(x0, y1 + 1) - at(x1 + 1, y0) + at(x0, y0)
}

// ── imagehash dispatch ──────────────────────────────────────────────────

/// Hamming distance over the hash's total bit count, giving every measure
/// the same `[0, 1]` range (0 = identical, 1 = every bit differs).
///
/// Python's `imagehash_difference` divides by `len(hash1.hash) ** 2`
/// instead. `hash1.hash` is a 2D array, so Python's builtin `len()` on it is
/// its row count (`shape[0]`), not its total element count — for a square
/// hash (`hash_size` x `hash_size`) those are the same number, so this
/// matches Python exactly for every measure except `colorhash`, whose
/// `(14, 3)` array makes Python divide by 14²=196 rather than its 42 total
/// bits, capping its range at 42/196 ≈ 0.21 instead of 1 — not a deliberate
/// design, just what falls out of `len()` on a non-square array. Diverging
/// here (dividing by 42) keeps `difference_threshold` meaning the same thing
/// regardless of which measure is selected.
fn hash_diff(a: &DynamicImage, b: &DynamicImage, hash_fn: impl Fn(&DynamicImage) -> Vec<bool>) -> f64 {
    let ha = hash_fn(a);
    let hb = hash_fn(b);
    let n = ha.len() as f64;
    let distance = ha.iter().zip(&hb).filter(|(x, y)| x != y).count() as f64;
    distance / n
}

/// `imagehash`'s `numpy.median`: average of the two middle elements for an
/// even-length array, the middle element for an odd-length one.
fn median(values: &mut [f64]) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 1 { values[n / 2] } else { (values[n / 2 - 1] + values[n / 2]) / 2.0 }
}

/// PIL's `convert('L')`: ITU-R 601-2 luma, as the exact fixed-point formula
/// Pillow uses (verified against it pixel-for-pixel).
fn pil_grayscale(img: &DynamicImage) -> GrayImage {
    let rgb = img.to_rgb8();
    GrayImage::from_fn(rgb.width(), rgb.height(), |x, y| {
        let p = rgb.get_pixel(x, y);
        let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
        Luma([((r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16) as u8])
    })
}

/// PIL's `convert('L')` then `.resize((width, height), ANTIALIAS)`
/// (`ANTIALIAS` = Lanczos, imagehash's default for every hash function).
/// Returns the resized grayscale pixels, row-major.
fn gray_resized(img: &DynamicImage, width: u32, height: u32) -> Vec<u8> {
    image::imageops::resize(&pil_grayscale(img), width, height, FilterType::Lanczos3).into_raw()
}

fn to_rgb_pair(a: &DynamicImage, b: &DynamicImage, filter: FilterType) -> (image::RgbImage, image::RgbImage) {
    let rgb_b = b.to_rgb8();
    let rgb_a = if a.dimensions() == b.dimensions() { a.to_rgb8() } else { a.resize_exact(rgb_b.width(), rgb_b.height(), filter).to_rgb8() };
    (rgb_a, rgb_b)
}

// ── average_hash ──────────────────────────────────────────────────────────

fn average_hash(img: &DynamicImage) -> Vec<bool> {
    let hash_size = 8u32;
    let pixels = gray_resized(img, hash_size, hash_size);
    let avg = pixels.iter().map(|&v| v as f64).sum::<f64>() / pixels.len() as f64;
    pixels.iter().map(|&v| (v as f64) > avg).collect()
}

// ── phash / phash_simple (DCT) ──────────────────────────────────────────

/// Unnormalized DCT-II (scipy's `fftpack.dct` default, `norm=None`):
/// `y_k = 2 * sum_n x_n * cos(pi * k * (2n + 1) / (2N))`. Any ported measure
/// here only compares values within one DCT output against each other
/// (median- or mean-thresholded), so the exact normalization constant
/// doesn't matter — only the basis shape does.
fn dct2_1d(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    (0..n)
        .map(|k| {
            let sum: f64 = x
                .iter()
                .enumerate()
                .map(|(i, &xi)| xi * (std::f64::consts::PI * k as f64 * (2.0 * i as f64 + 1.0) / (2.0 * n as f64)).cos())
                .sum();
            2.0 * sum
        })
        .collect()
}

/// Transforms each column (`axis=0` in `scipy.fftpack.dct`).
fn dct2_axis0(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let rows = mat.len();
    let cols = mat[0].len();
    let mut out = vec![vec![0.0; cols]; rows];
    for c in 0..cols {
        let col: Vec<f64> = (0..rows).map(|r| mat[r][c]).collect();
        let t = dct2_1d(&col);
        for r in 0..rows {
            out[r][c] = t[r];
        }
    }
    out
}

/// Transforms each row (`axis=1`, the default last-axis in `scipy.fftpack.dct`).
fn dct2_axis1(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    mat.iter().map(|row| dct2_1d(row)).collect()
}

fn gray_matrix(img: &DynamicImage, size: u32) -> Vec<Vec<f64>> {
    let pixels = gray_resized(img, size, size);
    let size = size as usize;
    (0..size).map(|y| (0..size).map(|x| pixels[y * size + x] as f64).collect()).collect()
}

fn phash(img: &DynamicImage) -> Vec<bool> {
    let (hash_size, highfreq_factor) = (8usize, 4usize);
    let img_size = (hash_size * highfreq_factor) as u32;
    let dct = dct2_axis1(&dct2_axis0(&gray_matrix(img, img_size)));
    let mut low = Vec::with_capacity(hash_size * hash_size);
    for row in dct.iter().take(hash_size) {
        low.extend_from_slice(&row[..hash_size]);
    }
    let med = median(&mut low.clone());
    low.iter().map(|&v| v > med).collect()
}

fn phash_simple(img: &DynamicImage) -> Vec<bool> {
    let (hash_size, highfreq_factor) = (8usize, 4usize);
    let img_size = (hash_size * highfreq_factor) as u32;
    let dct = dct2_axis1(&gray_matrix(img, img_size));
    let mut low = Vec::with_capacity(hash_size * hash_size);
    for row in dct.iter().take(hash_size) {
        low.extend_from_slice(&row[1..=hash_size]);
    }
    let avg = low.iter().sum::<f64>() / low.len() as f64;
    low.iter().map(|&v| v > avg).collect()
}

// ── dhash / dhash_vertical ──────────────────────────────────────────────

fn dhash(img: &DynamicImage) -> Vec<bool> {
    let hash_size = 8u32;
    let w = hash_size + 1;
    let pixels = gray_resized(img, w, hash_size);
    let w = w as usize;
    let mut out = Vec::with_capacity(hash_size as usize * hash_size as usize);
    for y in 0..hash_size as usize {
        for x in 1..w {
            out.push(pixels[y * w + x] > pixels[y * w + x - 1]);
        }
    }
    out
}

fn dhash_vertical(img: &DynamicImage) -> Vec<bool> {
    let hash_size = 8u32;
    let h = hash_size + 1;
    let pixels = gray_resized(img, hash_size, h);
    let w = hash_size as usize;
    let mut out = Vec::with_capacity(w * hash_size as usize);
    for y in 1..h as usize {
        for x in 0..w {
            out.push(pixels[y * w + x] > pixels[(y - 1) * w + x]);
        }
    }
    out
}

// ── whash (Haar wavelet) ──────────────────────────────────────────────────

/// One level of a 2D Haar DWT: separable row-then-column 1D transforms.
/// Quadrant naming doesn't need to match PyWavelets' (cH/cV/cD) convention —
/// only that `haar_2d_inverse` undoes it, which it does by construction.
fn haar_2d_forward(mat: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let rows = mat.len();
    let cols = mat[0].len();
    let (half_r, half_c) = (rows / 2, cols / 2);

    let mut l = vec![vec![0.0; half_c]; rows];
    let mut h = vec![vec![0.0; half_c]; rows];
    for (r, row) in mat.iter().enumerate() {
        let (a, d) = haar_1d_forward(row);
        l[r] = a;
        h[r] = d;
    }

    let mut ll = vec![vec![0.0; half_c]; half_r];
    let mut hl = vec![vec![0.0; half_c]; half_r];
    let mut lh = vec![vec![0.0; half_c]; half_r];
    let mut hh = vec![vec![0.0; half_c]; half_r];
    for c in 0..half_c {
        let col_l: Vec<f64> = (0..rows).map(|r| l[r][c]).collect();
        let (a, d) = haar_1d_forward(&col_l);
        for r in 0..half_r {
            ll[r][c] = a[r];
            hl[r][c] = d[r];
        }
        let col_h: Vec<f64> = (0..rows).map(|r| h[r][c]).collect();
        let (a, d) = haar_1d_forward(&col_h);
        for r in 0..half_r {
            lh[r][c] = a[r];
            hh[r][c] = d[r];
        }
    }
    (ll, lh, hl, hh)
}

/// Exact inverse of `haar_2d_forward`.
fn haar_2d_inverse(ll: &[Vec<f64>], lh: &[Vec<f64>], hl: &[Vec<f64>], hh: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let half_r = ll.len();
    let half_c = ll[0].len();
    let rows = half_r * 2;

    let mut l = vec![vec![0.0; half_c]; rows];
    let mut h = vec![vec![0.0; half_c]; rows];
    for c in 0..half_c {
        let col_ll: Vec<f64> = (0..half_r).map(|r| ll[r][c]).collect();
        let col_hl: Vec<f64> = (0..half_r).map(|r| hl[r][c]).collect();
        let col_l = haar_1d_inverse(&col_ll, &col_hl);
        for (r, &v) in col_l.iter().enumerate() {
            l[r][c] = v;
        }

        let col_lh: Vec<f64> = (0..half_r).map(|r| lh[r][c]).collect();
        let col_hh: Vec<f64> = (0..half_r).map(|r| hh[r][c]).collect();
        let col_h = haar_1d_inverse(&col_lh, &col_hh);
        for (r, &v) in col_h.iter().enumerate() {
            h[r][c] = v;
        }
    }

    (0..rows).map(|r| haar_1d_inverse(&l[r], &h[r])).collect()
}

/// Orthonormal single-level 1D Haar transform (`1/sqrt(2)`-scaled), exactly
/// invertible by `haar_1d_inverse`.
fn haar_1d_forward(x: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let s = std::f64::consts::SQRT_2;
    let n = x.len() / 2;
    let mut a = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for i in 0..n {
        a.push((x[2 * i] + x[2 * i + 1]) / s);
        d.push((x[2 * i] - x[2 * i + 1]) / s);
    }
    (a, d)
}

fn haar_1d_inverse(a: &[f64], d: &[f64]) -> Vec<f64> {
    let s = std::f64::consts::SQRT_2;
    let mut x = vec![0.0; a.len() * 2];
    for i in 0..a.len() {
        x[2 * i] = (a[i] + d[i]) / s;
        x[2 * i + 1] = (a[i] - d[i]) / s;
    }
    x
}

/// A multi-level 2D Haar decomposition: the final (coarsest) approximation,
/// plus each level's detail subbands, in the order they were produced
/// (finest level first — `waverec2_haar` undoes them in reverse).
struct HaarDecomposition {
    approx: Vec<Vec<f64>>,
    details: Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)>,
}

fn wavedec2_haar(image: &[Vec<f64>], level: usize) -> HaarDecomposition {
    let mut current = image.to_vec();
    let mut details = Vec::with_capacity(level);
    for _ in 0..level {
        let (ll, lh, hl, hh) = haar_2d_forward(&current);
        details.push((lh, hl, hh));
        current = ll;
    }
    HaarDecomposition { approx: current, details }
}

fn waverec2_haar(decomp: &HaarDecomposition) -> Vec<Vec<f64>> {
    let mut current = decomp.approx.clone();
    for (lh, hl, hh) in decomp.details.iter().rev() {
        current = haar_2d_inverse(&current, lh, hl, hh);
    }
    current
}

/// `whash(image, hash_size=8)`: Haar wavelets, `image_scale` from the image
/// size, `remove_max_haar_ll=True` (imagehash's defaults; the app never
/// overrides them).
fn whash(img: &DynamicImage) -> Vec<bool> {
    let hash_size = 8usize;
    let (orig_w, orig_h) = img.dimensions();
    let min_dim = orig_w.min(orig_h) as f64;
    let image_natural_scale = 2f64.powi(min_dim.log2().floor() as i32) as usize;
    let image_scale = image_natural_scale.max(hash_size);

    let ll_max_level = (image_scale as f64).log2().round() as usize;
    let level = (hash_size as f64).log2().round() as usize;
    let dwt_level = ll_max_level - level;

    let pixels = gray_resized(img, image_scale as u32, image_scale as u32);
    let mut matrix: Vec<Vec<f64>> =
        (0..image_scale).map(|y| (0..image_scale).map(|x| pixels[y * image_scale + x] as f64 / 255.0).collect()).collect();

    // Remove the coarsest LL frequency and reconstruct, using Haar
    // regardless of `mode` (imagehash always does this step with Haar).
    let mut decomp = wavedec2_haar(&matrix, ll_max_level);
    for row in decomp.approx.iter_mut() {
        row.iter_mut().for_each(|v| *v = 0.0);
    }
    matrix = waverec2_haar(&decomp);

    // Decompose again to the target level (`mode='haar'`, the app's only use).
    let decomp = wavedec2_haar(&matrix, dwt_level);
    let low: Vec<f64> = decomp.approx.iter().flatten().copied().collect();
    let med = median(&mut low.clone()); // `median` sorts in place; `low`'s spatial order must survive it
    low.iter().map(|&v| v > med).collect()
}

// ── colorhash ─────────────────────────────────────────────────────────────

/// PIL's `convert('HSV')`: verified pixel-for-pixel against Pillow for
/// representative colors. `s` and `h` use truncating integer division, as
/// Pillow's C implementation does; small ±1 rounding differences are
/// possible but don't matter here (see `colorhash`'s module doc).
fn rgb_to_hsv_pillow(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let (rf, gf, bf) = (r as i32, g as i32, b as i32);
    let max = rf.max(gf).max(bf);
    let min = rf.min(gf).min(bf);
    let delta = max - min;
    let v = max as u8;
    if delta == 0 {
        return (0, 0, v);
    }
    let s = ((delta * 255) / max) as u8;
    let h_sixth = if max == rf {
        (gf - bf) as f64 / delta as f64
    } else if max == gf {
        2.0 + (bf - rf) as f64 / delta as f64
    } else {
        4.0 + (rf - gf) as f64 / delta as f64
    };
    let mut h_deg = h_sixth * 60.0;
    if h_deg < 0.0 {
        h_deg += 360.0;
    }
    let h = ((h_deg * 255.0 / 360.0).floor() as i64).clamp(0, 255) as u8;
    (h, s, v)
}

/// `(v // 2**(binbits-i-1)) % 2**(binbits-i) > 0` for `i in 0..binbits`, as
/// `colorhash` computes it in Python — not a standard binary decomposition
/// (ported as-is, bug-for-bug, since the difference is the Hamming distance
/// between two hashes built the same way).
fn colorhash_bits(v: i64, binbits: u32) -> Vec<bool> {
    (0..binbits)
        .map(|i| {
            let divisor = 1i64 << (binbits - i - 1);
            let modulus = 1i64 << (binbits - i);
            ((v / divisor) % modulus) > 0
        })
        .collect()
}

fn colorhash(img: &DynamicImage) -> Vec<bool> {
    let binbits: u32 = 3;
    let maxvalue = 1i64 << binbits;
    let bin_edges = [0.0f64, 42.5, 85.0, 127.5, 170.0, 212.5, 255.0];
    let hue_bin = |h: u8| -> usize {
        let h = h as f64;
        (0..6).find(|&i| if i == 5 { h >= bin_edges[i] && h <= bin_edges[i + 1] } else { h >= bin_edges[i] && h < bin_edges[i + 1] }).unwrap_or(5)
    };

    let rgb = img.to_rgb8();
    let total = (rgb.width() as u64 * rgb.height() as u64) as f64;

    let (mut black, mut gray, mut colors) = (0u64, 0u64, 0u64);
    let mut faint_counts = [0u64; 6];
    let mut bright_counts = [0u64; 6];

    for p in rgb.pixels() {
        let (r, g, b) = (p[0] as u32, p[1] as u32, p[2] as u32);
        let l = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
        if l < 256 / 8 {
            black += 1;
            continue;
        }
        let (h, s, _v) = rgb_to_hsv_pillow(p[0], p[1], p[2]);
        if (s as u32) < 256 / 3 {
            gray += 1;
            continue;
        }
        colors += 1;
        let bin = hue_bin(h);
        if (s as u32) < 256 * 2 / 3 {
            faint_counts[bin] += 1;
        } else if (s as u32) > 256 * 2 / 3 {
            bright_counts[bin] += 1;
        }
    }

    let frac_black = black as f64 / total;
    let frac_gray = gray as f64 / total;
    let c = (colors.max(1)) as f64;

    let mut values: Vec<i64> =
        vec![((frac_black * maxvalue as f64) as i64).min(maxvalue - 1), ((frac_gray * maxvalue as f64) as i64).min(maxvalue - 1)];
    for &count in faint_counts.iter().chain(bright_counts.iter()) {
        values.push(((count as f64 * maxvalue as f64 / c) as i64).min(maxvalue - 1));
    }

    values.into_iter().flat_map(|v| colorhash_bits(v, binbits)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_images() -> (DynamicImage, DynamicImage) {
        let a = image::RgbImage::from_pixel(64, 64, image::Rgb([100, 120, 140]));
        let mut b = a.clone();
        for y in 0..8 {
            for x in 0..8 {
                b.put_pixel(x, y, image::Rgb([200, 50, 10]));
            }
        }
        (DynamicImage::ImageRgb8(a), DynamicImage::ImageRgb8(b))
    }

    /// `tests/data/frame_{0,1}.png` are two scans of Hokusai's "The Great Wave
    /// off Kanagawa" (public domain), see `tests/data/README.md`. The tests
    /// skip when either file is missing. The Python reference values below
    /// were generated from this pair with the real `screenshot.difference`.
    fn real_frames() -> Option<(DynamicImage, DynamicImage)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let (p0, p1) = (dir.join("frame_0.png"), dir.join("frame_1.png"));
        if !p0.exists() || !p1.exists() {
            eprintln!("skipping: {} not present on this machine", dir.display());
            return None;
        }
        Some((image::open(p0).unwrap(), image::open(p1).unwrap()))
    }

    #[test]
    fn unknown_measure_errors() {
        let (a, b) = synth_images();
        assert!(difference(&a, &b, "nope").is_err());
    }

    #[test]
    fn identical_images_are_zero_for_every_measure() {
        let (a, _) = synth_images();
        for measure in ["mse", "ssim", "average_hash", "phash", "phash_simple", "dhash", "dhash_vertical", "whash", "colorhash"] {
            assert_eq!(difference(&a, &a, measure).unwrap(), 0.0, "{measure} of an image with itself should be 0");
        }
    }

    #[test]
    fn mse_matches_python_exactly() {
        // Reference values from the Python backend's `screenshot.difference`
        // (removed in 0.2.0; it lives on in the `tauri` branch).
        let (a, b) = synth_images();
        let got = difference(&a, &b, "mse").unwrap();
        assert!((got - 0.0025272369384765625).abs() < 1e-9, "got {got}");

        if let Some((f0, f1)) = real_frames() {
            let got = difference(&f0, &f1, "mse").unwrap();
            assert!((got - 0.0785212367773056).abs() < 1e-6, "got {got}");
        }
    }

    #[test]
    fn ssim_is_close_to_python() {
        let (a, b) = synth_images();
        let got = difference(&a, &b, "ssim").unwrap();
        assert!((got - 0.008756290354531515).abs() < 1e-3, "got {got}");

        if let Some((f0, f1)) = real_frames() {
            let got = difference(&f0, &f1, "ssim").unwrap();
            assert!((got - 0.3787264328473226).abs() < 1e-3, "got {got}");
        }
    }

    /// Hash-based measures resize with `image`'s Lanczos3, not PIL's exact
    /// Lanczos, so a handful of bits can land on the other side of a
    /// median/mean threshold. The thresholds are coarse, so that is accepted;
    /// these tests check we're close, not exact.
    fn assert_close_to_python(measure: &str, a: &DynamicImage, b: &DynamicImage, expected: f64, tolerance: f64) {
        let got = difference(a, b, measure).unwrap();
        assert!((got - expected).abs() < tolerance, "{measure}: got {got}, expected {expected} (+/- {tolerance})");
    }

    #[test]
    fn hash_measures_are_close_to_python_on_synthetic_images() {
        let (a, b) = synth_images();
        for (measure, expected, tolerance) in [
            ("average_hash", 0.953125, 0.1),
            ("phash", 0.484375, 0.1),
            // A sharp 64x64 corner edge stresses the Lanczos resize
            // difference harder than a real screenshot does (see
            // `hash_measures_are_close_to_python_on_real_frames`, which
            // passes comfortably at 0.1); measured gap here is 0.125.
            ("phash_simple", 0.625, 0.15),
            ("dhash", 0.046875, 0.1),
            ("dhash_vertical", 0.046875, 0.1),
            // Same sharp-edge sensitivity as phash_simple above; measured
            // gap here is also 0.125.
            ("whash", 0.234375, 0.15),
        ] {
            assert_close_to_python(measure, &a, &b, expected, tolerance);
        }
    }

    #[test]
    fn hash_measures_are_close_to_python_on_real_frames() {
        let Some((f0, f1)) = real_frames() else { return };
        for (measure, expected) in [
            ("average_hash", 0.125),
            ("phash", 0.21875),
            ("phash_simple", 0.078125),
            ("dhash", 0.171875),
            ("dhash_vertical", 0.21875),
            ("whash", 0.125),
        ] {
            assert_close_to_python(measure, &f0, &f1, expected, 0.1);
        }
    }

    /// `colorhash` deliberately diverges from Python here (see `hash_diff`'s
    /// doc comment): dividing by its 42 total bits instead of Python's
    /// `14**2 = 196` rescales the same Hamming distance, so these expected
    /// values are Python's own (`3/196`, `1/196`) recomputed over 42 instead
    /// of 196. `colorhash` doesn't resize the image, so unlike the other
    /// hash measures this should match closely, not just approximately.
    #[test]
    fn colorhash_divides_by_its_total_bits_not_rows_squared() {
        let (a, b) = synth_images();
        assert_close_to_python("colorhash", &a, &b, 3.0 / 42.0, 1e-9);

        if let Some((f0, f1)) = real_frames() {
            assert_close_to_python("colorhash", &f0, &f1, 1.0 / 42.0, 1e-9);
        }
    }

    #[test]
    fn rgb_to_hsv_matches_pillow_on_representative_colors() {
        // (r, g, b) -> (h, s, v), from `im.convert('HSV').getpixel(...)` in Pillow.
        let cases = [
            ((255, 0, 0), (0, 255, 255)),
            ((0, 255, 0), (85, 255, 255)),
            ((0, 0, 255), (170, 255, 255)),
            ((10, 20, 30), (148, 170, 30)),
            ((123, 45, 200), (191, 197, 200)),
            ((255, 255, 255), (0, 0, 255)),
            ((0, 0, 0), (0, 0, 0)),
            ((200, 100, 50), (14, 191, 200)),
            ((50, 200, 100), (99, 191, 200)),
            ((100, 50, 200), (184, 191, 200)),
        ];
        for ((r, g, b), expected) in cases {
            assert_eq!(rgb_to_hsv_pillow(r, g, b), expected, "rgb({r},{g},{b})");
        }
    }

    #[test]
    fn grayscale_matches_pillow_on_representative_colors() {
        let rgb = image::RgbImage::from_fn(1, 1, |_, _| image::Rgb([123, 45, 200]));
        let gray = pil_grayscale(&DynamicImage::ImageRgb8(rgb));
        assert_eq!(gray.get_pixel(0, 0).0[0], 86); // verified against PIL
    }

    #[test]
    fn haar_round_trip_reconstructs_the_original() {
        let matrix: Vec<Vec<f64>> = (0..8).map(|y| (0..8).map(|x| (y * 8 + x) as f64).collect()).collect();
        let decomp = wavedec2_haar(&matrix, 3);
        let reconstructed = waverec2_haar(&decomp);
        for y in 0..8 {
            for x in 0..8 {
                assert!((reconstructed[y][x] - matrix[y][x]).abs() < 1e-9, "({x},{y})");
            }
        }
    }

    #[test]
    fn dct_median_threshold_is_scale_invariant() {
        // phash/phash_simple only compare DCT output against its own
        // median/mean, so any positive uniform scale factor must give the
        // same hash: a sanity check that our unnormalized DCT is safe to use
        // without matching scipy's exact normalization constant.
        let x: Vec<f64> = (0..32).map(|i| (i as f64 * 0.37).sin()).collect();
        let scaled: Vec<f64> = x.iter().map(|&v| v * 3.0).collect();
        let d1 = dct2_1d(&x);
        let d2 = dct2_1d(&scaled);
        for (a, b) in d1.iter().zip(&d2) {
            assert!((a * 3.0 - b).abs() < 1e-9);
        }
    }
}
