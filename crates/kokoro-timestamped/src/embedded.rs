//! Reading data files that are embedded in the crate xz-compressed (with
//! `xz -9e`), to keep the crate and the programs using it small.

/// Decompresses an embedded `.xz` file.
pub fn decompress_xz(compressed: &[u8]) -> Vec<u8> {
    let mut data = Vec::new();
    lzma_rs::xz_decompress(&mut &compressed[..], &mut data).expect("embedded data is valid xz");
    data
}
