# Test data

`frame_0.png` and `frame_1.png` are two scans of Hokusai's woodblock print
*The Great Wave off Kanagawa* (c. 1831), standing in for two screenshots in
the frame-difference and pipeline tests. The print is in the public domain.

Source: https://en.wikipedia.org/wiki/File:The_Great_Wave_off_Kanagawa.jpg

| File | Wikimedia Commons file |
|---|---|
| `frame_0.png` | `Tsunami_by_hokusai_19th_century.jpg` (1920 px thumbnail) |
| `frame_1.png` | `The_Great_Wave_off_Kanagawa.jpg` (original, 4335x2990) |

Both were centre-cropped to the aspect ratio 1423:799 and resized to
1423x799 (Lanczos), the size of the screenshots the tests were first written
with. The reference values in `src/difference.rs` were computed from these two
files with the Python backend's `screenshot.difference`.
