# app.py
import io
import re
import numpy as np
import streamlit as st
import soundfile as sf
from scipy.io import wavfile

st.set_page_config(page_title="5.1 Mono → 5.1 WAV Builder", layout="centered")

CHANNEL_MAP = {
    "L": "Left",
    "R": "Right",
    "C": "Centre",
    "LFE": "LFE",
    "LS": "Left Surround",
    "RS": "Right Surround",
}

# Standard 5.1 channel order (common WAV order)
OUTPUT_ORDER = ["L", "R", "C", "LFE", "LS", "RS"]


def detect_channel_code(filename: str) -> str | None:
    """
    Detect channel code from the token immediately before the extension.
    Examples:
      Something_MIX.C.wav
      Something_MIX.LFE.aiff
      Something_MIX.Ls.wav
    """
    # Remove extension
    base = re.sub(r"\.(wav|wave|aif|aiff)$", "", filename, flags=re.IGNORECASE)
    if "." not in base:
        return None
    token = base.split(".")[-1].strip().upper()

    # normalize Ls/Rs case-insensitive to LS/RS
    if token == "LS":
        return "LS"
    if token == "RS":
        return "RS"
    if token in ("LFE", "C", "L", "R"):
        return token

    # allow "Ls" / "Rs" that came through as "LS"/"RS" already,
    # plus a couple of mild variants
    if token in ("L_SUR", "LSUR", "L-S"):
        return "LS"
    if token in ("R_SUR", "RSUR", "R-S"):
        return "RS"

    return None


def read_audio_to_int32(uploaded_file) -> tuple[int, np.ndarray]:
    """
    Read WAV/AIFF with soundfile.
    Returns (sample_rate, mono_int32).
    """
    data = uploaded_file.getvalue()

    # always decode to float32, then convert to int32 scale
    with io.BytesIO(data) as bio:
        audio, sr = sf.read(bio, dtype="float32", always_2d=True)

    if audio.shape[1] != 1:
        raise ValueError(f"Expected mono file, got {audio.shape[1]} channels: {uploaded_file.name}")

    mono = audio[:, 0]
    mono = np.clip(mono, -1.0, 1.0)

    # float [-1,1] -> int32
    mono_i32 = (mono.astype(np.float64) * 2147483647.0).round().astype(np.int32)
    return int(sr), mono_i32


def int32_to_target_dtype(arr_int32: np.ndarray, out_fmt: str) -> np.ndarray:
    """
    Convert int32 interleaved array to chosen output dtype for wavfile.write.
    """
    if out_fmt == "int32":
        return arr_int32.astype(np.int32)

    if out_fmt == "int16":
        # downscale int32 -> int16
        x = (arr_int32.astype(np.float64) / 65536.0).round()
        return np.clip(x, -32768, 32767).astype(np.int16)

    if out_fmt == "float32":
        f = np.clip(arr_int32.astype(np.float64) / 2147483647.0, -1.0, 1.0)
        return f.astype(np.float32)

    raise ValueError("Unsupported output format.")


st.title("5.1 Mono → 5.1 WAV Builder")
st.caption("Drop the 6 mono stems (WAV/AIFF) and export a single 6-channel 5.1 WAV.")

with st.expander("Expected naming", expanded=True):
    st.markdown(
        """
Channel is detected from the **token immediately before the extension**:

- `... .L.wav` → Left
- `... .R.wav` → Right
- `... .C.wav` → Centre
- `... .LFE.wav` → LFE
- `... .Ls.wav` / `... .LS.wav` → Left Surround
- `... .Rs.wav` / `... .RS.wav` → Right Surround
"""
    )

uploaded = st.file_uploader(
    "Drag & drop your 6 mono files here (WAV/AIFF).",
    type=["wav", "wave", "aif", "aiff"],
    accept_multiple_files=True,
)

pad_shorter = st.checkbox("If lengths differ, pad shorter files with silence to match the longest", value=True)

out_choice = st.selectbox(
    "Output sample format",
    options=["32-bit PCM (int32)", "16-bit PCM (int16)", "32-bit float (float32)"],
    index=0,
)
out_fmt = "int32" if out_choice.startswith("32-bit PCM") else ("int16" if out_choice.startswith("16-bit") else "float32")

st.divider()

detected: dict[str, any] = {}
errors: list[str] = []

if uploaded:
    for f in uploaded:
        code = detect_channel_code(f.name)
        if not code:
            errors.append(f"Could not detect channel code from: **{f.name}** (needs .L/.R/.C/.LFE/.Ls/.Rs before extension)")
            continue
        if code in detected:
            errors.append(f"Duplicate channel **{code}**: `{f.name}` and `{detected[code].name}`")
            continue
        detected[code] = f

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Detected channels")
        for k in OUTPUT_ORDER:
            if k in detected:
                st.write(f"✅ **{k}** — {CHANNEL_MAP[k]} — `{detected[k].name}`")
            else:
                st.write(f"❌ **{k}** — {CHANNEL_MAP[k]} — (missing)")
    with cols[1]:
        st.subheader("Notes")
        if errors:
            for e in errors:
                st.warning(e)
        else:
            st.success("No naming issues detected.")

    st.divider()

    can_build = all(k in detected for k in OUTPUT_ORDER) and not errors

    if can_build:
        st.info("Output channel order: **L, R, C, LFE, Ls, Rs**")

        if st.button("Create 5.1 WAV", type="primary"):
            try:
                srs = {}
                audio = {}
                lengths = {}

                for code in OUTPUT_ORDER:
                    sr, mono_i32 = read_audio_to_int32(detected[code])
                    srs[code] = sr
                    audio[code] = mono_i32
                    lengths[code] = len(mono_i32)

                # Validate sample rate
                sr_set = set(srs.values())
                if len(sr_set) != 1:
                    raise ValueError(f"Sample rates differ: { {k: srs[k] for k in OUTPUT_ORDER} }")
                sr = sr_set.pop()

                # Match lengths
                max_len = max(lengths.values())
                min_len = min(lengths.values())
                if max_len != min_len:
                    if not pad_shorter:
                        raise ValueError(f"Lengths differ (min={min_len}, max={max_len}). Enable padding or fix source files.")
                    for k in OUTPUT_ORDER:
                        x = audio[k]
                        if len(x) < max_len:
                            audio[k] = np.pad(x, (0, max_len - len(x)), mode="constant")

                # Interleave into (N, 6)
                stacked_i32 = np.stack([audio[k] for k in OUTPUT_ORDER], axis=1)
                out_data = int32_to_target_dtype(stacked_i32, out_fmt)

                # Output filename from L stem
                example_name = detected["L"].name
                no_ext = re.sub(r"\.(wav|wave|aif|aiff)$", "", example_name, flags=re.IGNORECASE)
                base = no_ext.rsplit(".", 1)[0] if "." in no_ext else no_ext
                out_name = f"{base}_5.1.wav"

                out_buf = io.BytesIO()
                wavfile.write(out_buf, sr, out_data)
                out_bytes = out_buf.getvalue()

                st.success(f"Built 5.1 WAV: **{out_name}** ({sr} Hz, {len(out_data)} frames)")
                st.download_button(
                    "Download 5.1 WAV",
                    data=out_bytes,
                    file_name=out_name,
                    mime="audio/wav",
                )

            except Exception as e:
                st.error(f"Failed to build 5.1 WAV: {e}")
    else:
        st.warning("Add all 6 channels with correct naming to enable export.")
else:
    st.write("Upload your files to begin.")
