# app.py
import io
import re
import wave
import aifc
import numpy as np
import streamlit as st
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

# Standard 5.1 order for a multichannel WAV (SMPTE / common DAW order)
OUTPUT_ORDER = ["L", "R", "C", "LFE", "LS", "RS"]


def detect_channel_code(filename: str) -> str | None:
    """
    Detect channel code from the token immediately before the file extension.
    Supports filenames like:
      Something_MIX.C.wav
      Something_MIX.LFE.aiff
    """
    # Remove extension
    base = re.sub(r"\.(wav|wave|aif|aiff)$", "", filename, flags=re.IGNORECASE)

    # Take last token after a dot
    if "." not in base:
        return None

    token = base.split(".")[-1].strip()

    token_u = token.upper()
    # Normalize common forms
    if token_u == "LS":
        return "LS"
    if token_u == "RS":
        return "RS"
    if token_u in ("LFE", "C", "L", "R"):
        return token_u
    if token_u == "LSUR" or token_u == "L_SUR" or token_u == "L-S":
        return "LS"
    if token_u == "RSUR" or token_u == "R_SUR" or token_u == "R-S":
        return "RS"

    return None


def _decode_pcm_bytes(raw: bytes, sampwidth: int) -> np.ndarray:
    """
    Decode PCM frames into int32 numpy array (mono or interleaved).
    Supports 8/16/24/32-bit PCM.
    """
    if sampwidth == 1:
        # 8-bit PCM is usually unsigned
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        return arr.astype(np.int32)
    if sampwidth == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.int32)
    if sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8)
        b = b.reshape(-1, 3)
        # little-endian 24-bit to signed int32
        out = (b[:, 0].astype(np.int32) |
               (b[:, 1].astype(np.int32) << 8) |
               (b[:, 2].astype(np.int32) << 16))
        sign = out & 0x800000
        out = out - (sign << 1)
        return out.astype(np.int32)
    if sampwidth == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.int32)

    raise ValueError(f"Unsupported sample width: {sampwidth} bytes")


def read_wav_to_int32(file_bytes: bytes) -> tuple[int, np.ndarray]:
    """
    Read WAV using SciPy (handles many PCM depths). Returns (sr, mono_int32).
    """
    with io.BytesIO(file_bytes) as bio:
        sr, data = wavfile.read(bio)

    # SciPy can return (N,) or (N, ch). Convert to mono int32.
    if data.ndim == 2:
        if data.shape[1] != 1:
            raise ValueError(f"Expected mono WAV, got {data.shape[1]} channels.")
        data = data[:, 0]

    if data.dtype.kind == "f":
        # float - convert to int32 range
        data = np.clip(data, -1.0, 1.0)
        data = (data * 2147483647.0).astype(np.int32)
    else:
        # int16/int32/uint8 etc → int32
        if data.dtype == np.uint8:
            data = data.astype(np.int16) - 128
        data = data.astype(np.int32)

    return int(sr), data


def read_aiff_to_int32(file_bytes: bytes) -> tuple[int, np.ndarray]:
    """
    Read AIFF/AIFFC using stdlib aifc. Returns (sr, mono_int32).
    Note: AIFF stores big-endian PCM; aifc gives raw frames.
    We'll decode big-endian for 16/24/32; 8-bit is unsigned-ish.
    """
    with aifc.open(io.BytesIO(file_bytes), "rb") as af:
        sr = af.getframerate()
        nch = af.getnchannels()
        sampwidth = af.getsampwidth()
        nframes = af.getnframes()
        raw = af.readframes(nframes)

    if nch != 1:
        raise ValueError(f"Expected mono AIFF, got {nch} channels.")

    # Decode big-endian
    if sampwidth == 1:
        arr = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
        return int(sr), arr.astype(np.int32)
    if sampwidth == 2:
        return int(sr), np.frombuffer(raw, dtype=">i2").astype(np.int32)
    if sampwidth == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        out = (b[:, 0].astype(np.int32) << 16) | (b[:, 1].astype(np.int32) << 8) | b[:, 2].astype(np.int32)
        sign = out & 0x800000
        out = out - (sign << 1)
        return int(sr), out.astype(np.int32)
    if sampwidth == 4:
        return int(sr), np.frombuffer(raw, dtype=">i4").astype(np.int32)

    raise ValueError(f"Unsupported AIFF sample width: {sampwidth} bytes")


def read_audio_to_int32(uploaded_file) -> tuple[int, np.ndarray]:
    """
    Supports WAV / AIFF.
    Returns (sample_rate, mono_int32).
    """
    name = uploaded_file.name
    data = uploaded_file.getvalue()

    ext = name.lower().split(".")[-1]
    if ext in ("wav", "wave"):
        return read_wav_to_int32(data)
    if ext in ("aif", "aiff"):
        return read_aiff_to_int32(data)

    raise ValueError(f"Unsupported file type: .{ext}")


def int32_to_target_dtype(arr_int32: np.ndarray, bit_depth: int) -> np.ndarray:
    """
    Convert int32 audio to the chosen WAV output dtype.
    Note: scipy.io.wavfile.write supports int16/int32/float32 well.
    For 24-bit WAV specifically, SciPy may write as 32-bit. We'll offer 16/32/float.
    """
    if bit_depth == 16:
        arr = np.clip(arr_int32, -2147483648, 2147483647)
        arr = (arr / 65536.0).round().astype(np.int16)  # int32 -> int16
        return arr
    if bit_depth == 32:
        return arr_int32.astype(np.int32)
    if bit_depth == 32_000:  # float32 marker
        f = np.clip(arr_int32.astype(np.float64) / 2147483647.0, -1.0, 1.0)
        return f.astype(np.float32)

    raise ValueError("Unsupported output format selection.")


st.title("5.1 Mono → 5.1 WAV Builder")
st.caption("Drop the 6 mono stems (WAV/AIFF) and export a single 6-channel 5.1 WAV.")

with st.expander("Expected naming", expanded=True):
    st.markdown(
        """
The app detects the channel from the **token immediately before the extension**:

- `... .L.wav` → **Left**
- `... .R.wav` → **Right**
- `... .C.wav` → **Centre**
- `... .LFE.wav` → **LFE**
- `... .Ls.wav` → **Left Surround** (also accepts `.LS`)
- `... .Rs.wav` → **Right Surround** (also accepts `.RS`)

Example: `SisterAnna 5.1_MIX.LFE.wav`
"""
    )

uploaded = st.file_uploader(
    "Drag & drop your 6 mono files here (WAV/AIFF).",
    type=["wav", "wave", "aif", "aiff"],
    accept_multiple_files=True,
)

pad_shorter = st.checkbox("If lengths differ, pad shorter files with silence to match the longest", value=True)

out_fmt = st.selectbox(
    "Output sample format",
    options=["32-bit PCM (int32)", "16-bit PCM (int16)", "32-bit float (float32)"],
    index=0,
)

if out_fmt.startswith("32-bit PCM"):
    out_depth = 32
elif out_fmt.startswith("16-bit"):
    out_depth = 16
else:
    out_depth = 32_000  # float32 marker

st.divider()

detected = {}
errors = []

if uploaded:
    for f in uploaded:
        code = detect_channel_code(f.name)
        if not code:
            errors.append(f"Could not detect channel code from: **{f.name}** (needs .L/.R/.C/.LFE/.Ls/.Rs before extension)")
            continue
        if code in detected:
            errors.append(f"Duplicate channel **{code}** detected: **{f.name}** and **{detected[code].name}**")
            continue
        detected[code] = f

    # Display status
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
        st.info("Output channel order will be: **L, R, C, LFE, Ls, Rs**")

        if st.button("Create 5.1 WAV", type="primary"):
            try:
                # Read all channels
                srs = {}
                audio = {}
                lengths = {}

                for code, f in detected.items():
                    sr, mono = read_audio_to_int32(f)
                    srs[code] = sr
                    audio[code] = mono
                    lengths[code] = len(mono)

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
                    # pad
                    for k in OUTPUT_ORDER:
                        x = audio[k]
                        if len(x) < max_len:
                            audio[k] = np.pad(x, (0, max_len - len(x)), mode="constant")

                # Interleave into (N, 6)
                stacked = np.stack([audio[k] for k in OUTPUT_ORDER], axis=1)  # int32
                out_data = int32_to_target_dtype(stacked, out_depth)

                # Build output name from one of the stems
                example_name = detected[OUTPUT_ORDER[0]].name
                # remove extension
                no_ext = re.sub(r"\.(wav|wave|aif|aiff)$", "", example_name, flags=re.IGNORECASE)
                # remove channel suffix after last dot
                base = no_ext.rsplit(".", 1)[0] if "." in no_ext else no_ext
                out_name = f"{base}_5.1.wav"

                # Write to bytes
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
