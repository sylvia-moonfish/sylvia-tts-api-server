import concurrent.futures
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from GPT_SoVITS.AR.models.t2s_lightning_module import Text2SemanticLightningModule
from GPT_SoVITS.feature_extractor import cnhubert
from GPT_SoVITS.module.mel_processing import spectrogram_torch
from GPT_SoVITS.module.models import SynthesizerTrn
from GPT_SoVITS.text import cleaned_text_to_sequence
from GPT_SoVITS.text.cleaner import clean_text
from io import BytesIO
import LangSegment
import librosa
import multiprocessing
import numpy as np
import os
from scipy import signal as scipy_signal
import signal
import soxr
from time import time as ttime
from tools.my_utils import load_audio
import torch
import traceback
import uvicorn

# Define CUDA device here.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Set up hubert model.
cnhubert.cnhubert_base_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
ssl_model = cnhubert.get_model()
ssl_model = ssl_model.to(device)


def cut(text):
    """
    Separate the sentences to sentence array based on english punctuations.
    Float point numbers (i.e. 1.234) are preserved.
    """
    text = text.strip("\n")
    punct = {",", ".", ";", "?", "!"}
    mergeitems = []
    items = []

    for i, char in enumerate(text):
        if char in punct:
            # If the . was for float point, keep appending.
            if (
                char == "."
                and i > 0
                and i < len(text) - 1
                and text[i - 1].isdigit()
                and text[i + 1].isdigit()
            ):
                items.append(char)
            else:
                items.append(char)
                mergeitems.append("".join(items))
                items = []
        else:
            items.append(char)

    if items:
        mergeitems.append("".join(items))

    # Filter out any element with just punctuation in it.
    output = [item for item in mergeitems if not set(item).issubset(punct)]

    # Return as multi-sentence string separated by newline.
    return "\n".join(output)


def process_text(texts):
    """
    Filter out empty sentences.
    """
    _text = []

    if all(text in [None, " ", "\n", ""] for text in texts):
        raise ValueError("Invalid text")

    for text in texts:
        if text in [None, " ", ""]:
            pass
        else:
            _text.append(text)

    return _text


def merge_short_text_in_array(texts, threshold):
    """
    If there's sentences shorter than the threshold, combine
    them with the next sentence.
    """
    if (len(texts)) < 2:
        return texts

    result = []
    text = ""

    for element in texts:
        text += element

        if len(text) >= threshold:
            result.append(text)
            text = ""

    if len(text) > 0:
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text

    return result


def clean_text_inf(text, language, version):
    """
    Clean given text using GPT_SoVITS cleaner.
    Returns phonemes and normalized texts.
    """
    phones, word2ph, norm_text = clean_text(text, language, version)
    phones = cleaned_text_to_sequence(phones, version)

    return phones, word2ph, norm_text


class DictToAttrRecursive(dict):
    """
    Converts dict to object with attributes.
    """

    def __init__(self, input_dict):
        super().__init__(input_dict)

        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)

            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)

        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


def get_phones_and_bert(text, language, version, final=False):
    """
    Figure out language for each sentence then
    get the phonemes and bert.
    """
    if language in {"en", "all_ko"}:
        language = language.replace("all_", "")

        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            formattext = text

        while "  " in formattext:
            formattext = formattext.replace("  ", " ")

        phones, word2ph, norm_text = clean_text_inf(formattext, language, version)
        bert = torch.zeros((1024, len(phones)), dtype=torch.float32).to(device)
    elif language in {"ko", "auto"}:
        textlist = []
        langlist = []
        LangSegment.setfilters(["ja", "en", "ko"])

        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)

                textlist.append(tmp["text"])

        print(textlist)
        print(langlist)

        phones_list = []
        bert_list = []
        norm_text_list = []

        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang, version)
            bert = torch.zeros((1024, len(phones)), dtype=torch.float32).to(device)

            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)

        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert("." + text, language, version, final=True)

    return phones, bert.to(torch.float32), norm_text


def get_spepc(hps, filename):
    """
    Get spectrogram.
    """
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    _max = audio.abs().max()

    if _max > 1:
        audio /= min(2, _max)

    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)

    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )

    return spec


gpt_cache = {}


def change_gpt_weights(gpt_path):
    """
    Load GPT weight from given path.
    Load from cache if cached.
    """
    global hz, max_sec, t2s_model, config, gpt_cache
    hz = 50

    if gpt_path in gpt_cache:
        print("GPT cache hit! ", gpt_path)
        dict_s1 = gpt_cache[gpt_path]["dict_s1"]
    else:
        dict_s1 = torch.load(gpt_path, map_location="cpu")

    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]

    if gpt_path in gpt_cache:
        t2s_model = gpt_cache[gpt_path]["t2s_model"]
    else:
        t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
        t2s_model.load_state_dict(dict_s1["weight"])
        t2s_model = t2s_model.to(device)
        t2s_model.eval()

    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    gpt_cache[gpt_path] = {"dict_s1": dict_s1, "t2s_model": t2s_model}


sovits_cache = {}


def change_sovits_weights(sovits_path):
    """
    Load SoVITS weight from given path.
    Load from cache if cached.
    """
    global vq_model, hps, version, sovits_cache

    if sovits_path in sovits_cache:
        print("SoVITS cache hit! ", sovits_path)
        dict_s2 = sovits_cache[sovits_path]["dict_s2"]
    else:
        dict_s2 = torch.load(sovits_path, map_location="cpu")

    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    hps.model.version = "v2"
    version = hps.model.version

    if sovits_path in sovits_cache:
        vq_model = sovits_cache[sovits_path]["vq_model"]
    else:
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model,
        )
        del vq_model.enc_q
        vq_model.to(device)
        vq_model.eval()

    sovits_cache[sovits_path] = {"dict_s2": dict_s2, "vq_model": vq_model}
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))


splits = {",", ".", "?", "!", "~", ":", "-"}


def get_tts_audio(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    top_k=15,
    top_p=1,
    temperature=1,
    speed=1,
):
    """
    Generate TTS audio with given data.
    Returns ndarray float32 raw PCM audio data.
    """

    t = []
    t0 = ttime()

    prompt_text = prompt_text.strip("\n")

    if prompt_text[-1] not in splits:
        prompt_text += "."

    print("Actual Input Reference Text: ", prompt_text)

    text = text.strip("\n")

    print("Actual Input Target Text: ", text)

    zero_wav = np.zeros(int(hps.data.sampling_rate * 0.3), dtype=np.float32)

    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)

        if wav16k.shape[0] > 160000 or wav16k.shape[0] < 48000:
            raise OSError(
                "Reference audio is outside the 3-10 second range, please choose another one!"
            )

        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        wav16k = wav16k.to(device)
        zero_wav_torch = zero_wav_torch.to(device)

        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(1, 2)
        codes = vq_model.extract_latent(ssl_content)
        prompt_semantic = codes[0, 0]
        prompt = prompt_semantic.unsqueeze(0).to(device)

    t1 = ttime()
    t.append(t1 - t0)

    # text = cut(text)

    while "\n\n" in text:
        text = text.replace("\n\n", "\n")

    print("Actual Input Target Text (after sentence segmentation):", text)

    texts = text.split("\n")
    texts = process_text(texts)
    texts = merge_short_text_in_array(texts, 5)

    audio_output = []
    phones1, bert1, norm_text1 = get_phones_and_bert(
        prompt_text, prompt_language, version
    )

    for i_text, text in enumerate(texts):
        if len(text.strip()) == 0:
            continue

        if text[-1] not in splits:
            text += "."

        print("Actual Input Target Text (per sentence): ", text)

        phones2, bert2, norm_text2 = get_phones_and_bert(text, text_language, version)

        print("Processed text from the frontend (per sentence): ", norm_text2)

        bert = torch.cat([bert1, bert2], 1)
        all_phoneme_ids = torch.LongTensor(phones1 + phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        t2 = ttime()

        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                prompt,
                bert,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        t3 = ttime()

        refers = [get_spepc(hps, ref_wav_path).to(torch.float32).to(device)]

        audio = (
            vq_model.decode(
                pred_semantic,
                torch.LongTensor(phones2).to(device).unsqueeze(0),
                refers,
                speed=speed,
            )
            .detach()
            .cpu()
            .numpy()[0, 0]
        )
        max_audio = np.abs(audio).max()

        if max_audio > 1:
            audio /= max_audio

        audio_output.append(audio)
        audio_output.append(zero_wav)

        t4 = ttime()
        t.extend([t2 - t1, t3 - t2, t4 - t3])
        t1 = ttime()

    print("%.3f\t%.3f\t%.3f\t%.3f" % (t[0], sum(t[1::3]), sum(t[2::3]), sum(t[3::3])))

    yield hps.data.sampling_rate, np.concatenate(audio_output, 0).astype(np.float64)


def trim_silence(audio_data, threshold_db=-40, window_size=1024):
    """
    Trim silence from the beginning and end of audio data based on dB threshold.
    audio_data (np.ndarray): dtype float64.
    """

    # Convert amplitude to dB.
    def to_db(x):
        # Add small number to prevent log of 0.
        return 20 * np.log10(np.abs(x) + 1e-10)

    # Calculate dB values for each window.
    def get_window_db(data, start_idx):
        end_idx = min(start_idx + window_size, len(data))
        window = data[start_idx:end_idx]

        return np.max(to_db(window))

    # Find start index (first window above threshold)
    start_idx = 0

    while start_idx < len(audio_data):
        if get_window_db(audio_data, start_idx) > threshold_db:
            break

        start_idx += window_size

    # Find end index (last window above threshold)
    end_idx = len(audio_data)

    while end_idx > start_idx:
        if get_window_db(audio_data, max(0, end_idx - window_size)) > threshold_db:
            break

        end_idx -= window_size

    # Return trimmed audio.
    return audio_data[start_idx:end_idx]


def normalize_audio(audio_data, target_db=-1.0):
    """
    Normalize the gain of given audio. (dtype=float64)
    """

    # Find the peak amplitude.
    peak_amplitude = np.max(np.abs(audio_data))

    # Convert target dB to amplitude.
    target_amplitude = 10 ** (target_db / 20.0)

    # Calculate scaling factor.
    scaling_factor = target_amplitude / peak_amplitude

    # Apply normalization.
    normalized_audio = audio_data * scaling_factor

    # Ensure we do not exceed [-1, 1] range.
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)

    return normalized_audio


def process_audio_soxr(
    audio_data,
    original_sample_rate=32000,
    target_sample_rate=48000,
):
    """
    Process PCM audio data using SOXR library.
    """

    t1 = ttime()

    trimmed_audio = trim_silence(audio_data)

    t2 = ttime()

    normalized_audio = normalize_audio(trimmed_audio)

    t3 = ttime()

    resampled_audio = soxr.resample(
        normalized_audio, original_sample_rate, target_sample_rate, quality="HQ"
    )

    t4 = ttime()

    print(
        "Trim: %.3f, Normalize: %.3f, Resample: %.3f"
        % ((t2 - t1), (t3 - t2), (t4 - t3))
    )

    # Convert to int16.
    resampled_audio = (resampled_audio * 32768).astype(np.int16)

    # Convert to stereo if needed.
    if len(resampled_audio.shape) == 1:
        stereo_audio = np.column_stack((resampled_audio, resampled_audio))

        return stereo_audio
    else:
        return resampled_audio


def lanczos_kernel(x, a=3):
    """
    Compute the Lanczos kernel.
    Kernel size parameter default to 3.
    """

    x = np.asarray(x)
    kernel = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) < a
    x_mask = x[mask]
    kernel[mask] = (
        a
        * np.sin(np.pi * x_mask)
        * np.sin(np.pi * x_mask / a)
        / (np.pi * np.pi * x_mask * x_mask)
    )
    kernel[x == 0] = 1

    return kernel


def process_chunk(data, start, end, total_samples, scale, a=3):
    """
    Process a chunk of samples using vectorized operations.
    """
    chunk_size = end - start
    result = np.zeros(chunk_size, dtype=np.float64)

    # Calculate the corresponding input positions for each output sample.
    # This maps our output sample positions back to input sample positions.
    input_positions = start * scale + np.arange(chunk_size) * scale

    # Calculate indices and weights for all positions at once.
    indices = np.arange(-a, a + 1)
    sample_indices = input_positions[:, np.newaxis] + indices
    sample_indices = np.clip(sample_indices, 0, total_samples - 1).astype(int)

    # Calculate weights for all positions at once.
    x_values = (input_positions[:, np.newaxis] - sample_indices) / scale
    weights = lanczos_kernel(x_values, a)

    # Apply weights to all positions at once.
    for i in range(chunk_size):
        samples = data[sample_indices[i]]
        result[i] = np.sum(samples * weights[i]) / np.sum(weights[i])

    return result


def lanczos_resample(data, num_samples, a=3, num_workers=None):
    """
    Resample data using Lanczos interpoliaton.
    """

    # If worker count is unspecified, use CPU count.
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)

    # Calculate the scaling factor for the kernenl.
    scale = len(data) / num_samples

    # Determine chunk size (process approx 1s of audio per chunk).
    chunk_size = min(num_samples // num_workers, 48000)  # 1s at 48kHz
    chunks = [
        (i, min(i + chunk_size, num_samples)) for i in range(0, num_samples, chunk_size)
    ]

    # Process chunks in parallel.
    result = np.zeros(num_samples, dtype=np.float64)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                process_chunk,
                data=data,
                start=start,
                end=end,
                total_samples=len(data),
                scale=scale,
                a=a,
            )
            for start, end in chunks
        ]

        # Collect results.
        for (start, end), future in zip(chunks, futures):
            result[start:end] = future.result()

    return result


def process_audio(
    audio_data,
    original_sample_rate=32000,
    target_sample_rate=48000,
    convert_to_stereo=True,
    num_workers=None,
):
    """
    Process PCM audio data to resample it to target sample rate.
    Return it in 16-bit PCM stereo.
    """

    t1 = ttime()

    # Normalize to [-1, 1] range if needed.
    max_val = np.max(np.abs(audio_data))

    if max_val > 1.0:
        audio_data = audio_data / max_val

    # Calculate the number of output samples.
    num_samples = int(
        round(len(audio_data) * target_sample_rate / original_sample_rate)
    )

    # Apply anti-aliasing filter before resampling.
    nyquist = min(target_sample_rate, original_sample_rate) / 2
    cutoff = 0.9 * nyquist  # Leave some margin to avoid edge effects.
    sos = scipy_signal.butter(
        8, cutoff, btype="low", fs=original_sample_rate, output="sos"
    )
    filtered_audio = scipy_signal.sosfilt(sos, audio_data)

    t2 = ttime()

    # Perform Lanczos resampling.
    resampled_audio = lanczos_resample(
        filtered_audio, num_samples, num_workers=num_workers
    )

    t3 = ttime()

    print("AA: %.3f, Resample: %.3f" % ((t2 - t1), (t3 - t2)))

    # Convert to int16.
    resampled_audio = (resampled_audio * 32768).astype(np.int16)

    # Convert to stereo if needed.
    if convert_to_stereo and len(resampled_audio.shape) == 1:
        stereo_audio = np.column_stack((resampled_audio, resampled_audio))

        return stereo_audio
    else:
        return resampled_audio


def pack_audio(io_buffer: BytesIO, data: np.ndarray):
    """
    Pack ndarray audio into BytesIO.
    """

    data = process_audio_soxr(data)
    io_buffer.write(data.tobytes())
    io_buffer.seek(0)

    return io_buffer


def tts(
    ref_wav_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
    top_k=15,
    top_p=1,
    temperature=1,
    speed=1,
):
    """
    Main TTS call to generate audio data and process them.
    """

    try:
        tts_generator = get_tts_audio(
            ref_wav_path,
            prompt_text,
            prompt_language,
            text,
            text_language,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            speed=speed,
        )
        sr, audio_data = next(tts_generator)
        audio_data = pack_audio(BytesIO(), audio_data).getvalue()

        return Response(audio_data, media_type="audio/raw")

    except Exception as e:
        return JSONResponse(
            status_code=400, content={"message": "TTS failed.", "exception": str(e)}
        )


APP = FastAPI()


@APP.get("/tts")
async def tts_get(
    gpt_path,
    sovits_path,
    prompt_path,
    prompt_text,
    prompt_language,
    text,
    text_language,
):
    change_gpt_weights(gpt_path)
    change_sovits_weights(sovits_path)

    return tts(prompt_path, prompt_text, prompt_language, text, text_language)


if __name__ == "__main__":
    try:
        uvicorn.run(app=APP, host="127.0.0.1", port=9880, workers=1)
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
