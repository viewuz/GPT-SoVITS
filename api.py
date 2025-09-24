import os
import sys
import traceback
from typing import Generator

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Response, Form, File, UploadFile, Path, Body
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
from scipy.signal import resample

i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")

args = parser.parse_args()

port = args.port
host = '0.0.0.0'
argv = sys.argv

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
print(tts_config)
tts_pipeline = TTS(tts_config)

APP = FastAPI()

speakers = {}


class TTS_Request(BaseModel):
    text: str = None
    sample_rate: int = 0
    streaming: bool = False
    speed_factor: float = 0.6

    top_k: int = 15
    top_p: float = 1
    temperature: float = 1

    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    fragment_interval: float = 0.3
    seed: int = -1
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False


def pack_audio(io_buffer: BytesIO, data: np.ndarray, origin_rate: int, target_rate: int):
    if target_rate == 0 or target_rate == origin_rate:
        io_buffer = io_buffer.write(data.tobytes())
        io_buffer.seek(0)
        return io_buffer
    else:
        num_samples = int(len(data) * target_rate / origin_rate)
        resampled_data = resample(data, num_samples)
        io_buffer.write(resampled_data.astype(data.dtype).tobytes())
        io_buffer.seek(0)
        return io_buffer


def handle_control(command: str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


async def tts_handle(speaker_id: str, req: dict):
    """
    Text to speech handler.
    """
    if speaker_id not in speakers:
        return JSONResponse(status_code=404, content={"message": "speaker not found"})

    (ref_audio_path, prompt_text, prompt_lang) = speakers[speaker_id]

    streaming_mode = req.get("streaming", False)
    return_fragment = req.get("return_fragment", False)
    sample_rate = req.get("sample_rate", 0)

    if streaming_mode or return_fragment:
        req["return_fragment"] = True

    req['aux_ref_audio_paths'] = None
    req['ref_audio_path'] = ref_audio_path
    req['prompt_text'] = prompt_text
    req['prompt_lang'] = prompt_lang
    req['text_lang'] = text_lang

    try:
        tts_generator = tts_pipeline.run(req)

        if streaming_mode:

            def streaming_generator(_generator: Generator):
                for sr, chunk in _generator:
                    yield pack_audio(BytesIO(), chunk, sr, sample_rate).getvalue()

            return StreamingResponse(
                streaming_generator(
                    tts_generator,
                ),
                media_type=f"audio/raw",
            )

        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, sample_rate).getvalue()
            return Response(audio_data, media_type=f"audio/raw")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "tts failed", "Exception": str(e)})


@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})

    handle_control(command)
    return None


@APP.post("/speakers")
async def speakers_post_endpoint(
        id: str = Form(...),
        prompt_wav: UploadFile = File(...),
        prompt_text: str = Form(...),
        prompt_lang: str = Form(...),
):
    try:
        os.makedirs("uploaded_audio", exist_ok=True)
        save_path = os.path.join("uploaded_audio", prompt_wav.filename)

        with open(save_path, "wb") as buffer:
            buffer.write(await prompt_wav.read())

        # tts_pipeline.set_ref_audio(save_path)
        speakers[id] = (save_path, prompt_text, prompt_lang)

        return JSONResponse(status_code=200, content={"message": "success"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})


@APP.post("/speakers/{id}")
async def tts_post_endpoint(
        id: str = Path(..., description="Speaker ID"),
        request: TTS_Request = Body(...)
):
    req = request.dict()
    return await tts_handle(id, req)


@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": "change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


if __name__ == "__main__":
    try:
        uvicorn.run(app=APP, host=host, port=port, workers=1)
    except Exception:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
