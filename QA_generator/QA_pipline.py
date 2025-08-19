import base64
import json
import os
import time
from io import BytesIO
import numpy as np
import math
import re
from format_json import process_directory

try:
    from decord import VideoReader, cpu
except ImportError:
    pass
from openai import OpenAI
from PIL import Image

os.environ['OPENAI_API_KEY'] = "XXXXXXXXXXXXXXXXXX"
os.environ['OPENAI_API_URL'] = "XXXXXXXXXXXXXXXXXX"


def encode_video(video_path):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = vr.get_avg_fps()
    duration_seconds = np.round((total_frame_num / fps)).astype(int)
    if 180<duration_seconds<=300:
        duration_seconds=np.round((duration_seconds / 2)).astype(int)
    elif 300<duration_seconds<=480:
        duration_seconds = np.round((duration_seconds / 3)).astype(int)
    elif 480<duration_seconds<=600:
        duration_seconds = np.round((duration_seconds / 4)).astype(int)

    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, duration_seconds, dtype=int)

    # Ensure the last frame is included
    if total_frame_num - 1 not in uniform_sampled_frames:
        uniform_sampled_frames = np.append(uniform_sampled_frames, total_frame_num - 1)

    frame_idx = uniform_sampled_frames.tolist()
    frames = vr.get_batch(frame_idx).asnumpy()

    base64_frames = []
    for frame in frames:
        img = Image.fromarray(frame)
        img = img.resize((336, 336))
        output_buffer = BytesIO()
        img.save(output_buffer, format="PNG")
        byte_data = output_buffer.getvalue()
        base64_str = base64.b64encode(byte_data).decode("utf-8")
        base64_frames.append(base64_str)

    return base64_frames


def flatten(input):
    new_list = []
    for i in input:
        for j in i:
            new_list.append(j)
    return new_list


def chunker(seq, size):
    for i in range(0, len(seq), size):
        chunk = seq[i:i + size]
        start = i
        end = i + len(chunk) - 1
        yield chunk, (start, end)


def seconds_to_mmss(total_seconds):
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"

def normalize_qa_format(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for qa in data.get("QA_pairs", []):
        options = qa.get("Options")

        if isinstance(options, str):
            split_opts = re.findall(r'[A-D]\. ?[^A-D]*', options)
            if not split_opts:
                split_opts = ["A. ", "B. ", "C. ", "D. "]
            qa["Options"] = [opt.strip() for opt in split_opts]

        elif isinstance(options, list) and set(opt.strip() for opt in options) == {"Yes.", "No."}:
            qa["Answer"] = qa["Answer"].replace(".", "").strip()

        elif isinstance(options, list) and all(re.match(r"^[A-D]\. ", opt) for opt in options):
            match = re.match(r"^([A-D])\.", qa["Answer"])
            if match:
                qa["Answer"] = match.group(1)

    return data

def single_video_caption(video_root, prompt):
    model_version: str = "gpt-4o"
    timeout: int = 120
    max_retries: int = 5
    max_size_in_mb: int = 20
    max_tokens: int = 4096

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))

    root = video_root
    root_files = os.listdir(root)
    video_files = [os.path.join(root, f) for f in root_files]

    res = []
    for videos in video_files:

        visuals = [os.path.join(videos, f) for f in os.listdir(videos) if ".mp4" in f]

        imgs = []  # frames for video
        for visual in visuals:
            if isinstance(visual, str) and (
                    ".mp4" in visual or ".avi" in visual or ".mov" in visual or ".flv" in visual or ".wmv" in visual):
                frames = encode_video(visual)
                imgs.extend([frames])
            else:
                raise ValueError("Videos are not found!.")

        post_prompt = "Based on the given video frames, provide a detailed caption."
        payload = {"messages": []}
        payload["model"] = model_version
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0
        # payload["top_p"] = None
        # payload["num_beams"] = 1
        payload["messages"].append({"role": "system", "content": prompt})
        payload["messages"].append({"role": "user", "content": []})
        for i, img in enumerate(imgs):
            batch_size = 10
            frame_batches = list(chunker(img, batch_size))
            video_caption = ""
            for batch_idx, (batch, (start_idx, end_idx)) in enumerate(frame_batches):
                payload["messages"][1]["content"].append({"type": "text",
                                                          "text": f"{post_prompt} (Video {i + 1}, Frames: {start_idx}-{end_idx}/{len(img)})"})
                for frame in batch:
                    payload["messages"][1]["content"].append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{frame}"}})
                for attempt in range(max_retries):
                    try:
                        response = client.chat.completions.create(**payload)
                        response_text = response.choices[0].message.content
                        video_caption += f"{seconds_to_mmss(start_idx)}-{seconds_to_mmss(end_idx)}: {response_text}\n"
                        break  # If successful, break out of the loop

                    except Exception as e:
                        error_msg = str(e)
                        print(f"Attempt {attempt + 1}/{max_retries} failed with error: {error_msg}")

                        # On last attempt, log error and set empty response
                        if attempt == max_retries - 1:
                            print(f"All {max_retries} attempts failed. Last error: {error_msg}")
                            response_text = ""
                        else:
                            time.sleep(timeout)
                payload["messages"][1]["content"].clear()
            res.append({"VideoID": i + 1, "Caption": video_caption})
        with open(videos + '/single_video_captions.json', "w", encoding="utf-8") as f:
            json.dump(res, f)
        res.clear()
    return print("Single_caption Done!")

def inter_video_caption(video_root, prompt):
    model_version: str = "gpt-4o"
    timeout: int = 120
    max_retries: int = 5
    max_size_in_mb: int = 20
    max_tokens: int = 4096

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))

    root = video_root
    root_files = os.listdir(root)
    video_files = [os.path.join(root, f) for f in root_files]

    res = []
    for videos in video_files:

        visuals = [os.path.join(videos, f) for f in os.listdir(videos) if ".mp4" in f]

        payload = {"messages": []}
        payload["model"] = model_version
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0
        # payload["top_p"] = None
        # payload["num_beams"] = 1
        captions = [os.path.join(videos, f) for f in os.listdir(videos) if ".json" in f]
        single_caption = next((f for f in captions if "single" in f.lower()), None)
        with open(single_caption, "r", encoding="utf-8") as f:
            single_data = json.load(f)
        payload["messages"].append({"role": "system", "content": prompt})
        payload["messages"].append({"role": "user", "content": []})
        payload["messages"][1]["content"].append(
            {"type": "text", "text": "Based on the given captions, provide a detailed summary between videos."})

        for i in range(len(visuals)):
            payload["messages"][1]["content"].append(
                {"type": "text",
                 "text": f"Video {single_data[i]['VideoID']}: caption: \n{single_data[i]['Caption']}\n"})
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content
                break  # If successful, break out of the loop

            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1}/{max_retries} failed with error: {error_msg}")

                # On last attempt, log error and set empty response
                if attempt == max_retries - 1:
                    print(f"All {max_retries} attempts failed. Last error: {error_msg}")
                    response_text = ""
                else:
                    time.sleep(timeout)
        res.append({"Summary": response_text})
        with open(videos + '/inter_video_captions.json', "w", encoding="utf-8") as f:
            json.dump(res, f)
        res.clear()
    return print("Inter_summary Done!")

def QA_generator(video_root, prompt):
    model_version: str = "gpt-4o"
    timeout: int = 120
    max_retries: int = 5
    max_size_in_mb: int = 20
    max_tokens: int = 4096

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_API_URL"))

    root = video_root
    root_files = os.listdir(root)
    video_files = [os.path.join(root, f) for f in root_files]

    res = []
    for videos in video_files:

        captions = [os.path.join(videos, f) for f in os.listdir(videos) if ".json" in f]
        single_caption = next((f for f in captions if "single" in f.lower()), None)
        inter_caption = next((f for f in captions if "inter" in f.lower()), None)

        with open(single_caption, "r", encoding="utf-8") as f:
            single_data = json.load(f)
        with open(inter_caption, "r", encoding="utf-8") as f:
            inter_data = json.load(f)

        post_prompt = "Please generate high-quality questions focusing on the  correlations, matches and differences between videos."
        payload = {"messages": []}
        payload["model"] = model_version
        payload["max_tokens"] = max_tokens
        payload["temperature"] = 0
        # payload["top_p"] = None
        # payload["num_beams"] = 1

        payload["messages"].append({"role": "system", "content": prompt})
        payload["messages"].append({"role": "user", "content": []})
        payload["messages"][1]["content"].append({"type": "text", "text": post_prompt})
        single_context = "Here is the list of individual captions for each video:\n"
        for item in single_data:
            single_context += f"VideoID {item['VideoID']}: caption: {item['Caption']}\n"
        payload["messages"][1]["content"].append({"type": "text", "text": single_context})
        inter_context = "Here are the interconnections between the videos, including differences, correlations, matches, etc.:\n"
        inter_context += inter_data[0]['Summary']
        payload["messages"][1]["content"].append({"type": "text", "text": inter_context})
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(**payload)
                response_text = response.choices[0].message.content
                break  # If successful, break out of the loop

            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1}/{max_retries} failed with error: {error_msg}")

                # On last attempt, log error and set empty response
                if attempt == max_retries - 1:
                    print(f"All {max_retries} attempts failed. Last error: {error_msg}")
                    response_text = ""
                else:
                    time.sleep(timeout)
        res.append(response_text)
        with open(videos + '/QAs.json', "w", encoding="utf-8") as f:
            json.dump(res, f)
        res.clear()
    return print('QAs Done!')


if __name__ == "__main__":
    with open("./single_video_caption.md", "r", encoding="utf-8") as f:
        single_prompt = f.read()
    with open("./Inter-video_summary.md", "r", encoding="utf-8") as f:
        inter_prompt = f.read()
    with open("./QA_prompt.md", "r", encoding="utf-8") as f:
        QA_prompt = f.read()

    single_video_caption(video_root="./dataset/", prompt=single_prompt)
    inter_video_caption(video_root="./dataset/", prompt=inter_prompt)
    QA_generator(video_root="./dataset/", prompt=QA_prompt)

    root = "./dataset/"
    process_directory(root)
    print("Format done!")