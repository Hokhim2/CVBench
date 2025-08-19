import os
import json
import re
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import torch
import numpy as np
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import argparse
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

def add_frames(image_size = (224, 224)):
    
    background_color = (0, 0, 0)  
    text_color = (255, 255, 255)  
    texts = [
        "The video 1", "Video 1 End",
        "The video 2", "Video 2 End",
        "The video 3", "Video 3 End",
        "The video 4", "Video 4 End",
    ]

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except IOError:
        font = ImageFont.load_default()

    images = []
    for i, text in enumerate(texts):
        img = Image.new("RGB", image_size, background_color)
        draw = ImageDraw.Draw(img)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)
        draw.text(text_position, text, font=font, fill=text_color)
        images.append(img)
    return images

BSZ = 1

# parser = argparse.ArgumentParser(description="Evaluation benchmark")
# parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
# parser.add_argument('--file_name', type=str, required=True, help="Name of the file")
# args = parser.parse_args()

MODEL_PATH = "Video-R1/Video-R1-7B"
#MODEL_PATH = "OpenGVLab/VideoChat-R1-thinking_7B"
file_name = "mvr"

TASK_CATEGORIES = [
    "Cross-video Anomaly Detection",
    "Cross-video Scene Recognition",
    "Multi-video Key-Action Recognition",
    "Cross-video Event Retrieval",
    "Cross-video Object Recognition",
    "Multi-video Attribute Recognition",
    "Joint-video Counting",
    "Cross-video Entity Matching",
    "Multi-view Scene Understanding",
    "Multi-video Temporal Reasoning",
    "Joint-video Spatial Navigating",
    "Video Difference Caption",
    "Cross-video Counterfactual Reasoning",
    "Joint-video Summarization",
    "Cross-video Procedural Transfer"
]

llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = 8192 * 2,
    gpu_memory_utilization=0.8,
    limit_mm_per_prompt={"image": 1, "video": 1},
)


sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
    stop_token_ids=[],
)


processor = AutoProcessor.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer


for dataset_name in ['example']:

    OUTPUT_PATH = f"./src/r1-v/eval_results/eval_{dataset_name}_{file_name}_greedy_output.json"
    PROMPT_PATH = f"./src/r1-v/Evaluation/{dataset_name}.json"
    
    if PROMPT_PATH.endswith('.jsonl'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
    elif PROMPT_PATH.endswith('.json'):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError("Input file must be .json or .jsonl")

    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )


    messages = []
    for x in data:
        question = x["question"]
        option = x["options"]
        is_yesno = all(opt.strip().strip(".").lower() in ["yes", "no"] for opt in option)
        post_prompt1 = "Please provide only the single option letter (e.g., A, B, C, D) within the <answer> </answer> tags."
        post_prompt2 = "Please provide only the single option word (e.g., Yes, No) within the <answer> </answer> tags."
        if is_yesno:
            option_prompt = "Select the best answer to the following yes-no question based on the listed all videos. Respond with only the word (Yes or No) of the correct option."
            option_str = "\n".join(option)
            post_prompt = post_prompt2
        else:
            option_prompt = "Select the best answer to the following multiple-choice based on the listed all videos. Respond with only the letter (A, B, C, or D) of the correct option."
            option_str = "\n".join(option)
            post_prompt = post_prompt1

        question = question + "\n" + option_str
        full_prompt = "\n" + option_prompt + "\n" + QUESTION_TEMPLATE.format(Question=question) + "\n" + post_prompt

        visuals = ['/root/autodl-tmp/Video-R1/src/r1-v/Evaluation/CVBench/'+x[f'video_{i + 1}'] for i in range(4) if x[f'video_{i + 1}'] is not None]

        video_content = []
        for j, v in enumerate(visuals):
            v = v.replace("\\", "/")
            v = os.path.normpath(v)
            video_content.append({"type": "video", "video": v, "nframes": 8, "resized_height": 224,"resized_width": 224})
        messages.append([{"role": "user", "content": video_content + [{"type": "text", "text": full_prompt + "\n" + "Please pay close attention to the video frames with special cues that are interspersed at the beginning and end of a video's content. For example, frames with the words 'The video X' represent the beginning of the video called video X, and frames with the words 'Video X End' represent the end of the video called video X."}]}])
            
    print(messages[0])
    print('***************')
    final_output = []
    start_idx = 0
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
                final_output = existing.get("results", [])
                start_idx = len(final_output)
                print(f"Resuming from sample index {start_idx}")
        except Exception as e:
            print(f"Error reading existing output file: {e}")


    def extract_think(output_str):
        pattern = r'<think>\s*(.*?)\s*</think>'
        match = re.search(pattern, output_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def extract_answer(text):
        pattern = r'<answer>\s*(.*?)\s*</answer>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def normalize_number(num_str):
        try:
            num_str = num_str.replace(',', '')
            return float(num_str)
        except Exception as e:
            return None
        
    def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):

        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)
        
        epsilon = 1e-8
        rel_error = torch.abs(pred - target) / (torch.abs(target) + epsilon)
        
        thresholds = torch.arange(start, end + interval/2, interval, dtype=torch.float32)
        
        conditions = rel_error < (1 - thresholds)  
        mra = conditions.float().mean()  
        return mra.item()


    def reward_fn(sample, model_output, question_type):
        try:
            output_ans = extract_answer(model_output)
            if output_ans == '':
                output_ans = model_output
            a = sample.get("answer", "")
            gt_ans = extract_answer(f"<answer>{a}</answer>")
            if question_type == "multiple choice":
                return 1.0 if output_ans.strip() == gt_ans.strip() else 0.0
            elif question_type == "numerical":
                gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
                out_has_decimal = ("." in output_ans) or ("," in output_ans)
                if gt_has_decimal != out_has_decimal:
                    return 0.0
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                return 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0
            elif question_type == "regression":
                gt_number = normalize_number(gt_ans)
                out_number = normalize_number(output_ans)
                if gt_number is None or out_number is None:
                    return 0.0
                mra = mean_relative_accuracy(out_number, gt_number)
                return mra
            else:
                return 0.0
        except Exception as e:
            return 0.0

    mean_acc = []
    mean_mra = []
    task_stats = {category: {"total": 0, "correct": 0} for category in TASK_CATEGORIES}

    for i in tqdm(range(start_idx, len(messages), BSZ), desc="Processing batches"):
        batch_messages = messages[i:i + BSZ]

        prompts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in batch_messages]

        single_video_placeholder = "<|vision_start|><|video_pad|><|vision_end|>"

        processed_prompts = []
        for p in prompts:
            parts = p.split(single_video_placeholder)
            if len(parts) > 1:
                new_prompt = parts[0] + single_video_placeholder + ''.join(parts[1:])
            else:
                new_prompt = p
            processed_prompts.append(new_prompt)
        
        prompts=processed_prompts

        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(batch_messages, return_video_kwargs=True)

            transform = T.ToTensor()
            text_images = add_frames(image_size=(224, 224))
            all_frames = []
            for ii, vv in enumerate(video_inputs):
                start_text_img = text_images[ii * 2]
                end_text_img = text_images[ii * 2 + 1]
                start_tensor = (transform(start_text_img) * 255).to(torch.uint8)
                end_tensor = (transform(end_text_img) * 255).to(torch.uint8)
                full_video = torch.cat([start_tensor.unsqueeze(0),vv,end_tensor.unsqueeze(0)], dim=0)  # (10, 3, 224, 224)
                all_frames.append(full_video)

            video_inputs=torch.cat(all_frames, dim=0)
            video_inputs=[video_inputs]

            image_idx = 0
            video_idx = 0

            llm_inputs = []

            
            for idx, prompt in enumerate(prompts):
                mm_type = batch_messages[idx][0]['content'][0]['type']
                sample_mm_data = {}
                sample_video_kw = {}
                if mm_type == 'image':
                    sample_mm_data["image"] = image_inputs[image_idx]
                    image_idx += 1
                elif mm_type == 'video':
                    sample_mm_data["video"] = video_inputs[video_idx]
                    for key, value in video_kwargs.items():
                        sample_video_kw[key] = value[video_idx]
                    video_idx += 1
                        
                
                llm_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": sample_mm_data,
                    "mm_processor_kwargs": sample_video_kw,
                })
                

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            batch_output_text = [out.outputs[0].text for out in outputs]
            
        except Exception as e:
            print('error:', data[i]['path'])
            print('Exception:', e)
            batch_output_text = ['<answer>error</answer>'] * BSZ
            

        for j, (sample, model_output) in enumerate(zip(data[i:i+BSZ], batch_output_text), start=i):
            think_chain = extract_think(model_output)
            final_ans = extract_answer(model_output)
            if final_ans == "":
                final_ans = model_output
            sample["output"] = model_output
            sample["prediction"] = final_ans
            #q_type = sample.get("problem_type", "")
            sample["reward"] = reward_fn(sample, model_output, 'multiple choice')
            sample['correct'] = True if sample["reward"]==1.0 else False

            mean_acc.append(sample["reward"])
            task_category = sample.get("task_type", "Unknown")
            if task_category in task_stats:
                task_stats[task_category]["total"] += 1
                if sample['correct']:
                    task_stats[task_category]["correct"] += 1

            # if sample['problem_type'] != 'regression':
            #     mean_acc.append(sample["reward"])
            # else:
            #     mean_mra.append(sample["reward"])

            if think_chain:
                sample["process"] = f"<think>{think_chain}</think>"
            final_output.append(sample)
        

        try:
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump({"results": final_output}, f, indent=2, ensure_ascii=False)
            print(f"Processed batch {(i - start_idx)//BSZ + 1}, saved {len(final_output)} samples.")
        except Exception as e:
            print(f"Error writing to output file: {e}")

    task_results = []
    for category in TASK_CATEGORIES:
        stats = task_stats[category]
        task_results.append({
            "Task Category": category,
            "Total Questions": stats["total"],
            "Correct Answers": stats["correct"],
            "Accuracy": f"{(stats['correct'] / stats['total'] * 100):.1f}%" if stats['total'] > 0 else "N/A"
        })

    final_acc={'mean_acc': 0.0, 'mean_mra': 0.0}
    final_acc['mean_acc'] = torch.tensor(mean_acc).mean().item()
    if mean_mra != []:
        final_acc['mean_mra'] = torch.tensor(mean_mra).mean().item()
    
    try:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump({"results": final_output, "final_acc": [final_acc], "task_acc": task_results}, f, indent=2, ensure_ascii=False)
        print(f"Final accuracy saved to {OUTPUT_PATH}")
    except Exception as e:
        print(f"Error writing final accuracy to output file: {e}")
    
    print(f"Results saved to {OUTPUT_PATH}")
