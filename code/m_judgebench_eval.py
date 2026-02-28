import json
import random
import torch
import tqdm
from PIL import Image
import warnings
import os
from transformers import AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from datasets import load_dataset
import argparse
import re

warnings.filterwarnings("ignore")

def read_jsonl(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def main(args):

    if not args.use_transformers:
        from vllm import LLM, EngineArgs, SamplingParams
        llm = LLM(model=args.model_path, gpu_memory_utilization=0.9, tensor_parallel_size=args.tp, limit_mm_per_prompt={"image": 50, "video": 10})
        sampling_params = SamplingParams(
                temperature=0.001,
                top_p=0.001,
                repetition_penalty=1.05,
                max_tokens=2048,
                stop_token_ids=[]
            )
    else:
        # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     args.model_path, torch_dtype="auto", device_map="auto"
        # )
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda()
    
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(
        args.model_path, min_pixels=min_pixels, max_pixels=max_pixels
    )


    total_cnt = 0
    total_corr = 0

    dataset = read_jsonl(args.benchmark_path)

    group_correct = {'shortcot_same_model': 0, 'shortcot_diff_model': 0, 'longcot_same_model': 0, 'longcot_diff_model': 0, 'incidental_error_bygpt': 0, 'reasoning_error_bygpt': 0, 'visual_error_bygpt': 0, 'imbalanced_length_short_answer_correct': 0, 'imbalanced_length_shortcot_correct': 0, 'imbalanced_length_longcot_correct': 0}
    group_total = {'shortcot_same_model': 0, 'shortcot_diff_model': 0, 'longcot_same_model': 0, 'longcot_diff_model': 0, 'incidental_error_bygpt': 0, 'reasoning_error_bygpt': 0, 'visual_error_bygpt': 0, 'imbalanced_length_short_answer_correct': 0, 'imbalanced_length_shortcot_correct': 0, 'imbalanced_length_longcot_correct': 0}

    correct = 0
    random.seed(0)

    for i in tqdm.trange(len(dataset)):
        total_cnt += 1
        data = dataset[i]

        if isinstance(data['image'], list):
            images = [{"type": "image", "image": args.image_base_path + image} for image in data['image']]
        else:
            images = [{"type": "image", "image": args.image_base_path + data['image']}]

        chosen_response, reject_response = data['chosen'], data['rejected']

        if random.random() < 0.5:
            R1, R2, answer = chosen_response, reject_response, "Answer 1 is better"
        else:
            R1, R2, answer = reject_response, chosen_response, "Answer 2 is better"

        Query = data["question"]


        prompt_text_unified_short = (
            "You are given an image and a question related to it. Your job is to evaluate the two responses based on these five factors:\n\n"
            "1. Accuracy of Object Descriptions: Review how accurately the objects are described in the responses, ensuring they match those in the ground truth. Be mindful of irrelevant or incorrect objects being mentioned.\n\n"
            "2. Relationship Between Objects: Check if the response properly describes how the objects relate to each other, reflecting their actual positions or interactions, as seen in the image.\n\n"
            "3. Description of Attributes: Assess how well the response captures the attributes (e.g., size, color, shape) of the objects in the image, in line with the ground truth.\n\n"
            "4. Helpfulness: Consider whether the response offers useful information that enhances the understanding of the image. Does it add context or provide extra insights? Also, evaluate whether it follows the instructions given in the prompt.\n\n"
            "5. Ethical Concerns: Review the response to ensure it avoids sensitive, harmful, or inappropriate content. The response should be fair, respect privacy, and be free of bias or offensive material.\n\n"
            "Please prioritize selecting the response with the most accurate answer as chosen, then analyze whether the reasoning process is thorough and correct." # v2 update
            "After evaluating both answers, determine which one is better based on these factors and clearly state your decision, such as 'Answer 1 is better' or 'Answer 2 is better.'\n\n"
            f"Question: {Query}\n"
            f"Answer 1: {R1}\n"
            f"Answer 2: {R2}\n"
        )

        prompt_text_unified_think = (
            "Given a question and a reference image, please analyze in detail the two provided answers (Answer 1 and Answer 2). "
            "Evaluate them based on the following three core dimensions:\n"
            "1. Semantic accuracy: How well the answer reflects the visual content of the image\n"
            "2. Correctness: Whether the answer is logically and factually correct\n"
            "3. Clarity: Whether the answer is clearly and fluently expressed\n"
            "For each dimension, provide a score from 0 to 5 for both answers, and briefly explain your reasoning. "
            "Then, compute the total score for each answer by explicitly adding the scores for all dimensions and showing the full calculation. "
            "Enclose your full reasoning within <think> and </think> tags. "
            "Then, in the <answer> tag, output exactly one of the following: 'Answer 1 is better' or 'Answer 2 is better'. No other text is allowed in the <answer> section.\n\n"
            "Example format:\n"
            "<think>\n"
            "1. Semantic accuracy: Answer 1 (X/5) - ...; Answer 2 (X/5) - ...\n"
            "2. Correctness: Answer 1 (X/5) - ...; Answer 2 (X/5) - ...\n"
            "3. Clarity: Answer 1 (X/5) - ...; Answer 2 (X/5) - ...\n"
            "Total score:\nAnswer 1: X+X+X=Y\nAnswer 2: X+X+X=Y\n"
            "</think>\n"
            "<answer>Answer X is better</answer>\n\n"
            "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n"
            "Please prioritize selecting the response with the most accurate answer as chosen, then analyze whether the reasoning process is thorough and correct." # v2 update
            f"Your task is provided as follows:\nQuestion: [{Query}]\n"
            f"Answer 1: [{R1}]\n"
            f"Answer 2: [{R2}]\n"
        )

        prompt_text_r1reward_think = (
            "You are a highly skilled and impartial evaluator tasked with comparing two responses generated by a Large Multimodal Model for a given question. "
            "Please prioritize selecting the response with the most accurate answer as chosen, then analyze whether the reasoning process is thorough and correct." # v2 update
            "- Start with a thorough, side-by-side comparative analysis enclosed within <think> and </think> tags. A tie is not permitted; you must choose a better option.\n\n"
            "- Conclude with a single numeric choice enclosed within <answer> and </answer> tags:\n"
            "  - Output \"1\" if Response 1 is better.\n"
            "  - Output \"2\" if Response 2 is better.\n\n"
            "###### **Input:**  \n"
            f"###### [Question]:\n{Query}  \n\n"
            f"###### [Response 1]:\n{R1}  \n\n"
            f"###### [Response 2]:\n{R2}  \n\n"
            "###### **Output Format (strictly follow):**  \n"
            "<think>Your detailed comparative analysis goes here</think><answer>1/2</answer>"
        )

        prompt_text = (
            "You are given an image and a question related to it. Your task is to evaluate two multimodal reasoning responses based on the following five criteria:\n\n"
            "1. **Answer Correctness:** Check whether each response provides the most accurate and relevant final answer to the given question. Prioritize factual correctness above all other factors.\n\n"
            "2. **Reasoning Soundness:** Examine whether the reasoning steps logically lead to the final answer. Identify any reasoning fallacies, invalid inferences, or irrelevant logic chains that might affect reliability.\n\n"
            "3. **Perceptual Understanding:** Assess the correctness of visual grounding — whether objects, regions, or relationships in the image are correctly interpreted and referenced during reasoning.\n\n"
            "4. **Conciseness and Coherence:** Evaluate whether the reasoning process is clear, coherent, and appropriately detailed. The response should neither omit essential reasoning nor over-elaborate with redundant or misleading content. Do not prefer longer reasoning by default.\n\n"
            "5. **Style and Robustness:** Consider whether the model maintains consistent judgment quality across different response styles or model sources. The decision should rely on reasoning quality, not stylistic fluency or writing preference.\n\n"
            "Please prioritize selecting the response with the most accurate final answer as the chosen one, and then consider the thoroughness and correctness of its reasoning process.\n\n"
            "After evaluating both responses, clearly state your decision in the format: 'Answer 1 is better' or 'Answer 2 is better.'\n\n"
            f"Question: {Query}\n"
            f"Answer 1: {R1}\n"
            f"Answer 2: {R2}\n"
            "Make sure you clearly state your decision only, without any additional explanation."
        )

        if args.short_unified:
            messages_input = images + [{"type": "text", "text": prompt_text_unified_short}]
        elif args.cot_unified:
            messages_input = images + [{"type": "text", "text": prompt_text_unified_think}]
        elif args.cot_r1reward:
            messages_input = images + [{"type": "text", "text": prompt_text_r1reward_think}]
        else:
            messages_input = images + [{"type": "text", "text": prompt_text}]
        
        messages = [
            {
                "role": "user",
                "content": messages_input,
            }
        ]

        chat_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        image_inputs, video_inputs = process_vision_info(messages)

        if not args.use_transformers:
            try:
                llm_inputs = {
                    "prompt": chat_input,
                    "multi_modal_data": {'image': image_inputs},
                }

                outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
                output = outputs[0].outputs[0].text
            except Exception as e:
                print(e)
                continue
        else:

            inputs = processor(
                text=[chat_input],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True
            ).to("cuda")

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=4096)
            generated_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
        print(output)

        print(f"GT: {answer}")
        group_total[data['type']] += 1
        
        if args.cot_r1reward:
            output_simple = "0"
            content_match = re.search(r"<answer>(.*?)</answer>", output, re.DOTALL | re.IGNORECASE)
            if content_match:
                output_simple = content_match.group(1).strip()
            else:
                # Fallback: Find the last digit '1' or '2' in the output as a guess
                last_match = re.search(r"[12](?!.*[12])", output)
                output_simple = last_match.group(0) if last_match else None
            try:
                if int(output_simple) == 1:
                    output = 'Answer 1 is better'
                else:
                    output = 'Answer 2 is better'
            except Exception as e:
                output = 'Error!'

        if answer in output:
            correct += 1
            total_corr += 1
            group_correct[data['type']] += 1
            print('correct + 1')

    print("---------------------------------")
    print(args.model_path, '\n')
    accuracy = correct / len(dataset)
    print(f"{args.benchmark_path}, Acc.: {correct} / {len(dataset)} = {accuracy}")
    print(group_correct)
    print(group_total)
    acc_1 = group_correct['shortcot_same_model'] / group_total['shortcot_same_model']
    acc_2 = group_correct['shortcot_diff_model'] / group_total['shortcot_diff_model']
    acc_3 = group_correct['longcot_same_model'] / group_total['longcot_same_model']
    acc_4 = group_correct['longcot_diff_model'] / group_total['longcot_diff_model']
    acc_5 = group_correct['visual_error_bygpt'] / group_total['visual_error_bygpt']
    acc_6 = group_correct['reasoning_error_bygpt'] / group_total['reasoning_error_bygpt']
    acc_7 = group_correct['incidental_error_bygpt'] / group_total['incidental_error_bygpt']
    acc_8 = group_correct['imbalanced_length_short_answer_correct'] / group_total['imbalanced_length_short_answer_correct']
    acc_9 = group_correct['imbalanced_length_shortcot_correct'] / group_total['imbalanced_length_shortcot_correct']
    acc_10 = group_correct['imbalanced_length_longcot_correct'] / group_total['imbalanced_length_longcot_correct']
    
    print(f'shortcot_same_model: {acc_1}')
    print(f'shortcot_diff_model: {acc_2}')
    print(f'longcot_same_model: {acc_3}')
    print(f'longcot_diff_model: {acc_4}')
    paircot_correct = group_correct['shortcot_same_model'] + group_correct['shortcot_diff_model'] + group_correct['longcot_same_model'] + group_correct['longcot_diff_model']
    paircot_total = group_total['shortcot_same_model'] + group_total['shortcot_diff_model'] + group_total['longcot_same_model'] + group_total['longcot_diff_model']
    print('Pairwise CoT Comparison: ', paircot_correct / paircot_total)

    print(f'imbalanced_length_short_answer_correct: {acc_8}')
    print(f'imbalanced_length_shortcot_correct: {acc_9}')
    print(f'imbalanced_length_longcot_correct: {acc_10}')
    imbalanced_correct = group_correct['imbalanced_length_short_answer_correct'] + group_correct['imbalanced_length_shortcot_correct'] + group_correct['imbalanced_length_longcot_correct']
    imbalanced_total = group_total['imbalanced_length_short_answer_correct'] + group_total['imbalanced_length_shortcot_correct'] + group_total['imbalanced_length_longcot_correct']
    print('Length Bias Avoidance: ', imbalanced_correct / imbalanced_total)

    print(f'visual_error_bygpt: {acc_5}')
    print(f'reasoning_error_bygpt: {acc_6}')
    print(f'incidental_error_bygpt: {acc_7}')
    process_correct = group_correct['visual_error_bygpt'] + group_correct['reasoning_error_bygpt'] + group_correct['incidental_error_bygpt']
    process_total = group_total['visual_error_bygpt'] + group_total['reasoning_error_bygpt'] + group_total['incidental_error_bygpt']
    print('Process Error Detection: ', process_correct / process_total)
    print('Total Acc: ', total_corr / total_cnt)
    print("---------------------------------")

if __name__ == "__main__":
 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='../model/M-Judger-RL-Qwen8B/')
    parser.add_argument("--benchmark_path", type=str, default='../data/m_judgebench_data.jsonl')
    parser.add_argument("--image_base_path", type=str, default='../data/')
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument('--short_unified', action='store_true')
    parser.add_argument('--cot_unified', action='store_true')
    parser.add_argument('--cot_r1reward', action='store_true')
    parser.add_argument('--use_transformers', action='store_true')
    
    args = parser.parse_args()

    main(args)