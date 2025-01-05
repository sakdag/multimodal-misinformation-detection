######### evidence retrieval
#image retrieval
python3 eval_ir_llms.py --mode=test --media=img --model_name==Salesforce/instructblip-flan-t5-xl --use_llm_score=False --mocheg_result_path=checkpoints/mocheg_img_results/query_result_img.pkl --prompt=Is this image and text query mentioning the same person or topic? answer with yes or no.
#Text retrieval
python3 eval_ir_llms.py --mode=test --media=txt --model_name==Mistral-7B-OpenOrca --use_llm_score=True --mocheg_result_path=checkpoints/mocheg_txt_results/query_result_txt.pkl --prompt=Is this corpus related to the query? Answer with yes or no.
######## Fact verification
# with text evidence
python3 FNdetection_llm.py --media=txt --model_name=Open-Orca/Mistral-7B-OpenOrca --evidence_type=gold --two_level_prompting=True --task1_out=retrieval/output/ir_llms/mocheg/txt --level1_prompt=Is this text evidence sufficient to confirm or reject this claim?answer with yes if it is suffiecient and aswer with no if it is not enough information ----level2_prompt=Does this text evidence support or reject this claim?answer with yes if it supports, answer with no if it rejects.
# with multimodal evidence
python3 FNdetection_llm.py --media=multimodal --model_name=llava-hf/llava-v1.6-mistral-7b-hf --evidence_type=gold --two_level_prompting=True --task1_out=retrieval/output/ir_llms/mocheg/txt --task1_out==retrieval/output/ir_llms/mocheg/img --level1_prompt=Is this image and text evidence sufficient to confirm or reject this claim?answer with yes if they are suffiecient and aswer with no if they are not enough information ----level2_prompt=Does this text and image evidence support or reject this claim?answer with yes if they supports, answer with no if they reject.

