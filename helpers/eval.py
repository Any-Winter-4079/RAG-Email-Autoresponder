import json

#######################################
# Helper 1: Write eval output to file #
#######################################
def write_eval_output_to_file(data_variant_results_dir, output_name, eval_output, data_variant):
    json_path = data_variant_results_dir / f"{output_name}.json"
    jsonl_path = data_variant_results_dir / f"{output_name}.jsonl"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(eval_output, json_file, ensure_ascii=False, indent=2)
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(eval_output, ensure_ascii=False) + "\n")
    print(f"\twrote {data_variant}/{json_path.name}")
    print(f"\twrote {data_variant}/{jsonl_path.name}\n")
