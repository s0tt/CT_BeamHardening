import json

def parse_dataset_paths(ct_path, gt_path):
    # interpret input as JSON file if no gt specified
    if gt_path is None:
        return_list = []
        f = open(ct_path, "r")
        data = json.load(f, encoding="utf-8")
        for entry in data["datasets"]:
            return_list.append((str(entry["ct"]), str(entry["gt"])))
        f.close()
        return return_list
    else:
        return [(ct_path, gt_path)]
