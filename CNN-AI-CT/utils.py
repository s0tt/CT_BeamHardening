import json

def parse_dataset_paths(ct_path) -> list:
    # interpret input as JSON file if no gt specified
    return_list = []
    f = open(ct_path, "r")
    data = json.load(f, encoding="utf-8")
    for entry in data["datasets"]:
        return_list.append((str(entry["ct"]), str(entry["gt"]), str(entry["name"])))
    f.close()
    return return_list

def add_datasets_to_noisy_images_json(dataset_path, noisy_data_path): 
    """
        adds dataset entrys to noisy_data_path json if they do not yet exist
    """
    dataset_file = open(dataset_path, "r")
    dataset_data = json.load(dataset_file, encoding="utf-8")

    with open(noisy_data_path, "r+") as noise_index_file: 
        noise_index_data = json.load(noise_index_file, encoding="utf-8")
        for entry in dataset_data["datasets"]:
                if entry["name"] not in noise_index_data["datasets"].keys(): 
                        noise_index_data["datasets"].append(
                           {
                            "name": entry['name'],
                            "noisy_samples_known": True,
                            "nr_samples": 0,
                            "nr_noisy_samples": 0, 
                            "noisy_indexes": [],
                            }
                        )
        json.dump(noise_index_data, noise_index_file)

    dataset_file.close()

def parse_json_after_noisy_flags(json_path): 
    return_list = []
    f = open(json_path, "r")
    data = json.load(f, encoding="utf-8")
    for entry in data["datasets"]:
            return_list.append((str(entry["noisy_samples_known"])))
    f.close()
    
    return return_list

 