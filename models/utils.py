import json
import h5py
import sys


def parse_dataset_paths(ct_data_path, dataset_names) -> list:
    # returns tuple with (poly_path, mono_path, name)
    return_list = []
    f = open(ct_data_path, "r")
    if (sys.version_info < (3,9)):
        data = json.load(f, encoding="utf-8")
    else:
        data = json.load(f)

    isAllData = False
    if isinstance(dataset_names, list):
        isAllData = any(["all" == elem for elem in dataset_names])
    elif isinstance(dataset_names, str):
        isAllData = ("all" == dataset_names)

    for entry in data["datasets"]:
        if entry['name'] in dataset_names or isAllData:
            return_list.append(
                (str(entry["ct"]), str(entry["gt"]), str(entry["name"]), float(entry["material_mode"]))
            )
    f.close()
    return return_list


def add_datasets_to_noisy_images_json(dataset_paths, noisy_data_path):
    """
        adds dataset entrys to noisy_data_path json if they do not yet exist
    """

    dataset_paths = parse_dataset_paths(dataset_paths)

    noise_index_file = open(noisy_data_path, "r")
    noise_index_data = json.load(noise_index_file, encoding="utf-8")
    noise_index_file.close()

    for entry in dataset_paths:
        entry_exists = False
        for entry_noise in noise_index_data["datasets"]:
            if entry[2] == entry_noise["name"]:
                entry_exists = True

        if not entry_exists:
            noise_index_data["datasets"].append(
                {
                    "name": entry[2],
                    "noisy_samples_known": False,
                    "nr_samples": 0,
                    "nr_noisy_samples": 0,
                    "noisy_indexes": [],
                }
            )

    with open(noisy_data_path, "w") as noise_index_file:
        json.dump(noise_index_data, noise_index_file, indent=4)


def get_mean_grey_value(volume_path):
    with h5py.File(volume_path, 'r') as h5f:
        mean_grey_value = h5f["Volume"].attrs['Mean_grey_value']

    return mean_grey_value
