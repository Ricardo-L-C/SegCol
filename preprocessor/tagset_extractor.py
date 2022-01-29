import argparse, traceback, json, shutil
from pathlib import Path

SIMPLE_BACKGROUND = 412368
WHITE_BACKGROUND = 515193
sketch_tags = [513837, 1931] # grayscale, sketch
include_tags = [470575, 540830] # 1girl, 1boy
hair_tags = [87788, 16867, 13200, 10953, 16442, 11429, 15425, 8388, 5403, 16581, 87676, 16580, 94007, 403081, 468534, 403081, 524070]
eye_tags = [10959, 8526, 16578, 10960, 15654, 89189, 16750, 13199, 89368, 95405, 89228, 390186, 517427, 376034]
blacklist_tags = [63, 4751, 12650, 172609, 555246, 513475, 440465, 511246, 678477] # comic, photo, subtitled, english, black border, multiple_views, speech_bubble, black_speech_bubble, shared_speech_bubble

IS_SKETCH = 5

def load_metafile_list(path):
    file_list = [p for p in (path / "2021-old").iterdir() if p.is_file()]
    file_list += [p for p in (path / "2020").iterdir() if p.is_file()]
    file_list += [p for p in (path / "2019").iterdir() if p.is_file()]
    file_list += [p for p in (path / "2018").iterdir() if p.is_file()]
    file_list += [p for p in (path / "2017").iterdir() if p.is_file()]

    return file_list

def make_tag_dict(file_list):
    tag_dict = {}

    for p in file_list:
        print(f"get jsons in {p.absolute()}")
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    meta = json.loads(line)

                    if "tags" not in meta:
                        continue

                for tag in meta["tags"]:
                    if tag["category"] != "0":
                        continue
                    tag_id = int(tag["id"])

                    if not tag_id in tag_dict:
                        tag_dict[tag_id] = tag["name"]

        except Exception:
            print(f"json reader failed: {p.absolute()}")
            traceback.print_exc()

    return tag_dict


def json_filter(meta):
        w, h = int(meta["image_width"]), int(meta["image_height"])
        if w < 256 and h < 256:
            return False
        if not (3 / 4 < w / h < 4 / 3):
            return False

        if int(meta["score"]) < 4:
            return False

        tags = set(int(tag["id"]) for tag in meta["tags"] if tag["category"] == "0")

        for i in blacklist_tags:
            if i in tags:
                return False

        if SIMPLE_BACKGROUND not in tags:
            return False

        if len(tags.intersection(sketch_tags)) >= 1 and WHITE_BACKGROUND in tags:
            if all(len(tags.intersection(l)) == 0 for l in [hair_tags, eye_tags]):
                return IS_SKETCH
            else:
                return False

        if not all(len(tags.intersection(l)) == 1 for l in [include_tags, hair_tags, eye_tags]):
            return False

        return True


def main_tag_extract(dataset, meta_list):
    resolutions = {}

    tag_lines = []
    selected = set()
    img_root_path = dataset / "512px"

    save_path = dataset / "train_image_base"
    save_path.mkdir(exist_ok=True)

    sketch_path = dataset / "liner_test_base"
    sketch_path.mkdir(exist_ok=True)

    print("copying images...")
    for i in meta_list:
        print(f"get jsons in {i.absolute()}")
        try:
            with i.open("r", encoding="utf-8") as f:
                for line in f:
                    meta = json.loads(line)

                    if int(meta["id"]) in selected:
                        continue

                    filtered = json_filter(meta)

                    if filtered == False:
                        continue

                    file_id = int(meta["id"])
                    file_path = img_root_path / f"{file_id%1000:04d}" / f"{file_id}.jpg"

                    if not file_path.exists():
                        continue

                    if filtered == IS_SKETCH:
                        shutil.copy2(file_path, sketch_path)
                    else:
                        shutil.copy2(file_path, save_path)

                    tag_line = " ".join(tag["id"] for tag in meta["tags"] if tag["category"] == "0")
                    tag_lines.append(f"{file_id} {tag_line}\n")

                    resolutions[int(meta["id"])] = (int(meta["image_width"]), int(meta["image_height"]))

                    selected.add(int(meta["id"]))

        except Exception:
            print(f"json reader failed: {i.absolute()}")
            traceback.print_exc()

    with (dataset / "tags.txt").open("w") as f:
        for tag_line in tag_lines:
            f.write(tag_line)

    with (dataset / "resolutions.json").open("w") as f:
        json.dump(resolutions, f)

if __name__=="__main__":
    desc = "Seg colorization tagset extractor"
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--dataset", type=Path, default="./dataset", help="path to dataset directory")
    parser.add_argument("--make_tag_dict", action="store_true", help="make (tag_id - tag_name) text to dataset/tags_dict.json")

    args = parser.parse_args()

    metafile_list = load_metafile_list(args.dataset / "metadata")

    if args.make_tag_dict:
        id2name = make_tag_dict(metafile_list)
        name2id = {v: k for k, v in id2name.items()}
        tag_dict = {"id2name": id2name, "name2id": name2id}
        with (args.dataset / "tags_dict.json").open("w") as f:
            json.dump(tag_dict, f)
    else:
        main_tag_extract(args.dataset, metafile_list)
