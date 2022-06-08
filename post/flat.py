from tricks import *
from skimage.morphology import skeletonize, dilation


def get_regions(img, skeleton_map):
    marker = skeleton_map[:, :, 0]
    normal = topo_compute_normal(marker) * 127.5 + 127.5
    marker[marker > 200] = 255
    marker[marker < 255] = 0
    labels, nil = label(marker / 255)
    water = cv2.watershed(normal.clip(0, 255).astype(np.uint8), labels.astype(np.int32)) + 1
    water = thinning(water)
    all_region_indices = find_all(water)
    regions = np.zeros_like(skeleton_map, dtype=np.uint8)
    for region_indices in all_region_indices:
        regions[region_indices] = np.random.randint(low=0, high=255, size=(3,)).clip(0, 255).astype(np.uint8)
    result = np.zeros_like(skeleton_map, dtype=np.uint8)
    for region_indices in all_region_indices:
        result[region_indices] = np.median(img[region_indices], axis=0)
    return regions, result


if __name__=='__main__':
    import sys

    from pathlib import Path

    p = Path(sys.argv[1])

    for i in p.iterdir():
      if i.is_file():
        skeleton_map = cv2.imread(str(p / f"../{p.name}_skeleton" / i.name))
        img = cv2.resize(cv2.imread(str(i)), skeleton_map.shape[:2])

        region, flat = get_regions(img, skeleton_map)
        cv2.imwrite(str(p / f"res/{i.stem}_region.png"), region)
        cv2.imwrite(str(p / f"res/{i.stem}_flat.png"), flat)
