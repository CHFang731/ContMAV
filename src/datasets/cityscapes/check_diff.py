import os

# 设置路径
img_path = "/mnt/8tb_hdd2/fang_dataset/ContMAV_dataset/leftImg8bit/test"
lbl_path = "/mnt/8tb_hdd2/fang_dataset/ContMAV_dataset/gtFine/test"

# 获取所有文件的 ID (去掉后缀)
def get_ids(folder, suffix):
    ids = set()
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(suffix):
                # 提取文件名中的 ID 部分
                ident = f.replace(suffix, "")
                ids.add(ident)
    return ids

# 提取 ID
print("正在扫描文件...")
img_ids = get_ids(img_path, "_leftImg8bit.png")
lbl_ids = get_ids(lbl_path, "_gtFine_labelIds.png")

print(f"图片 ID 数量: {len(img_ids)}")
print(f"标签 ID 数量: {len(lbl_ids)}")

# 找出差异
only_in_img = img_ids - lbl_ids
only_in_lbl = lbl_ids - img_ids

if only_in_img:
    print(f"\n 有 {len(only_in_img)} 个 ID 只在图片中，缺少标签:")
    print(list(only_in_img)[:5]) # 只显示前5个
if only_in_lbl:
    print(f"\n 有 {len(only_in_lbl)} 个 ID 只在标签中，缺少图片:")
    print(list(only_in_lbl)[:5]) # 只显示前5个

if not only_in_img and not only_in_lbl:
    print("\n 恭喜！所有 ID 均完美匹配。问题可能出在 train 或 val 集，或者脚本的排序问题。")
