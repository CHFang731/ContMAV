import os

# 配置您的数据集根目录
ROOT_DIR = "/mnt/8tb_hdd2/fang_dataset/ContMAV_dataset"

SPLITS = ['train', 'val', 'test']

def get_identifiers(folder, suffix):
    """获取指定文件夹下所有文件的ID（去除后缀），并返回集合"""
    ids = set()
    # 遍历所有子目录 (例如 train/aachen, train/bochum 等)
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith(suffix):
                # 去掉后缀得到纯 ID
                ident = f.replace(suffix, "")
                ids.add(ident)
    return ids

print(f"正在检查数据集: {ROOT_DIR} ...\n")

total_errors = 0

for split in SPLITS:
    print(f"--- 正在检查 {split} 集 ---")
    
    img_path = os.path.join(ROOT_DIR, "leftImg8bit", split)
    lbl_path = os.path.join(ROOT_DIR, "gtFine", split)
    
    # 检查目录是否存在
    if not os.path.exists(img_path) or not os.path.exists(lbl_path):
        print(f" 错误: 找不到目录 {img_path} 或 {lbl_path}")
        continue

    # 提取 ID
    # Cityscapes 标准后缀
    img_ids = get_identifiers(img_path, "_leftImg8bit.png")
    lbl_ids = get_identifiers(lbl_path, "_gtFine_labelIds.png")
    
    print(f"  图片数量: {len(img_ids)}")
    print(f"  标签数量: {len(lbl_ids)}")
    
    # 比较差异
    only_in_img = img_ids - lbl_ids
    only_in_lbl = lbl_ids - img_ids
    
    if not only_in_img and not only_in_lbl:
        print(f"   {split} 集完美匹配！")
    else:
        if only_in_img:
            print(f"   发现 {len(only_in_img)} 个 ID 只有图片，没有标签:")
            for i in list(only_in_img)[:3]: print(f"     - {i}")
            total_errors += 1
        if only_in_lbl:
            print(f"   发现 {len(only_in_lbl)} 个 ID 只有标签，没有图片:")
            for i in list(only_in_lbl)[:3]: print(f"     - {i}")
            total_errors += 1

print("\n" + "="*30)
if total_errors == 0:
    print(" 恭喜！所有数据集 (train, val, test) 均完美匹配。")
    print("如果脚本 prepare_dataset.py 仍然报错，请检查是否有隐藏文件 (.DS_Store 等)。")
else:
    print(f" 发现 {total_errors} 处不匹配，请根据上方提示修复。")
