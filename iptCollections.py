import pandas as pd
import zipfile
import os

# 定义根目录
root_dir = r'D:\fn'

# 读取 Excel 文件
excel_path = 'collections_list.xlsx'  # 替换为实际路径
df = pd.read_excel(excel_path)

# 处理每个 IPT 文件
for index, row in df.iterrows():
    ipt_file = os.path.join(root_dir, row['ipt_filename'])  # 拼接文件路径
    if os.path.exists(ipt_file):
        try:
            # 解压缩文件
            with zipfile.ZipFile(ipt_file, 'r') as zip_ref:
                # 查找 occurrence.csv 或 occurrence.txt 文件
                occurrence_files = [f for f in zip_ref.namelist() if 'occurrence.csv' in f or 'occurrence.txt' in f]
                if occurrence_files:
                    with zip_ref.open(occurrence_files[0]) as occ_file:
                        # 判断文件格式并读取 occurrence 文件的前 10 行
                        if occurrence_files[0].endswith('.csv'):
                            occ_df = pd.read_csv(occ_file, nrows=10)
                        else:  # 假设 .txt 文件用制表符分隔
                            occ_df = pd.read_csv(occ_file, delimiter='\t', nrows=10)

                        # 统计 occurrence header 的列数
                        header_column_count = len(occ_df.columns)
                        df.at[index, 'header_column_count'] = header_column_count

                        # 提取 institutioncode 和 collectioncode 的第一个非空值
                        institution_code = occ_df['institutionCode'].dropna().astype(str).iloc[
                            0] if 'institutionCode' in occ_df.columns and not occ_df[
                            'institutionCode'].dropna().empty else 'none'
                        collection_code = occ_df['collectionCode'].dropna().astype(str).iloc[
                            0] if 'collectionCode' in occ_df.columns and not occ_df[
                            'collectionCode'].dropna().empty else 'none'

                        # 合并并写入新列
                        df.at[index, 'institution_collection'] = f"{institution_code}:{collection_code}"
                else:
                    df.at[index, 'institution_collection'] = 'none'
                    df.at[index, 'header_column_count'] = 0
        except Exception as e:
            print(f"Error processing {ipt_file}: {e}")
            df.at[index, 'institution_collection'] = 'none'
            df.at[index, 'header_column_count'] = 0
    else:
        print(f"File not found: {ipt_file}")
        df.at[index, 'institution_collection'] = 'none'
        df.at[index, 'header_column_count'] = 0

# 保存更新后的 Excel 文件
output_path = os.path.join(root_dir, 'updated_collections_list_number.xlsx')
df.to_excel(output_path, index=False)
print(f"更新完成，已保存至 {output_path}")