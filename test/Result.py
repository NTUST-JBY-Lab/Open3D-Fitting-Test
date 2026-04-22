import os

def cleanup_result(directory: str, file_extension: list[str] = ['_alphashape.obj', '_pcd.ply', '_RANSAC.obj', '_poisson.obj', '_silhouette.obj']):
    """
    對於 directory 下的所有檔案，如果檔名以 file_extension 中的其中一個值結尾，則刪掉
    """
    for file in os.listdir(directory):
        # 如果 file 以任何一個 file_extension 結尾
        if any(file.endswith(ext) for ext in file_extension):
            # delete
            os.remove(os.path.join(directory, file))
