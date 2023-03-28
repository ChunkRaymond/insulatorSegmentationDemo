import os 
from pathlib import Path
import shutil
import cv2 as cv 
from tqdm import tqdm 
def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def removeFile(targetDir, baseDir):
    targetDir = Path(targetDir)
    baseDir = Path(baseDir)
    # for file in baseDir.glob('*.png'):
    #     obj_img = cv.imread(str(file),0)
    #     w,h = obj_img.shape

    #     for i in range(w):
    #         for j in range(h):
    #             if obj_img[i][j] != 0:
    #                 obj_img[i,j] = 0
    #             else:
    #                 obj_img[i,j] = 255
    #     pass 
    #     cv.imwrite(str(file),obj_img)

    #     targetFile = targetDir/str(file.stem+'.JPG')
    #     if not targetFile.exists():
    #         targetFile.unlink(True)
    #         print(str(targetFile)+' is not exists!')
    step = 0
    for file in tqdm(targetDir.glob('*.jpg')):
        baseFile = baseDir/(str(file.stem)+'.png')
        print(step+1)
        print(str(file),str(baseFile))
        step+=1
        if not baseFile.exists():
            file.unlink()
            print('not exists!')
            
        pass
    pass 
# removeFile('data/trainData/images','data/trainData/mask_img')