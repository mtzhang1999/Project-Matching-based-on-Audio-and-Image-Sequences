import os
import torch
import task1_pre
import task1_run
import task2_run
import task2_1_run
import task2_rgbrun
import task2_mask
import numpy as np
import task2_match
import math
np.set_printoptions(threshold=10000)

def get_max(a, n):
    max = -1e10
    ret = 0
    for i, x in enumerate(a):
        if x > max:
            max = x
            ret = i
    return ret

def test_task1(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task1/test/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 1, 'audio_0001.pkl': 3, ...}
    class number:
        '061_foam_brick': 0
        'green_basketball': 1
        'salt_cylinder': 2
        'shiny_toy_gun': 3
        'stanley_screwdriver': 4
        'strawberry': 5
        'toothpaste_box': 6
        'toy_elephant': 7
        'whiteboard_spray': 8
        'yellow_block': 9
    '''
    f=os.listdir(root_path)
    imgs=[]
    res2=[]
    pkl_list = []
    for audio_file in f:
        if audio_file[-4:] in ['.pkl']:
            task1_pre.Spectrogram_task(root_path,audio_file)
            img_t=[]
            for n in range(4):
                img_t.append('./task_test/'+audio_file.split('.')[0]+str(n)+'.jpg')
            imgs.append(img_t)
            pkl_list.append(audio_file)
    res2 = task1_run.task1(imgs)
    result = {}
    for i, res in enumerate(res2):
        result[pkl_list[i]] = get_max(res, 10)
    return result

def test_task2(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task2/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': 23, ‘audio_0001’: 11, ...}
    This means audio 'audio_0000.pkl' is matched to video 'video_0023' and ‘audio_0001’ is matched to 'video_0011'.
    '''

    f=os.listdir(root_path)
    imgs=[]
    rgbimgs=[]
    au_gen=[]
    au_loc=[]
    vi_loc=[]
    vi_gen=[]
    audio_file=[]
    vidio_file=[]
    for all_file in f:
        if all_file[-4:] in ['.pkl']:
            task1_pre.Spectrogram_task(root_path,all_file)
            img_t=[]
            audio_file.append(all_file)
            for n in range(4):
                img_t.append('./task_test/'+all_file.split('.')[0]+str(n)+'.jpg')
            imgs.append(img_t)
        else:
            rgbimgs.append(os.path.join(root_path,all_file))
            vi_loc.append(task2_mask.mask_test(os.path.join(root_path,all_file)))
            vidio_file.append(all_file)

    vi_gen=(task2_rgbrun.rgb_test(rgbimgs))
    au_loc=(task2_run.task2(imgs))
    au_gen=(task2_1_run.task2_1(imgs))

    aunum=len(audio_file)
    vinum=len(vidio_file)
    matchmatrix=[]
    results={}
    for i in range(aunum):
        for j in range(vinum):
            if(au_gen[i]==vi_gen[j][0]):
                matchmatrix.append((i,j,(800-math.sqrt((au_loc[i][0]-vi_loc[j][0])*(au_loc[i][0]-vi_loc[j][0])+(au_loc[i][1]-vi_loc[j][1])*(au_loc[i][1]-vi_loc[j][1])))))
            else:
                matchmatrix.append((i,j,0.6*(800-math.sqrt((au_loc[i][0]-vi_loc[j][0])*(au_loc[i][0]-vi_loc[j][0])+(au_loc[i][1]-vi_loc[j][1])*(au_loc[i][1]-vi_loc[j][1])))))
    matchres=task2_match.KM_running(matchmatrix)
    print(matchres)
    print(len(matchres))
    for i in range(len(matchres)):
        results[audio_file[matchres[i][0]]]=matchres[i][1]
    return results

def test_task3(root_path):
    '''
    :param root_path: root path of test data, e.g. ./dataset/task3/test/0/
    :return results: a dict of classification results
    results = {'audio_0000.pkl': -1, ‘audio_0001’: 12, ...}
    This means audio 'audio_0000.pkl' is not matched to any video and ‘audio_0001’ is matched to 'video_0012'.
    '''
    results = {}
    f=os.listdir(root_path)
    imgs=[]
    rgbimgs=[]
    au_gen=[]
    au_loc=[]
    vi_loc=[]
    vi_gen=[]
    audio_file=[]
    vidio_file=[]
    for all_file in f:
        if all_file[-4:] in ['.pkl']:
            task1_pre.Spectrogram_task(root_path,all_file)
            img_t=[]
            audio_file.append(all_file)
            for n in range(4):
                img_t.append('./task_test/'+all_file.split('.')[0]+str(n)+'.jpg')
            imgs.append(img_t)
        else:
            rgbimgs.append(os.path.join(root_path,all_file))
            vi_loc.append(task2_mask.mask_test(os.path.join(root_path,all_file)))
            vidio_file.append(all_file)
 
    vi_gen=(task2_rgbrun.rgb_test(rgbimgs))
    au_loc=(task2_run.task2(imgs))
    au_gen=(task2_1_run.task2_1(imgs))

    # aau_gen=np.array(au_gen)
    # aau_loc=np.array(au_loc)
    # avi_loc=np.array(vi_loc)
    # avi_gen=np.array(vi_gen)
    # np.save('au_gen3.npy',aau_gen) 
    # np.save('au_loc3.npy',aau_loc)
    # np.save('vi_loc3.npy',avi_loc)
    # np.save('vi_gen3.npy',avi_gen)
    # # load
    # # au_gen=np.load('au_gen3.npy',allow_pickle=True)
    # # au_gen=au_gen.tolist()
    # # au_loc=np.load('au_loc3.npy',allow_pickle=True)
    # # au_loc=au_loc.tolist()
    # # vi_loc=np.load('vi_loc3.npy',allow_pickle=True)
    # # vi_loc=vi_loc.tolist()
    # # vi_gen=np.load('vi_gen3.npy',allow_pickle=True)
    # # vi_gen=vi_gen.tolist()
    aunum=len(audio_file)
    vinum=len(vidio_file)
    matchmatrix=[]
    results={}
    for i in range(aunum):
        for j in range(vinum):
            if(au_gen[i]==vi_gen[j][0]):
                matchmatrix.append((i,j,(800-math.sqrt((au_loc[i][0]-vi_loc[j][0])*(au_loc[i][0]-vi_loc[j][0])+(au_loc[i][1]-vi_loc[j][1])*(au_loc[i][1]-vi_loc[j][1])))))
            else:
                matchmatrix.append((i,j,0.6*(800-math.sqrt((au_loc[i][0]-vi_loc[j][0])*(au_loc[i][0]-vi_loc[j][0])+(au_loc[i][1]-vi_loc[j][1])*(au_loc[i][1]-vi_loc[j][1])))))
    matchres=task2_match.KM_running(matchmatrix)
    print(matchres)
    print(len(matchres))
    for i in range(len(matchres)):
        if matchres[i][2]>350:
            results[audio_file[matchres[i][0]]]=matchres[i][1]
        else:
            results[audio_file[matchres[i][0]]]=-1
    return results

if __name__ == "__main__":
    # result=test_task1('./task1/test/')
    # result=test_task2('./task2/0/')
    result=test_task3('./task3/0/')
    print(result)