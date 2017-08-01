# -*- coding: utf-8 -*-
"""
Created on Sat Jul 15 18:26:58 2017
@author: Yogi007
QQ:362500913
"""
import multiprocessing as mp
import numba as nb
import time as t
import cv2 as cv
import os
global white
white=255 #指定白色数值为255

def removeImageNoise(imgfile,min_box=4,max_box=10):
    """
    removeImageNoise功能:对输入图片进行二值化处理并消除其中较小的孤立噪点.
    imgfile:需进行处理的图片文件路径;
    min_ratio与max_ratio:指定图像灰度化后进行二值化时白色区域在总图像面积中的占比;
    min_box与max_box:设定对二值化图像进行扫描的矩阵大小范围，
    噪点大小要在这个范围内才能被消除.
    """
    img=cv.imread(imgfile)
    imgray_=cv.cvtColor(img,cv.COLOR_BGR2GRAY) #图像灰度化
    imgray=cv.equalizeHist(imgray_) #直方图均衡化,解决少量对比不明显图片的问题
    
    return clearNoise(imgray.copy(),min_box,max_box)
@nb.jit
def clearNoise(lena,min_box,max_box):
    global white
    nHeight = lena.shape[0];nWidth = lena.shape[1]
    minbox=min_box; maxbox=max_box
    """在图像中遍历minbox到maxbox的矩形滑块,如果边缘为连续白色则整块填充为白色"""
    for i in range(nWidth):
        for j in range(nHeight):
            for box in range(minbox,maxbox+1):
                if i+box>nWidth or j+box>nHeight:
                    break
                else:
                    noise=True
                    if j>2 and j<nHeight-3:
                        if lena[j][i:i+box].sum()<box*white:
                            noise=False
                            continue
                    if j+box-1>2 and j+box-1<nHeight-3:
                        if lena[j+box][i:i+box].sum()<box*white:
                            noise=False
                            continue
                    if i>2 and i<nWidth-3:
                        if lena[j:j+box,i].sum()<box*white:
                            noise=False
                            continue
                    if i+box-1>2 and i+box-1<nWidth-3:
                        if lena[j:j+box,i+box].sum()<box*white:
                            noise=False
                            continue
                    if noise: lena[j:j+box,i:i+box]=white

    return lena

def removeNoiseTofileOnMP(cnt,afiles,minbox=4,maxbox=10):
    """
    removeNoiseTofileOnMP:直接将降噪完成的图片存储到指定位置;
    cnt:多进程同步Value,实现整体进度统计功能.
    """
    while(afiles.qsize()>0):
        cnt.acquire();afile=afiles.get();cnt.release()
        simgfile=afile[0];timgfile=afile[1]
        status=cv.imwrite(timgfile,removeImageNoise(simgfile,min_box=minbox,
                                                    max_box=maxbox))
        if not status:
            cnt.acquire()
            print('removeNoiseTofileOnMP failed:',simgfile)
            cnt.release();return status
        else:
            cnt.acquire();cnt.value+=1;cnt.release()
    return status

def getFiles(sfolder,tfolder):
    """
    getFiles:根据源图片目录和目标存储目录生成图片源路径与目标路径列表.
    """
    os.mkdir(tfolder)
    sfiles=[];tfiles=[]
    for file in os.listdir(sfolder):
        if '.png' in file:
            sfiles.append(sfolder+file)
            tfiles.append(tfolder+file)
    return sfiles,tfiles

def mpRemoveNoise(sfolder='./CaptchaImages_ext/',tfolder='./CaptchaImages_ext_ok/',
                  minbox=4,maxbox=10,n_process=-1):
    """
    mpRemoveNoise:多进程并行完成图像降噪预处理，每隔3秒显示进度和剩余时间;
    sfolder:源图片的文件夹;tfolder:降噪图片存储文件夹;
    n_process:并行的进程数,建议等于或略大于CPU核心数,-1为CPU核心数.
    """
    """cnt为多进程同步Value用于统计进度,afiles为源文件与目标文件列表用于进程间共享,
    mps为并行进程列表"""
    cnt=mp.Value('i',0);afiles=mp.Queue();mps=[]
    sfiles,tfiles=getFiles(sfolder,tfolder)
    filecnt=len(sfiles)
    beginTime=t.time()
    tagTime=beginTime
    if n_process==-1: n_cpu=mp.cpu_count()
    for i in range(len(sfiles)):
        afiles.put([sfiles[i],tfiles[i]])
    
    for i in range(n_cpu):
        ps=mp.Process(target=removeNoiseTofileOnMP,
                     args=(cnt,afiles,minbox,maxbox))
        mps.append(ps);ps.start()
    
    while(len(mps)>0):
        for i in range(len(mps)):
            if not mps[i].is_alive():
                del mps[i];break
        t.sleep(1)
        
        nowTime=t.time()
        if nowTime-tagTime>3:
            tagTime=nowTime;elapsed=round(nowTime-beginTime)
            speed=round(cnt.value/(nowTime-beginTime),2)
            #print('speed:',speed,'/s', 'progress:', cnt.value,'/',filecnt,end=', ')
            #print('elapsed:',elapsed,end=', ')
            if speed==0: remain='unknow'
            else: remain=round((filecnt-cnt.value)/speed)
            print('remain:',remain,'s')
    return 0

if __name__=='__main__':
    mpRemoveNoise(sfolder='./CaptchaImages_ext/',n_process=-1) #n_process等于CPU数量
#    cv.imshow('noNoise',removeImageNoise('1089.png'))
#    cv.waitKey(0);cv.destroyAllWindows()
