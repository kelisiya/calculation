#coding=UTF-8
import numpy as np
import pandas as pd

'''
# 演示如何从一组中选择出现频率最多的元素
a=[12,23,34,45,67,12,12,34,12] 

b = {}
for i in a:
  if i not in b:
    b[i] = 1
  else:
    b[i] += 1
print (max(b.items(), key = lambda x: x[1])[0])

exit()
'''


f_labels = open('myresult.txt','a')

results=[]

resultfiles=['result1.txt','result2.txt','result3.txt','result4.txt',
            'result5.txt','result6.txt','result7.txt','result8.txt']

for resultfile in resultfiles:
    fp=open(resultfile,'r')
    results.append(fp.readlines())

results=np.array(results)

myresult=[]


for i in range(len(results[0])):
    b = {}
    a = [results[0,i],results[1,i],results[2,i],results[3,i],results[4,i],results[5,i],results[6,i],results[7,i]]
    for x in a:
      if x not in b:
        b[x] = 1
      else:
        b[x] += 1
    toprint = max(b.items(), key = lambda x: x[1])[0]
    myresult.append(toprint)

np.savetxt('myresult.txt',myresult,fmt='%s')

#大师 我还不大会用numpy 所以为了去掉myresult.txt中的空行 用了下面的笨方法 请大师改进

fr2 = open("myresult2.txt","a") 

fr1 = open("myresult.txt","r") 
lines1 = fr1.readlines()
num = 400000
for r in range(num):
  print(r)
  print (lines1[r])
  if len(lines1[r]) > 4:
    fr2.write(lines1[r])

#最终生成的 myresult2.txt 可作为结果提交
