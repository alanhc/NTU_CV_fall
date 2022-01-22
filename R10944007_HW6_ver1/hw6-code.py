#!/usr/bin/env python
# coding: utf-8

# # Computer Vision I (922 U0610) - Homework 6
# Author: alanhc
# 
# ID: r10944007
# 
# Date: 10/27

# ## README
# 0. create env: `conda env create -f environment.yml`
# 1. enter env: `conda activate ntu-cv`
# 2. run jupyter `jupyter notebook`

# Write a program which counts the Yokoi connectivity number on a downsampled image(lena.bmp).
# - Downsampling Lena from 512x512 to 64x64:
#   - Binarize the benchmark image lena as in HW2, then using 8x8 blocks as a unit, take the topmost-left pixel as the downsampled data.
# - Count the Yokoi connectivity number on a downsampled lena using 4-connected.
# - Result of this assignment is a 64x64 matrix.
# - You can use any programing language to implement homework, however, you'll get zero point if you just call existing library.

# In[2]:


from PIL import Image
import numpy as np

# Todo: 讀檔，確定影像大小
img = Image.open("input/lena.bmp")
img = np.array(img)

h, w = img.shape

print("image shape:", img.shape)
show = Image.fromarray(img).resize((256,256))
show


# In[3]:


# Todo: sampling
## Hint: topmost-left: 左上角點可用//8求
# Algorithm:
## 1. 根據hint，先算出一個row/col有幾個sampling點，再用*8來還原sampling點在原本的位置
## 2. >128 來產生threshold 128的影像

## 512->64.  512/64=8
# 1.
ans = np.zeros((h//8,w//8))
for y in range(h//8):
    for x in range(w//8):
        # 1.
        if (img[y*8][x*8]>=128):# 2. 
            ans[y][x] = 255


# In[4]:


import matplotlib.pyplot as plt
print(ans.shape)
plt.imshow(ans, cmap="gray")
img_downSampling = ans


# ![](img/textbook-formula.PNG)

# In[5]:


"""
index pixels
7 2 6 
3 0 1
8 4 5 
"""
# Todo: Yokoi connectivity numbe
# Algorithm:
## 1. 計算 h(a,b,c,d)，*程式為f_h* ，藉由三個coner點判斷類型 q,r,s，其中b,c,d,e為帶入的kernel點Xi
## 2. 計算 connectivity operator f(a1,a2,a3,a4)，其中a1,a2,a3,a4是由 h(x0,x1,x6,x2), h(x0,x2,x7,x3),  
##                                                                 h(x0,x3,x8,x4), h(x0,x4,x5,x1)
## 3. 根據課本式子參考上圖，計算5或n

s_h, s_w = img_downSampling.shape

def index_values(img, y, x, n):
    # shift is a convert table reelated to index  textbook:x0,x1,x2...
    shift = {
        7:[-1,-1],
        2:[-1,0],
        6:[-1,1],
        3:[0,-1],
        0:[0,0],
        1:[0,1],
        8:[1,-1],
        4:[1,0],
        5:[1,1]
    }
    now_y = y+shift[n][0]
    now_x = x+shift[n][1]
    
    if (now_y>=0 and now_x>=0 and now_y<s_h and now_x<s_w):
        return img[ now_y ][ now_x ]
    else: 
        return 0
    
# 1.
def f_h(img, pos,b,c,d,e):
    y, x = pos
    
    b = index_values(img, y, x, b) 
    c = index_values(img, y, x, c)
    d = index_values(img, y, x, d)
    e = index_values(img, y, x, e)
    
    if (b==c and (d!=b or e!=b)):
        return "q"
    if (b==c and (d==b and e==b)):
        return "r"
    if (b!=c):
        return "s"
    else:
        print("=", b,c,d,e)

def f(a1,a2,a3,a4):
    if (a1==a2 and a2==a3 and a3==a4 and a4=="r"):
        return 5
    s = str(a1+a2+a3+a4)
    ## 找{a1,a2,a3,a4}有幾個q
    ct=0
    for i in range(len(s)):
        if s[i]=="q":
            ct+=1
    return ct


# In[6]:


src = img_downSampling
ans = np.zeros((s_h, s_w))
for y in range(s_h):
    for x in range(s_w):
        ## 2. 
        a1 = f_h(src, (y,x),0,1,6,2)
        a2 = f_h(src, (y,x),0,2,7,3)
        a3 = f_h(src, (y,x),0,3,8,4)
        a4 = f_h(src, (y,x),0,4,5,1)
        if (src[y][x]==0):
            continue
        ## 3.
        ans[y][x] = f(a1,a2,a3,a4)
ans.shape


# In[7]:


for y in range(ans.shape[0]):
    for x in range(ans.shape[1]):
        now = int(ans[y][x])
        if (now==0):
            print(" ", end="")
        else:
            print("%1d"%now, end="")
    print()


# In[10]:


import pandas as pd
pd.DataFrame(ans).to_csv("output/result.csv")


# In[12]:


import matplotlib.pyplot as plt
img = Image.open("output/Yokoi_connectivity_number.png")
plt.figure(figsize=(16,16))
plt.imshow(img)


# Ref
# - https://en.wikipedia.org/wiki/Neighborhood_operation
# - textbook
# 
# 
