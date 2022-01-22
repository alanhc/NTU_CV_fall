#!/usr/bin/env python
# coding: utf-8

# # Computer Vision I (922 U0610) - Homework 3
# Author: alanhc
# 
# ID: r10944007
# 
# Date: 9/30

# ## README
# 0. create env: `conda env create -f environment.yml`
# 1. enter env: `conda activate ntu-cv`
# 2. run jupyter `jupyter notebook`

# - Write a program to generate images and histograms:
#     - (a) original image and its histogram
#     - (b) image with intensity divided by 3 and its histogram
#     - (c) image after applying histogram equalization to (b) and its histogram

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


# Todo: (a) original image and its histogram
# Algorithm:
## 1. 計算value的count
## 2. 畫圖，縱軸為次數，橫軸為pixel值

# 1.
count = np.zeros(256, dtype="int")
for y in range(h):
    for x in range(w):
        count[ img[y][x] ]+=1
# 2. 
import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.imshow(img, cmap="gray", vmax=255)
plt.colorbar()
plt.subplot(1,2,2)
plt.bar(range(256), count, width=1)
plt.title("histogram of lena")
plt.xlabel("pixel value")
plt.ylabel("count")
plt.savefig("output/1.png")


# In[4]:


# Todo: (b) image with intensity divided by 3 and its histogram
# Algorithm:
## 1. 以//3將每個pixel整/除3 
## 2. 計算value的count
## 3. 畫圖，縱軸為次數，橫軸為pixel值

img_divide3 = img.copy()
# 1. 
for y in range(h):
    for x in range(w):
        img_divide3[y][x] = img_divide3[y][x]//3
# 2.
count_3 = np.zeros(256, dtype="int")
for y in range(h):
    for x in range(w):
        count_3[ img_divide3[y][x] ]+=1
# 3. 
import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.imshow(img_divide3, cmap="gray", vmax=255)
plt.colorbar()
plt.subplot(1,2,2)
plt.bar(range(256), count_3, width=1)
plt.title("histogram of lena")
plt.xlabel("pixel value")
plt.ylabel("count_3")
#plt.savefig("output/1.png")


# ### image after applying histogram equalization to (b) and its histogram
# ![]("./img/eq.svg") ---(1)

# In[5]:


# Todo: 計算pdf, cdf 並套用以上公式
# Algorithm:
## 1. 計算pdf
## 2. 計算cdf
## 3. 使用 Histogram equalization 的轉換式
## 4. 畫圖，縱軸為次數，橫軸為pixel值
# 1. 計算pdf
pdf = count_3 #pdf即為之前求的count
# 2. 計算cdf
cdf = np.zeros(pdf.shape)
cdf[0] = pdf[0]
for i in range(1, pdf.shape[0]):
    cdf[i] = cdf[i-1] +  pdf[i]
### 視覺化pdf及cdf
plt.figure(figsize=(16,4))
plt.subplot(1,2,1)
plt.bar(range(256), pdf, width=1)
plt.title("pdf")
plt.subplot(1,2,2)
plt.bar(range(256), cdf, width=1)
plt.title("cdf")


# In[6]:


# 3. 使用 Histogram equalization 的轉換式
## 3-1 求轉換式參數 L, cdf_min, cdf_max
L=256
cdf_min = 1e9
cdf_max = -1

for i in range(cdf.shape[0]):
    cdf_max = max(i, cdf_max)
    cdf_min = min(i, cdf_min)
## 3-2 建立查詢表 h()
f_h = np.zeros(256).astype("int")
for i in range(f_h.shape[0]):
    f_h[i] = np.round( (cdf[i]-cdf_min)/((h*w)-cdf_min) * (L-1) )


# In[7]:


## 3-3 根據 h() 轉換pixel
img_hisogram_equalization = np.zeros(img.shape)
h, w = img_hisogram_equalization.shape
for y in range(h):
    for x in range(w):
        img_hisogram_equalization[y][x] = int(f_h[ img_divide3[y][x] ])
## 3-4 計算pixel count

count_equalizatio = np.zeros(256, dtype="int")
for y in range(h):
    for x in range(w):
        count_equalizatio[ int(img_hisogram_equalization[y][x]) ]+=1


# In[8]:


import matplotlib.pyplot as plt
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.imshow(img_divide3, cmap="gray", vmax=255)
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(img_hisogram_equalization, cmap="gray", vmax=255)
plt.colorbar()
plt.subplot(2,2,3)
plt.bar(range(256), count_3, width=1)
plt.title("histogram of lena")
plt.xlabel("pixel value")
plt.ylabel("count")
plt.subplot(2,2,4)
plt.bar(range(256), count_equalizatio, width=1)
plt.title("histogram of lena")
plt.xlabel("pixel value")
plt.ylabel("count")


# 可以由上圖看出來，使用 Histogram equalization 可以增加整張圖片的對比度
# ![](./img/Histogrammeinebnung.png)

# ## Reference
# 1. https://en.wikipedia.org/wiki/Histogram_equalization
# 
# 
# 
# 
# 
