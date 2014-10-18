#maintaining now!

#Stacked Multi-layer Self-Organizing Map for Background Modeling (SMSOM-BM)

An implementation of our unpublished paper **[1]**, and the experiment results of **[1]** are based on it. If you have any question, please feel free to contact zhaozj89@gmail.com.

#Working environment

* Windows 7 (64bit)


* Visual Studio 2010


* OpenCV 2.4.5


* CUDA 5.0




#How to use


###For Windows users:


The exutable binary file is .\Debug\smsom.exe; therefore, you should first use cmd in Windows to navigate to the directory **Debug**. Please ensure Windows can find OpenCV library in your computer, and you have CUDA compatible GPU installed in your computer.


* navigate to Debug


* execute: `smsom <start_frame_number> <end_frame_number> <input_file_name> <output_file_name>`



For example, if I put the input images in: **E:\Data\input\**, 
the image files' name format is: **in000001.jpg** (any number), and I use 1-100 images to train the model, then I can execute:



`smsom 1 100 E:\\Data\\input\\in%06d.jpg E:\\Data\\results\\bin%06d.png`,

where I put the result images in **E:\Data\**.


###For Linux users:


You have to build yourself. The source code of 3 layer SMSOM-BM is **.\smsom\main.cu**, and you can refer to **[2]** for how to use CUDA on Linux platform.


Dataset
=====
The dataset used in our paper can be downloaded at **[3]**.



References
=====

[1] Zhenjie Zhao, Xuebo Zhang, and Yongchun Fang. Stacked Multi-layer Self-Organizing Map for
Background Modeling. Submitted to IEEE Transactions on Image Processing.

[2] http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3DimlP7Yp

[3] http://www.changedetection.net/

