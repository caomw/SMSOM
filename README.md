#maintaining now!

#Stacked Multi-layer Self-Organizing Map for Background Modeling (SMSOM-BM)

An implementation of our unpublished paper **[1]**, and the experiment results of **[1]** are based on it. If you have any question, please feel free to contact zhaozj89@gmail.com.

#Preferred working environment

* Windows 7 (64bit)


* Visual Studio 2010


* OpenCV 2.4.5


* CUDA 5.0




#How to use


###For Windows users:


The exutable binary file is **.\Debug\smsom.exe**; therefore, you should first use **cmd** in Windows to navigate to the directory **Debug**. Please ENSURE Windows can find OpenCV library, and you have CUDA compatible GPU installed in your computer.


Then you have two options:

* If you have foreground free traning images, then execute:

  `smsom train <start_frame_number> <end_frame_number> <input_file_name> <output_file_name>`
    
  where `<start_frame_number>` and `<end_frame_number>` stand for the index range of the training images; `<input_file_name>`     is the format of the input image's name, and the last parameter `<output_file_name>` is optional, if you omit it, then the      output images are just shown in you screen, but not stored in your computer. 

  For example, if I put the input images in: **E:\Data\input\**, 
  the image files' name format is: **in000001.jpg** (any number), and I use 1-100 images to train the model, then I can execute:

  `smsom train 1 100 E:\\Data\\input\\in%06d.jpg E:\\Data\\results\\bin%06d.jpg`

  where I put the result images in **E:\Data\results**.

  or

  `smsom train 1 100 E:\\Data\\input\\in%06d.jpg`

  where I do not store the output images.
  
* If you do not have foreground free training images, you can execute:
  
  `smsom nottrain <input_file_name> <output_file_name>`

  where the meanings of `<input_file_name>` and `<output_file_name>` (optional) are the same as the previous case. In this    
  situation, we set the threshold tau=0.06 (see **[1]** for more details). 
 
###For Linux users:


You have to build yourself. The source code of 3 layer SMSOM-BM is **.\smsom\main.cu**, and you can refer to **[2]** for how to use CUDA on Linux platform.


Demos
=====
We have created some demo scripts (see **.\Debug\**) to show the performance of the method. You can use them by:

* download the dataset from **[3]**
* decompress the dataset
* copy @@ into **E:\Data\input\**
* navigate to **.\Debug**, and double click **demo1.bat** (perfectly safe)

You can try more data if you get used to this pattern of using this software.



References
=====

[1] Zhenjie Zhao, Xuebo Zhang, and Yongchun Fang. Stacked Multi-layer Self-Organizing Map for
Background Modeling. Submitted to IEEE Transactions on Image Processing.

[2] http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/#axzz3DimlP7Yp

[3] http://www.changedetection.net/

