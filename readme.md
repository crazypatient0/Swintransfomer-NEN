# SWIN-TRANSFORMER-NEN
- This algorithm is a combination of Swin-T network and Node equalization network.
- It is constructed to solve the problem of classification of metastatic and non-metastatic thyroid cancer cells.
- In this algorithm we use Swin-t as backbone to convert input image to a probability matrix,then use Node equalization
network to compute the average of matrix
- Finally, a threshold is trained to differentiate the metastatic and non-metastatic thyroid cancer cells.

## First Part - image preprocess
- A variety of basic functions are used to preprocess the images in this stage,such as
clahe,dilate,erode,laplician,canny
- after preprocess we finally got two different dataset, One contains RGB IMG and the other contains GRAY ones.

## Second Part - Swin-t training
* Structure of dataset
> -train 
>> tm1 (20000 images of metastatic)  
>> tm0 (20000 images of non-metastatic)  

> -val
>>tm1 (5000 images of metastatic)  
>> tm0 (5000 images of non-metastatic)  

- Feed Swin-T network with RGB and GRAY dataset which contain totally 100000 images.
- After a few times training, we've captured the best-performing network models.
- Download trained pth from << BAIDU NET DISK>> 
- trained pth (code：h7t8)https://pan.baidu.com/s/1HAkVu5PmJeRP0mR18R0kvw?pwd=h7t8
- put them in folder _'predictpath'_

## Third part - Node equalization network
- In this algorithm we crop the raw image in to 9 block,assume that each block is related to its neighbors.
- In NEN model,each block share its probability,just lisk GCN network does,eventually it's going to converge.
- The output of NEN model is the final prediction probability of area shown in image being metastatic cancer cells.

## Fourth part - tested  threshold
- Evaluated this classifier using various metrics like acc,roc,tpr,tnr,fpr,fnr,auc 
- Eventually we find out best performing threshold which achieve 99% test accuracy,at the mean time 0% of fpr and 98%tpr!

## Get start！
1. git clone https://github.com/crazypatient0/Swintransfomer-NEN.git
2. copy your input image into dataset
3. run start.py (remember to change file path)