import cv2; 
import math; 
import numpy as np; 
def DarkChannel(im,sz): 
	b,g,r = cv2.split(im) 
	# hrr pixel ki intensity ko 
	dc = cv2.min(cv2.min(r,g),b); 
	cv2.imshow("DC",dc); 
	# hrr 15*15 patch ki intensity, min intensity k barabar ho jaegi ==> Kernel ka matlab kya hai dost 
	# ye basically 15*15 ka element create kra hai 
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)) 
	cv2.imshow("Kernel",kernel); 
	#Basically kernel ko iterate krva rhe hai dc prr, agr poora match krjae toh 1 else dark = cv2.erode(dc,kernel) 
	return dark 
	# Ye toh kaafi clear hai 
def AtmLight(im,dark): 
	[h,w] = im.shape[:2] 
	imsz = h*w 
	numpx = int(max(math.floor(imsz/1000),1)) 
	darkvec = dark.reshape(imsz,1); 
	imvec = im.reshape(imsz,3); 
	# sort krva dia gya hai yahan prr 
	# return indices of the sorted list 
	indices = darkvec.argsort(); 
	# brightest 0.1% pixels nikalne ki koshish ho skti hai ye
	indices = indices[imsz-numpx::] 
	# array filled with zero create krli hai 
	#iss poore loop k through bas average nikala hai last 0.1 % pixels ka because vo sabse jada brightest hongi 
	atmsum = np.zeros([1,3]) 
	for ind in range(1,numpx): 
	atmsum = atmsum + imvec[indices[ind]] 
	A = atmsum / numpx; 
	# sum ko divide krvake result bhej dia gya hai 
	# why is the length of => Also hrr index prr same value present hai=> Aisa hrr barr nhi hai dost=> Blsically RGB channel k liye Atmospheric Light 
	print(A) 
	return A 
def TransmissionEstimate(im,A,sz): 
	omega = 0.95; 
	#im3 mai kya dala gya hai? 
	im3 = np.empty(im.shape,im.dtype); 
	#print(im3) 
	#Bss loop ka matlab bta do bhai 
	for ind in range(0,3): 
	im3[:,:,ind] = im[:,:,ind]/A[0,ind] 
	# yahan krdia gya hai transmission map calculate 
	T = omega*DarkChannel(im3 ,sz) 
	transmission = 1 - T 
	return transmission
	# Guided Filter and transmission refine I think are a part of Softmatting 
def Guidedfilter(im,p,r,eps): 
	mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r)); 
	mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r)); 
	mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r)); 
	cov_Ip = mean_Ip - mean_I*mean_p; 
	mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r)); 
	var_I = mean_II - mean_I*mean_I; 
	a = cov_Ip/(var_I + eps); 
	b = mean_p - a*mean_I; 
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r)); 
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r)); 
	q = mean_a*im + mean_b; 
	return q; 
def TransmissionRefine(im,et): 
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); 
	gray = np.float64(gray)/255; 
	r = 60; 
	eps = 0.0001; 
	t = Guidedfilter(gray,et,r,eps); 
	return t; 
def Recover(im,t,A,tx = 0.1): 
	res = np.empty(im.shape,im.dtype); 
	t = cv2.max(t,tx); 
	for ind in range(0,3): 
	res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind] 
	return res
def main(): 
	fn = 'input4.jpg' 
	src = cv2.imread(fn); 
	if(src.all() == None): 
	print("Correct path of image is not provided") return; 
	cv2.imshow("dark",src); 
	I = src.astype('float64')/255; 
	#I= src; 
	cv2.imshow("Normalize",I); 
	dark = DarkChannel(I,15); 
	A = AtmLight(I,dark); 
	te = TransmissionEstimate(I,A,15); 
	t = TransmissionRefine(src,te); 
	J = Recover(I,t,A,0.1); 
	cv2.imshow("dark",dark); 
	cv2.imshow("t",t); 
	cv2.imshow('I',src); 
	cv2.imshow('J',J); 
	cv2.imwrite("./image/J.png",J*255); 
	cv2.waitKey(); 
	if __name__ == '__main__': 
	import sys 
	main()
