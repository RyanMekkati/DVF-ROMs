# DVF-Algorithms-2D-3D

A motion detection algorithm in C++ that takes in input sequences of png files and shows the DVF of the sequence.

A few methods are used to be compared ultimately:

1. Optical flow (Horn & Schnuck)
Non diffeomorphic DVF  

2. LDDMM
Very elegant method that uses principles of diffeomorphisms, Lie groups and Lie algebras.



This was an introduction to my thesis on adaptive radiotherapy and detection motion of mobile tumors by combining deep learning with stochastocial proccesses, differential geometry and topology. Some results are also presented for the different methods


Dimensionality reduction:

It is a must to have some dimensionality reduction for the problem at hand, especially if we want to do real time tracking. ROI tracking would already be one way of doing it. 

Also using multiple 4DCTs of people to do principal component analysis would also be a good idea


MULTIPLE 4DCT ON WHICH WE CALCULATE THE LDDMM FLOW, WE THEN EXTRACT PRINCIPAL COMPONENTS GIVING N VECTOR FIELDS. WE CAN THEN EXPONENTIATE THEM TO GET PRINCIPLE COMPONENTS OF DIFFEOMORPHISMS, AND MAYBE EXPRESS EACH TRANSFORMATION BY A COMPOSITION OF THOSE N DIFFEOS?!?!?!
