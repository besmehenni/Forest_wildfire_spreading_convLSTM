## Modeling of the spreading of forest wildfire using a neural network with ConvLSTM cells. Prediction 3-days forward.




#### **Date: originally issued 23rd April 2020, updated 24th June 2021 in Towardsdatascience.com**
#### **Author: Bessam Mehenni**

### **Subject : Modeling of the spreading of a forest wildfire using a neural network with ConvLSTM cells. Province of Alberta, Canada.**<br/><br/>
Full report is available and can be downloaded (“Report_EN rev 1”)

PDF of an extract of code (model with ConvLSTM) can be downloaded ("Extract_of_code")<br/><br/>

>_Timelaps I created from MODIS data about the Fort McMurray fire sequence_

![PIC8](/md_images/spread_gif.gif)


##  **1.	Introduction**

This study was conducted as part of a final project which I chose as a topic to close my training at the Jedha Data science school in Paris. The aim is to test the ability of a 2D spatial model with a ConvLSTM layer to predict the propagation of a forest fire. 

This is an exploratory study that implements a deep learning from scratch. The reader is cautioned that this study is not based on a thorough knowledge of the state of the art in forest fire spread modeling. The author is not knowledgeable in the discipline of forest fires either.

The model we design uses sequences of thermal anomaly coordinates collected by satellite [1] and processed by me to obtain the appearance of a pixel image. Also, features that contribute to fire propagation are associated to each pixel and are taken into account by the model. The meteorological features come from a government site of Alberta [2].

The fire sequences we are studying occurred specifically in the Alberta region of Canada. They are large fires. Four fire sequences, including the famous Fort McMurray fire in May 2016 and the Richardson fire in June 2011, will be used to train and test the model. The Fort McMurray fire has caused the displacement of 80,000 people and required the mobilization of 3000 firefighters. We gave the names "Richardson phase 2" and "Central Alberta" to two fire sequences, these names are purely fictional.


![PIC1](/md_images/pic1.PNG)

The data come from three different sources:
-	MODIS satellite data were downloaded ad-hoc from source [1] having specified the spatial and temporal windows. 
-	the meteorological data were downloaded ad-hoc from source [2].
-	the topography data were manually read from the source [3].<br/><br/>

## **2.	Approach**

The main axes of the grid established by the meteorological web site were used as the basis for developing a finer grid of tabular data. We began by conducting some manual readings to find out the coordinates of the weather grid.

![PIC10](/md_images/pic8.PNG)


Then, we had to build a finer secondary grid (we chose 2km x 2km) where each elementary mesh contains the following information: whether there is fire or not according to MODIS records, transposed or interpolated meteorological data, elevation data of the sites.
The major difficulties of this project concern the preprocessing of the data. It is a craftsman's work, difficult to generalize: 
-	the data collection is not automated and even involves manual readings.
-	the pre-processing was tailor-made for the four selected fire study regions.

We spent a lot of time on developing the pre-processing programs. During their execution, the pre-processing programs ran for hours during the night in order to arrive fortunately at the finished datasets that we provide here on GitHub (train_val & test datasets).

Once the preparation was complete, we started by visualizing the data. Our prejudices about forest fires, the idea that a forest fire spreads at a homogeneous speed from near to far, were turned upside down when we observed a noticeable propagation pattern of several tens of km per day. The wind intensities on those days at the weather stations could be described as "normal", which calls into question the idea of a primary cause related to the wind. Here it is in fact the intensity of the fire outbreaks that creates large ascending air flows; these flows carry incandescent particles and they participate in a very rapid advance of the fire.

We studied how fire clusters evolve, by means of the descriptive modeling DBSCAN. This technique was very useful to understand the propagation. It consists in assigning the cluster points from close to close. 

The next step was to try to model the fire propagation in order to predict it. We tried a deep learning model using ConvLSTM. A. Xavier says in [5]: "It is a Recurrent layer, just like LSTM, but internal matrix multiplicators are exchanged with convolution operations". This attempt to model fire propagation using ConvLSTM did not yield satisfactory results.


Fluid mechanics researchers have introduced dimensionless variables into the empirical correlations whose purpose is to model physical phenomena. Despite this failed attempt, our progress during this project leads us to believe that a purely data-driven model will have difficulties in translating propagation phenomena of such complexity. We believe that it would be relevant for a data-driven model to rely on some dimensionless variables of physics (Froude number to quote one).<br/><br/>


## **3.	References**

[1] https://firms.modaps.eosdis.nasa.gov/, the 27th/03/2020

[2] http://www.agriculture.alberta.ca/acis/, the 27th/03/2020

[3] https://earth.google.com/web/, the 27th/03/2020

[4] Nicolas Sardoy. Transport et combustion de particules générées par un feu de végétation (2007). Sciences de l’Ingénieur [Physics]. Université de Provence - Aix-Marseille I, 2007.Français. tel-00289521v2

[5] Alexandre Xavier from Neuronio, An introduction to ConvLSTM (25th/03/2019), https://medium.com/neuronio/an-introduction-to-convlstm-55c9025563a7

