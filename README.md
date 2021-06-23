## Modeling of the spreading of forest wildfire using a convLSTM neural network. Prediction 3-days forward.




#### **Date: 23rd April 2020**
#### **Author: Bessam Mehenni**

### **Subject : Modeling of the spreading of a forest wildfire using a ConvLSTM recurrent neural network. Province of Alberta, Canada.**<br/><br/>
Full report is available and can be downloaded (“Report_EN”)<br/><br/>
![PIC8](/md_images/spread_gif.gif)


##  **1.	Introduction**

This study was carried out as part of a final academic project at the Datascience Jedha school in Paris. The aim is to test the ability of a 2D spatial model of the ConvLSTM type to predict the propagation of a forest fire. 

This is an exploratory study that implements a deep learning from scratch. The reader is cautioned that this study is not based on a thorough knowledge of the state of the art in forest fire spread modeling.

The model we design uses sequences of thermal anomaly coordinates collected by satellite [1] and processed by me to obtain the appearance of a pixel image. Also, features that contribute to fire propagation are associated to each pixel and are taken into account by the model. The meteorological features come from a government site of Alberta [2].

The fire sequences we are studying occurred specifically in the Alberta region of Canada. They are large fires. Four fire sequences, including the famous Fort McMurray fire in May 2016 and the Richardson fire in June 2011, will be used to train and test the model. I gave the names "Richardson phase 2" and "Central Alberta" to two fire sequences, these names are fully personal.


![PIC1](/md_images/pic1.PNG)


## **2.	Data preprocessing**

The model actually admits two images and an associated matrix of fire influence features that correspond to two successive days of data. The output is an image of the fire positions, a prediction at a horizon of 3 days after the last input image.

Each input sample of the model is a 5D-array (sample_nb, nb_steps, rows, columns, nb_features) where nb_steps is the 2-day time step, rows and columns are the number of rows (here 45) and columns (here 65), nb_features is the number of fire influence features, here 9. The image has 2925 pixels, each pixel represents an area of 2km x 2km.

The features of influence we have chosen are of several kinds:
-	meteorological: precipitation (mm), hygrometry (%), solar radiation received on the ground (MJ/m2), aggregated average wind speed (km/h) and wind direction (°).
-	topographic: elevations of each cell. 
-	vegetation: cumulative precipitation (mm) assumed to reflect to some extent the water content in the vegetation.
-	fire characteristic: brightness.

In the framework of this study, the human factor of influencing the propagation is excluded, in particular the fire fighting and prevention measures intended to control a fire (counterfire, pruning of vegetation zones, watering etc.).

The wind direction is a quantity resulting from the readings of two or three weather stations around the pixel under consideration. It is obtained by an inverse distance weighted (IDW) interpolation method.

The air temperature factor is not taken into account because it is intrinsic to air humidity (see diagram of humid air). 
The slopes of the relief, a factor of importance of propagation, are not explicitly given to the model. Only the altitudes of the cells are given. 

![PIC2](/md_images/pic2.PNG)

We decided, rightly or wrongly, to use the Brightness, a quantity measured in the same way as the position of the light by the satellite sensor. We believe that the infrared radiation is probably a mirror image of the temperature of the fire and thus of the fire intensity, which research has shown to influence the propagation of the incandescent particles [3].

Data collected from fires in Alberta suggest several modes of fire spread. One of the phenomena is characterized by enormous apparent fire velocities and is proving to be very devastating. It is responsible for fire travel distances of several kilometers per day. This phenomenon leaves a swarm of small fires in its trail. 

It is a phenomenon that can be observed in Canada, the United States, Australia and sometimes in some southern European countries such as Portugal. The explanation is that incandescent plant particles called “brandons” are projected forward. They participate in the very rapid advance of the flame front. This fire behaviour is very clearly different from other, more conventional behaviour, for which the apparent fire velocity is a few km per day.<br/><br/>

![PIC3](/md_images/pic3.PNG)



We have revealed this phenomenon during the Fort McMurray fire episode in Canada by applying a descriptive modeling (Dbsan). A few days of occurrence of this phenomenon in May 2016 resulted in an extraordinary fire spread. The physical factors that make such a phenomenon possible are complex to identify. They are related to the presence of convective instabilities in the atmosphere. Also the heat power of the fire line will closely determine the distance covered by the incandescent particles. 

The dataset is divided chronologically into two parts: a 38-day training sample (approx. 56% or 111150 observations) and a 30-day validation sample (approx. 44% or 87750 observations). We also reserve a 9-day test sample. <br/><br/>

## **3.	Architecture of the neural network and associated parameters**

The proposed architecture consists of a first layer of 64 kernel ConvLSTM with a kernel size of (3x3), followed by a Maxpooling layer (2x2) which performs a subsampling operation to reduce the dimensions of the feature maps, followed by a dimension flattening and three fully-connected layers of 128,128 and 2925 neurons layers each. The layers are linked to a ReLu activation feature. A one-way classification is done at the output of the last layer using a sigmoid activation function that calculates the probability of the fire modality ("1"). 


![PIC4](/md_images/pic4.PNG)

In order to mitigate model overfitting, regularization techniques are introduced such as L2 regularization applied to certain layers. Some layers are followed by a drop-out at a rate of 0.4. The Adam gradient optimization algorithm has been selected. The model is using the Keras library and Tensorflow as back-end in a GPU acceleration environment (Google Colab).<br/><br/>

## **4.	Classification results**

The following figures are the confusion matrix and the ROC curve on the test sample. It can be seen that the model predicted correctly 22 pixels in fires. The Precision is 2.3%: the model made a lot of mistakes by having predicted a large number of pixels on fire where in reality there were no fires.

![PIC5](/md_images/pic5.PNG)
![PIC6](/md_images/pic6.PNG)


The overall Accuracy Score is 90.0%. We must pay special attention to the Recall parameter because it is a problem not to predict burning pixels that are actually burning. Here the Recall, which we would like to see as high as possible, is only 14%. 
The following figure compares the predicted and actual images over the 9 days of the test sample. The fire probability threshold is set at 0.65.

![PIC7](/md_images/pic7.PNG)


We can see that the predictions are scattered pixels of fire where in reality the fire zones are gathered. 

A positive point is that there is an intensification of fire activity on day n°3 and that the model in its prediction follows this trend. The observation is the same on day n°6 with a shift of the prediction to day n°7. This delay may be related to the uncertainty of the predictions 3 days in advance.

On the contrary, day n°9 shows a reduction in fire activity, which the model in its prediction also shows. 

Convolution identified patterns in the training images. The fire patterns are repeated from one day to the next, accompanied by the highlighting of certain areas of the image. These areas change from day to day. It is possible that the model was not sufficiently trained with heterogeneous cases. Our study suffers from a limited amount of training data. In this sense, we can distinguish a limit to the convolution: the training images did not have fire zones in the lower left corner of the images. This characteristic can be found in the model predictions, which favours non-fire in this part of the image.<br/><br/>

## **5.	Conclusion**

Our model does not provide an accurate view of the patterns of fire spread at three-days forward. The reason for this limitation could be the convolution principle itself and/or the limited amount of data used for its training and/or the complexity of the phenomena that the model is unable to understand based on the features we have given it.
The fire data are from significant fire episodes in terms of duration and extent. It seems difficult to image that we can collect more fire data from Alberta in large quantities. 

Nevertheless, our model has shown an ability to track an increase or decrease in fire activity, depending on the combined weather and non-weather conditions.
It should be pointed out that we have naively fed our model by putting it in charge of building its understanding of propagation phenomena. But phenomena are not simple. They are complex and heterogeneous. For example, in the Fort McMurray surveys, we have observed that during two different days with similar conditions (wind speed, hydrometry, etc.), antagonistic phenomena occurred. In one case, a spread of 5 km in one day. In the other case, a propagation by particle jumps of more than 50 km. 

We can say that the direct factors we have chosen are not sufficient to explain these phenomena. Other parameters related to the physics of fire and air in the atmosphere are involved.<br/><br/>

## **6.	Next**

Research has investigated how fire and wind conditions affect the distribution behaviour of incandescent vegetation particles falling to the ground [3]. Froude's dimensionless number is introduced in order to propose laws for the calculation of the mean distance of fire jumps. The Froude number is a parameter used in fluid mechanics models to describe the flow regime. It is representative of the relative importance of inertial forces related to flow velocity versus floatability forces. In the case of the flare-up of the brandons, the regimes present may correspond either to a flow governed by the lateral wind or to an upward flow governed by the intensity of the fire. Formulas for calculating the average distance of the jump fire have been proposed as a function of the flow regime [3]. 

A model can aspire to performance if we make it easier for it to understand complex phenomena. I think that the best models for predicting fire propagation will probably be at a crossroads between Deep learning (or even Reinforcement learning) and physical modelling. It will have to rely, in addition to the direct factors known to influence fire propagation (humidity, wind speed, etc.), on parameters generally used in physical modelling to explain phenomena. The Froude number would be one of them.<br/><br/>

## **7.	References**

[1] https://firms.modaps.eosdis.nasa.gov/, the 27th/03/2020

[2] http://www.agriculture.alberta.ca/acis/, the 27th/03/2020

[3] Nicolas Sardoy. Transport et combustion de particules générées par un feu de végétation. Sciences de l’Ingénieur [Physics]. Université de Provence - Aix-Marseille I, 2007.Français. tel-00289521v2

