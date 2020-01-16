#ADOPT

Although aviation accidents are rare, safety incidents occur more frequently and require a careful analysis to detect and mitigate risks in a timely manner. Analyzing safety incidents using operational data and producing event-based explanations is invaluable to airline companies as well as to governing organizations such as the Federal Aviation Administration (FAA) in the United States. However, this task is challenging because of the complexity involved in mining multi-dimensional heterogeneous time series data, the lack of time-step-wise annotation of events in a flight, and the lack of scalable tools to perform analysis over a large number of events. We propose a precursor mining algorithm: Automatic Discovery of Precursors in Time series data (ADOPT) that identifies events in the multidimensional time series that are correlated with the safety incident. Precursors are valuable to systems health and safety monitoring and in explaining and forecasting safety incidents. Current methods suffer from poor scalability to high dimensional time series data and are inefficient in capturing temporal behavior. We propose an approach by combining multiple-instance learning (MIL) and deep recurrent neural networks (DRNN) to take advantage of MIL's ability to learn using weakly supervised data and DRNN's ability to model temporal behavior. 


The objective of this project is to automate the analysis of flight safety incidents in a way that scales well and offers explanations. These explanations include:

* When the degraded states start to appear?
* What are the degraded states?
* What is the likelihood of the event is to occur?
* What corrective actions can be taken?

This project aims to:

* Create a novel deep temporal multiple-instance learning (DT-MIL) framework that combines multiple-instance learning with deep recurrent neural networks suitable for weakly-supervised learning problems involving time series or sequential data. 
* Provide a novel approach to explaining safety incidents using precursors mined from data.
* Deliver a detailed evaluation of the DT-MIL model using real-world aviation data and comparison with baseline models. 
* Perform a precursor analysis and explanation of high speed exceedance safety incident using flight data from a commercial airline







This repository contains the following files in its top level directory:

* [source](source)  
The source code of the repository, this includes the ADOPT model, GUI configuration tools, and a command line program that utilizes the model.

* [documentation](documentation)  
Documents describing how to configure and run the program, as well as how to interpret the results. 

* [datasets](datasets)  
A directory containing a sample dataset. Other datasets may also be added here by the user.

* [requirements.txt](requirements.txt)   
General module requirements for the program. A more specific requiremnts.txt can be found in [source](source)
* [ADOPT NASA Open Source Agreement](ADOPT NASA Open Source Agreement.pdf)  
Licensing for ADOPT
* [ADOPT Individual CLA.pdf](ADOPT Individual CLA.pdf)  
NASA Individual Contributor License Agreement
* [ADOPT Corporate CLA.pdf](ADOPT Corporate CLA.pdf)   
NASA Corporate Contributor License Agreement




##Contact Info

NASA Point of contact: Nikunj Oza <nikunj.c.oza@nasa.gov>, Data Science Group Lead.

For questions regarding [the research and development of the algorithm], please contact Bryan Matthews <bryan.l.matthews@nasa.gov>, Senior Research Engineer.

For questions regarding the source code, please contact Daniel Weckler <daniel.i.weckler@nasa.gov>, Software Engineer.


##Copyright and Notices

Notices:

Copyright © 2019 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

