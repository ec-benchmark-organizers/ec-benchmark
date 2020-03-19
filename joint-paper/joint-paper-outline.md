# Joint paper for the benchmarking exercise

This document serves as a proposal for the structure and content of the joint paper that is 
planned to contain the results of the benchmarking exercise on estimating extreme environmental 
conditions. This document shall support us in making decisions on the joint paper during the 
planned meeting at OMAE 2020.

Meeting time: TBA (ca. 3 hours some time between June 28 – July 3, 2020)  
Meeting location: OMAE 2020, Florida, USA, Room: TBA  
Not at OMAE? You can send a proposal which will be included in this document and you can vote 
on all proposals until 24 hours before the meeting via email (ecbenchmark@gmail.com).  

Maintainer of this document: Andreas F. Haselsteiner ("Andy"), a.haselsteiner@uni-bremen.de  
Document lives at: https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/joint-paper/joint-paper-outline.md

# Proposed structure and content

## 1 Introduction
Brief introduction on environmental contours and on this benchmarking exercise.

## 2 Participants

|Participant|Model for sea state data |Model for wind wave data |Contour construction method|Code  |
|-----------|-------------------------|-------------------------|---------------------------|------|
|1          |Global hierarchical model|Global hierarchical model|Direct sampling            |https://github.com/ec-benchmark-organizers/ec-benchmark/tree/master/participants-code/participant_1 |
|2          |Conditional extreme model|Global hierarchical model|IFORM                      |https://github.com/ec-benchmark-organizers/ec-benchmark/tree/master/participants-code/participant_2 |
|3          |Kernel density model     |Kernel density model     |Highest density            |- |

### 2.1 Contour method 1
Description of used methods on half a page (200-400 words and up to 1 table or figure).  
Contribution by: Jane Doe & John Doe

### 2.2 Contour method 2
Description of used methods on half a page (200-400 words and up to 1 table or figure).  
Contribution by: Jane Doe & John Doe

### 2.3 Contour method 3
Description of used methods on half a page (200-400 words and up to 1 table or figure).  
Contribution by: Jane Doe & John Doe

## 3. Results 
### 3.1 Sea state contours
**Figure with 2x3 panels that shows the contours**  
Panel 11: Dataset A with all 1-yr contours.  
Panel 12: Dataset B with all 1-yr contours.  
Panel 13: Dataset C with all 1-yr contours.  
Panel 21: Dataset A with all 20-yr contours.  
Panel 22: Dataset B with all 20-yr contours.  
Panel 23: Dataset C with all 20-yr contours.

**Note on the following tables:**  
Assuming that environmental state are independent, for any contour the number of 
expected datapoints anywhere outside the contour can be calculated as *N × αT* 
where *N* is the number of datapoints a sample holds and *αT* is the probability 
that an observations falls anywhere outside the contour. 
For IFORM, direct sampling, ISORM and highest density contours which are constructed 
using the same joint distribution and the same *αC*-value, the probability *αT* is different 
(where *αC* corresponds to the definition of probability of exceedance that is specific 
to the contour method).
For an IFORM contour, *αT* can be calculated using the chi squared distribution 
and the reliability index *β* (see https://arxiv.org/pdf/2003.05463.pdf, equation 12). 
For direct sampling contours  the number of expected datapoints outside the contour 
depends on the particular joint distribution, but is somewhat similar to the IFORM contour 
(see https://arxiv.org/pdf/2003.05463.pdf, section 6). For ISORM and highest density 
contours *αC = αT*.


|Participant | # points outside the 1-year contour | Expected # points outside |
|------------|-------------------------------------|---------------------------|
|1           | 24±7 (A: 30, B: 10, C: 40)          |20  (Highest density)      |
|2           | 84±12 (A: 70, B: 100, C: 85)        |196 (IFORM)                |
|3           | 150±40 (A: 70, B: 210, C: 60)       |ca. 196 (Direct sampling)  |

&nbsp;

|Participant | # points outside the 20-year contour | Expected # points outside |
|------------|--------------------------------------|---------------------------|
|1           | 2.2±0.5 (A: 0, B: 3, C: 3)           |1  (Highest density)       |
|2           | 10.1±5.3 (A: 3, B: 10, C: 13)        |11.5 (IFORM)               |
|3           | 4.1±1.0 (A: 2, B: 5, C: 5)           |ca. 11.5 (Direct sampling) |


### 3.2 Wind-wave contours
**Figure with 2x3 panels that shows the contours**  
Panel 11: Dataset D with all 1-yr contours.  
Panel 12: Dataset E with all 1-yr contours.  
Panel 13: Dataset F with all 1-yr contours.  
Panel 21: Dataset D with all 50-yr contours.  
Panel 22: Dataset E with all 50-yr contours.  
Panel 23: Dataset F with all 50-yr contours.  


|Participant | # points outside the 1-year contour | Expected # points outside |
|------------|-------------------------------------|---------------------------|
|1           | 24±7 (A: 30, B: 10, C: 40)          |50  (Highest density)      |
|2           | 184±12 (A: 170, B: 200, C: 185)     |492 (IFORM)                |
|3           | 150±40 (A: 70, B: 210, C: 60)       |ca. 492 (Direct sampling)  |

&nbsp;

|Participant | # points outside the 50-year contour | Expected # points outside |
|------------|--------------------------------------|---------------------------|
|1           | 2.2±0.5 (A: 0, B: 3, C: 3)           |1  (Highest density)       |
|2           | 10.1±5.3 (A: 3, B: 10, C: 13)        |12.0 (IFORM)                  |
|3           | 4.1±1.0 (A: 2, B: 5, C: 5)           |ca. 12.0 (Direct sampling)    |

### 3.3 Uncertainty of the wind-wave contours

**Figure with Nx3 panels that shows contour overlays**  
Panel 11: Participant 1 with contours based on 1 year of data.  
Panel 12: Participant 1 with contours based on 5 year of data.  
Panel 13: Participant 1 with contours based on 25 year of data.  
Panel 21: Participant 2 with contours based on 1 year of data.  
Panel 22: Participant 2 with contours based on 5 year of data.  
Panel 23: Participant 2 with contours based on 25 year of data.  
Panel N1: Participant N with contours based on 1 year of data.  
Panel N2: Participant N with contours based on 5 year of data.  
Panel N3: Participant N with contours based on 25 year of data.  

**Figure with Nx3 panels that shows contour's confidence intervals**  
Panel 11: Participant 1 with contours based on 1 year of data.  
Panel 12: Participant 1 with contours based on 5 year of data.  
Panel 13: Participant 1 with contours based on 25 year of data.  
Panel 21: Participant 2 with contours based on 1 year of data.  
Panel 22: Participant 2 with contours based on 5 year of data.  
Panel 23: Participant 2 with contours based on 25 year of data.  
Panel N1: Participant N with contours based on 1 year of data.  
Panel N2: Participant N with contours based on 5 year of data.  
Panel N3: Participant N with contours based on 25 year of data.  

## 4. Discussion

## 5. Conclusions

## A. Appendix

### A.1 Details about the submission of participant 1
Optionally, in an appendix further details for each 
contribution can be given that would exceed the word limit in seection 2.

### A.2 Details about the submision of participant 2

# Timeline and kind of publication
Meeting at OMAE 2020: Decisions on the structure and content of the paper  
July 31st: Deadline for first complete draft  
August 15th: Deadline for submitting comments on the first draft  
August 30st: Deadlien for revised draft  
September 15th: Deadline for comments on revised draft  
September 31st: Submission of the manuscript  
  
Kind of publication: Research article in a journal  
Target journal: To be discussed. @Lance: Special issue in the Journal of Offshore Mechanics and Artic Engineering?  

# Proposals for further analysis
At the OMAE 2020 meeting we will decide on the following proposals. 
  
Feel free to send me your proposals and I will add them here (a.haselsteiner@uni-bremen.de).  
  
If not all participants agree, we will vote and each participating team has one vote. 

## Proposal A: Assessment of statistical models in terms of marginal and projected variables

The estimation of environmental contours (usually) requires two distinct steps: firstly, 
the joint distribution of variables is estimated, then contours are constructed from the 
joint distribution using some method (IFORM, direct sampling, highest density regions, etc.). 
Contours for the same joint distribution which are constructed using different methods 
can differ significantly. It is therefore useful to directly compare the different statistical 
models for each dataset. One way to compare the statistical models is using exceedance plots 
for the marginal variables, which compare the observed data and fitted model (see Figure A1 below). 
As well as the marginal variables, these plots can also be produced for the marginal quantities 
under a rotation of the axes. That is, the quantities Z_θ=X cos⁡θ+Y sin⁡θ, for various angles θ. 
The marginal variables X and Y correspond to Z(0) and Z(90°), respectively. It is proposed 
that participants supply the following information for their statistical models for both (H_s, T_z) 
and (H_s, U_10):  

| Exceedance probability	|5 × 10^(-1) | 2 × 10^(-1) | 1 × 10^(-1) | 5 ×10^(-2) | 2 × 10^(-2) |1 × 10^(-2) | … | 1 × 10^(-6) |
|---------------------------|------------|-------------|-------------|------------|-------------|------------|---|-------------|
|Z(0°)                      | z(θ,α)     | ...         |             |            |             |            |   |             |						
|Z(45°)                     | ...        |             |             |            |             |            |   |             |									
|Z(90°)                     |            |             |             |            |             |            |   |             |	
|...                        |            |             |             |            |             |            |   |             |
|Z(315°)                    |            |             |             |            |             |            |   |             |

The quantity z(θ,α) is the quantile of Z(θ) at exceedance probability α. This information would 
show how well the statistical model captures the tail of the joint distribution when projected 
onto the margins and lines at ±45°. It would allow a comparison of the statistical models 
that is independent of the tail of the distribution. Participants could calculate these 
quantiles either by integrating the fitted statistical models or by Monte Carlo simulation.  


![alt text](https://github.com/ec-benchmark-organizers/ec-benchmark/blob/master/joint-paper/gfx/proposal-a-figure.jpg "Example exceedance plot for Hs.")  
*Figure A1: Example exceedance plot for Hs.*  

Proposed by: Edward Mackay, e.mackay@exeter.ac.uk

# Proposals against some suggested content

## Proposal I: Sample proposal against having Figure X in the paper
Description of the proposal.  
Proposed by: John Doe, john.doe@sampleuniversity.com