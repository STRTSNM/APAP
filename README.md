# APAP
AI enabled Patient Appointment Prioritization

The problem that this project is trying to tackle is that appointments are not prioritized according to the patients' needs. Because of this ,
the patients who need an appointment the do not get don't get one in time.

TO tackle this issue, the idea is to make an algorithm that will categorize patients based on their diseases for their appointment. It will also
use the patients medical history or more specifically , the diseases that the patients had in the past.

To acheive this, the first level of classification uses support-vector-machines and the 2nd layer uses linear regression.

This algorithm is fairly simple. It takes the input as the select few diseases that the patients had in the past. here 0 means a negative for the disease and 1 for a positive. The input is classified into three categories, namely ni, i, vi.
This classification is done by SVM.

And now, the patients in each category are classified (or) prioritized further to acheive better accuracy. This is the 2nd level of classification and it is done by Linear regression.
