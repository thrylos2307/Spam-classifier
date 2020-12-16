
The goal is to classify the message as spam or non-spam using multinomial Naive Bayes algorithm.
To classify messages as spam or non-spam ,we follow these steps to achieve our goal:-

* Learns how humans classify messages.
* Uses that human knowledge to estimate probabilities for new messages — probabilities for spam and non-spam.
* Classifies a new message based on these probability values — if the probability for spam is greater, then it classifies the message as spam. Otherwise, it classifies it as non-spam (if the two probability values are equal, then we may need a human to classify the message).

The dataset can be found in [The UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection).

The Naive Bayes algo uses the conditional probability to classify the message as spam or non-spam. It assume each feature as the individual feature unrelated to the presence of other feature . Here feature refers to the words in each message.

Steps we follow:

1. So first we take the attributes ,with which we going to deal finally ,that are messages and labels(spam or non spam). 
2. We remove the char like punctuations that are not impacting much while calculating probability for each word.
3. We split each word in each message, and create a list of all unique words(vocabulary say) that we have in the entire dataset.
4. Create a column for each word in vocabulary with 0 as initialization value and append it to the original dataset and mark the frequency of each word occuring in a particular message. One can think of as what we do in [get_dummies()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html).
Our table would look something like this:

| Label	| SMS	| 0	| 000pes	| 008704050406	| 0089	| ... 	| zoe	| zogtorius	| zouk	| zyada	| 鈥 |
| ---- |----|----|----|----|----|----	|----|-----|----|-----|----|----|-----|-----|-----|-----|-----|-----|-----|----|
| ham | [yep, by, the, pretty, sculpture] | 0 | 0 | 0 | 0 |	... | 0 | 0 | 0 | 0 | 0 |
| ham	| [yes, princess, are, you, going, to, make, me,...] | 0 | 0| 0 |  0 |	... | 0 | 0 | 0 | 0 | 0 |
| ham	| [welp, apparently, he, retired] | 0 |0 | 0  | 0 |	... | 0 | 0 | 0 | 0 | 0 |
| ham	| [havent] | 0 | 0 | 0 | 0 | ... | 0 | 0 | 0 | 0 | 0 |

5. Now we calculate the number of words in spam and non spam and find the frequency of each word in spam and non spam  in order to find the probability of each word given its belong to spam or non-spam message . The formula is given by:<br>
* for probability of word belonging to spam message: &emsp;
    <img src="https://render.githubusercontent.com/render/math?math=P(w_i|Spam) = \frac{N_{w_i|Spam} %2B \alpha}{N_{Spam} %2b \alpha \cdot N_{Vocabulary}}">        
 

* for probability of word belonging to non-spam message: &emsp;
<img src="https://render.githubusercontent.com/render/math?math=P(w_i|Ham) = \frac{N_{w_i|Ham} %2b \alpha}{N_{Ham} %2b \alpha \cdot N_{Vocabulary}}"> 

N_{w_i|spam}=total occurence of word belonging to spam and similary for ham.<br>
N_{Vocabulary}=unique number of word in the dataset.<br>
alpha=known as laplace smoothing parameter, used when we don't have the occurence of word in the class.

6. To classify new message we can need to find the conditional probability that a message is spam or non-spam given vocabulary(words), the fomula is given as :<br>
* for spam:&emsp;
 <img src="https://render.githubusercontent.com/render/math?math=P(Spam | w_1,w_2, ..., w_n) \propto P(Spam) \cdot \prod_{i=1}^{n}P(w_i|Spam)">  

* for non-spam:&emsp;
 <img src="https://render.githubusercontent.com/render/math?math=P(Ham | w_1,w_2, ..., w_n) \propto P(Ham) \cdot \prod_{i=1}^{n}P(w_i|Ham)">  

 Conditional probability is given by: P(Y | X) =P(X |Y )*P(Y ) / P(X)<br>
 so as we see for each new message:<br>
 P(Y = y0 | X = x) = P(X = x |Y = y0 )⋅ P(Y = y0 )/ P(X = x)<br>
P(Y = y1 | X = x) = P(X = x |Y = y1)⋅ P(Y = y1)/ P(X = x)<br>
.
.
.<br>
the value of x remain same ,then often we can write it as: P(Y | X)∝ P(X |Y )* P(Y )



So now we can classify each new message as spam and non-spam , for getting more better result on real world example we need to collect more dataset from messages ,emails ,etc.





