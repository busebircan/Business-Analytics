# Advanced Data Modelling for Text Data Using Sentiment Analysis and LDA Topic Modelling
Amazon Shopping App Review Analysis Example
# Objective
Mobile application performance is one of the key indicators of customer satisfaction and thus success
for online retailing companies in rapidly growing, dynamic and competitive e-commerce business
environment. Through app stores such as Google Play Store allowing customers to share scores and
reviews for the apps, online retailers have the chance to easily reach direct customer feedback which is
decidedly one of the most effective ways of defect-hunting for their app as well as products and services
provided with it.
Amazon Shopping app example is covered in this text mining study by extracting and analysing
customer reviews from Google Play Store, the official app store for certified devices running on the
Android operating system and the largest app store in the world, using sentiment analysis and topic
modelling. Most recent reviews at the time of the study were extracted and analysed for most up-to-date
insights on possible quality issues of the app and/or services which are reported as results and
conclusions of this project.
# Methodology
Sentiment Analysis and Topic Modelling
Sentiment Analysis: Sentiment analysis is the task of extracting the positive or negative orientation expressed in a text [1].
The process of the analysis generally consists of four steps which are tokenization, feature extraction,
classification and validation. Amazon app reviews in this study are tokenized into sentences and sentence level sentiment analysis is
implemented by SentiWordNet’s Vader Sentiment Analyser in Python.
Topic Modelling: In machine learning and natural language processing, topic models are generative models, which
provide a probabilistic framework [2]. Topics are defined as hidden relations to be estimated linking the
words and their occurrences and therefore determined as a collection of unigrams. A probabilistic topic modelling approach Latent Dirichlet Allocation (LDA) is implemented to our
collection of negative reviews. The output of LDA is a big probability mass function over all possible
words in modelled text for each individual topic. Thus, the distance between the topics is an estimation
of distribution difference i.e., semantic difference between them. By adjusting relevance metric, λ, it is
possible to rank the words with their frequency within a topic rather than its overall frequency in the
text which facilitates comprehension of semantic differences between the topics.
# Conclusions
By analysing most recent customer reviews on Amazon Shopping App with text mining techniques, we
conclude though the percentage is rather favourable, negative reviews from customers has been
increasing in numbers bearing signals for certain quality issues to be attended to not to lose overall
customer satisfaction. Detected main issues needful to be checked and resolved are;
- Refund process for returned items and/or cancellations,
- Customer service contact availability and communication with customers,
- App bugs and errors such as “Something went wrong” error,
- Delivery services; untimely and wrong address deliveries,
- Technical issues related to Alexa and galaxy devices.
Additionally, we can conclude trigram level frequency distribution can be very helpful to gain insights
from such text data consisting of reviews.
# References
[1] Jurafsky, D. and Martin, J., 2022. Speech and Language Processing. [online] Web.stanford.edu.
Available at: <http://web.stanford.edu/~jurafsky/slp3/> [Accessed 31 March 2022].

[2] Tong, Zhou & Zhang, Haiyi. (2016). A Text Mining Research Based on LDA Topic Modelling.
Computer Science & Information Technology. 6. 201-210. DOI: 10.5121/csit.2016.60616. Available
at:https://www.researchgate.net/publication/303563965_A_Text_Mining_Research_Based_on_LDA_
Topic_Modelling> [Accessed 2 April 2022].
