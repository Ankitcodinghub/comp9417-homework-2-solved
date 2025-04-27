# comp9417-homework-2-solved
**TO GET THIS SOLUTION VISIT:** [COMP9417 Homework 2 Solved](https://www.ankitcodinghub.com/product/comp9417-machine-learning-solved-7/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;124057&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;3&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (3 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;COMP9417 Homework 2 Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (3 votes)    </div>
    </div>
Tutorial: Classification

Question 1. Distance between parallel hyperplanes

When constructing linear classifiers, a common calculation that comes up is to compute the distance between two parallel hyperplanes. Consider two parallel hyperplanes: H1 = {x ∈ Rn : wTx = a} and H2 = {x ∈ Rn : wTx = b}. Show that the distance between H1 and H2 is given by .

Hint: draw a picture.

Question 2 (Perceptron Training &amp; Capacity) (a) Consider the following training data:

x1 x2 y

-2 -1 -1

2 -1 1

1 1 1

-1 -1 -1

3 2 1

Apply the Perceptron Learning Algorithm with starting values w0 = 5, w1 = 1 and w2 = 1, and a learning rate η = 0.4. Be sure to cycle through the training data in the same order that they are presented in the table. Present your results in table form:

Iteration hw,xi yhw,xi w

(b) Consider the following three logical functions:

1. A ∧ ¬B

2. ¬A ∨ B

3. (A ∨ B) ∧ (¬A ∨ ¬B)

Which of these functions can a perceptron learn? Explain.

Question 3. Binary Logistic Regression, two perspectives

Recall from previous weeks that we can view least squares regression as a purely optimisation based problem (minimising MSE), or as a statistical problem (using MLE). We now discuss two perspectives of the Binary Logistic Regression problem. In this problem, we are given a dataset

where the xi’s represent the feature vectors, just as in linear regression, but the yi’s are now binary. The goal is to model our output as a probability that a particular data point belongs to one of two classes. We will denote this predicted probability by

P(y = 1|x) = p(x)

1

and we model it as

,

where wˆ is our estimated weight vector. We can then construct a classifier by assigning the class that has the largest probability, i.e.:

(1 σ(wˆTx) ≥ 0.5

if

yˆ = arg max P(yˆ = k|x) =

k=0,1 0 otherwise

note: do not confuse the function σ(z) with the parameter σ which typically denotes the standard deviation.

(b) We first consider the statistical view of logistic regression. Recall in the statistical view of linear regression, we assumed that y|x ∼ N(xTβ∗,σ2). Here, we are working with binary valued random variables and so we assume that

y|x ∼ Bernoulli(p∗), p∗ = σ(xTw∗)

where p∗ = σ(xTw∗) is the true unknown probability of a response belonging to class 1, and we assume this is controlled by some true weight vector w∗. Write down the log-likelihood of the data D (as a function of w), and further, write down the MLE objective (but do not try to solve it).

(c) An alternative approach to the logistic regression problem is to view it purely from the optimisation perspective. This requires us to pick a loss function and solve for the corresponding minimizer. Write down the MSE objective for logistic regression and discuss whether you think this loss is appropriate.

(d) Consider the following problem: you are given two discrete probability distributions, P and Q, and you are asked to quantify how far Q is from P. This is a very common task in statistics and information theory. The most common way to measure the discrepancy between the two is to compute the Kullback-Liebler (KL) divergence, also known as the relative entropy, which is defined by:

,

where we are summing over all of the possible values of the underlying random variable. A good way to think of this is that we have a true distribution P, an estimate Q, and we are trying to figure out how bad our estimate is. Write down the KL divergence between two bernoulli distributions P = Bernoulli(p) and Q = Bernoulli(q).

(e) Continuing with the optimisation based view: In our set-up, one way to quantify the discrepancy between our prediction pˆi and the true label yi is to look at the KL divergence between the two bernoulli distributions Pi = Bernoulli(yi) and Qi = Bernoulli(pˆi). Use this to write down an appropriate minimization for the logistic regression problem.

(f) In logistic regression (and other binary classification problems), we commonly use the cross-entropy loss, defined by

LXE(a,b) = −alnb − (1 − a)ln(1 − b).

Page 2

Using your result from the previous part, discuss why the XE loss is a good choice, and draw a connection between the statistical and optimisation views of logistic regression.

Question 4. Numerically solving the logistic regression problem

In the previous problem, we show that in order to solve the logistic regression problem, we must solve the following optimisation:

wˆ = argminL(w) w

= argmin,

w

where

.

Unfortunately in this case, we cannot solve for wˆ in closed form. In other words, we cannot simply take derivatives, equate to zero and solve to get a nice solution as in the linear regression case. Instead, we must rely on numerical techniques such as gradient descent. In this question, we will work through and derive the gradient descent updates for the logistic regression problem.

(a) We will need to take derivatives in order to do any form of gradient descent. Show that

σ0(z) = σ(z)(1 − σ(z)).

Then use this result to show that

,

where pi = σ(wTxi).

(b) Use the previous result to show that

.

(c) Using the results of the previous parts, compute

and write down the gradient descent update for w with step size η.

(d) A convex function does not have any local minima, and so we are guaranteed to converge to a global minimum when doing gradient descent on a convex function, regardless of our initialisation w(0). Prove that L(w) is convex.

Page 3
