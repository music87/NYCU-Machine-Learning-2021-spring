### Speedup Python ###

https://towardsdatascience.com/one-simple-trick-for-speeding-up-your-python-code-with-numpy-1afc846db418

> Vectorization involves expressing mathematical operations, such as the multiplication we’re using here, as occurring on entire arrays rather than their individual elements (as in our for-loop).
>
> With vectorization, the underlying code is parallelized such that the operation can be run on multiply array elements at once, rather than looping through them one at a time. As long as the operation you are applying does not rely on any other array elements, i.e a “state”, then vectorization will give you some good speed ups.
>
> Looping over Python arrays, lists, or dictionaries, can be slow. Thus, vectorized operations in Numpy are mapped to highly optimized C code, making them much faster than their standard Python counterparts.



### EM algorithm ###

**Load training data**

> $$
> X=\begin{bmatrix} xData_0,\text{0}^{th} \text{ train image} \\ xData_1,\text{1}^{st} \text{ train image} \\ \vdots \\ xData_{59999},\text{59999}^{th} \text{ train image} \end{bmatrix}=\begin{bmatrix} xDataPixel_{0,0}&xDataPixel_{0,1}&\dots&xDataPixel_{0,783} \\ xDataPixel_{1,0}&xDataPixel_{1,1}&\dots&xDataPixel_{1,783} \\ \vdots&\vdots&\vdots&\vdots \\ xDataPixel_{59999,0}&xDataPixel_{59999,1}&\dots&xDataPixel_{59999,783} \end{bmatrix}_{60000*784}
> $$
>
> 

**Initialize model parameter** $\theta$

> $$
> \begin{align}
> \theta^{(0)} &= \{\lambda^{(0)},P^{(0)}=pLabel_0^{(0)},pLabel_1^{(0)},...,pLabel_9^{(0)}\}\\
> 
> \lambda &= \begin{bmatrix} \lambda Label_0,\text{chance to pick label 0} \\ \lambda Label_1,\text{chance to pick label 1} \\ \vdots \\ \lambda Label_9,\text{chance to pick label 9} \end{bmatrix}_{10*1}\\\\
> 
> P &= \begin{bmatrix} pLabel_0,\text{chance of number of head for label 0(binomial)} \\ pLabel_1,\text{chance of number of head for label 1(binomial)} \\ \vdots \\ pLabel_9,\text{chance of number of head for label 9(binomial)} \end{bmatrix} \\
> &= \begin{bmatrix} pLabelPixel_{0,0}&pLabelPixel_{0,1}&\dots&pLabelPixel_{0,783} \\ pLabelPixel_{1,0}&pLabelPixel_{1,1}&\dots&pLabelPixel_{1,783} \\ \vdots&\vdots&\vdots&\vdots \\ pLabelPixel_{9,0}&pLabelPixel_{9,1}&\dots&pLabelPixel_{9,783} \end{bmatrix}_{10*784},\\
> &\text{where }pLabelPixel_{label,pixel} \text{ means chance of head for Label label on the Pixel pixel(bernoulli)} \\\\ 
> \end{align}
> $$

while (k-th model parameter $\theta^{(k)}$ doesn't converge) {

**E step : find responsibility $\omega$**

> $$
> \begin{align}
> \omega &= \begin{bmatrix} \omega_0:=P(Label=0,Data=xData_{\{i|i \in [0,59999]\}}|\theta^{(k)}=\{\lambda^{(k)},P^{(k)}=p_0^{(k)},p_1^{(k)},...,p_9^{(k)}\}) \\ \omega_1:=P(Label=1,Data=xData_{\{i|i \in [0,59999]\}}|\theta^{(k)}=\{\lambda^{(k)},P^{(k)}=p_0^{(k)},p_1^{(k)},...,p_9^{(k)}\}) \\ \vdots \\ \omega_9:=P(Label=9,Data=xData_{\{i|i \in [0,59999]\}}|\theta^{(k)}=\{\lambda^{(k)},P^{(k)}=p_0^{(k)},p_1^{(k)},...,p_9^{(k)}\}) \end{bmatrix} \\\\
> 
> \text{where } &P(Label=label,Data=xData_{\{i|i \in [0,59999]\}}|\theta^{(k)} = \{\lambda^{(k)},P^{(k)}=p_0^{(k)},p_1^{(k)},...,p_9^{(k)}\})
> = \lambda Label_{label}^{(k)}(pLabel_{label}^{(k)})^{xData_i}(1-pLabel_{label}^{(k)})^{(1-xData_i)}\\
> &= \begin{bmatrix} \lambda Label_{0}^{(k)}(pLabel_{0}^{(k)})^{xData_i}(1-pLabel_{0}^{(k)})^{(1-xData_i)}, i \in[0,59999] \\ \lambda Label_{1}^{(k)}(pLabel_{1}^{(k)})^{xData_i}(1-pLabel_{1}^{(k)})^{(1-xData_i)}, i \in[0,59999] \\ \vdots \\ \lambda Label_{9}^{(k)}(pLabel_{9}^{(k)})^{xData_i}(1-pLabel_{9}^{(k)})^{(1-xData_i)}, i \in[0,59999] \end{bmatrix}\\\\
> 
> \text{because }&\text{each pixel is indepndent to others, }
> \lambda Label_{label}^{(k)}(pLabel_{label}^{(k)})^{xData_i}(1-pLabel_{label}^{(k)})^{(1-xData_i)}=\lambda Label_{label}^{(k)}\prod_{pixel=0}^{783} (pLabelPixel_{label,pixel}^{(k)})^{xDataPixel_{i,pixel}} (1-pLabelPixel_{label,pixel}^{(k)})^{(1-xDataPixel_{i,pixel})}\\
> &= \begin{bmatrix} \lambda Label_{0}^{(k)} \prod_{pixel=0}^{783} (pLabelPixel_{0,pixel}^{(k)})^{xDataPixel_{i,pixel}} (1-pLabelPixel_{0,pixel}^{(k)})^{(1-xDataPixel_{i,pixel})}, i \in[0,59999] \\ \lambda Label_{1}^{(k)} \prod_{pixel=0}^{783} (pLabelPixel_{1,pixel}^{(k)})^{xDataPixel_{i,pixel}} (1-pLabelPixel_{1,pixel}^{(k)})^{(1-xDataPixel_{i,pixel})}, i \in[0,59999] \\ \vdots \\ \lambda Label_{9}^{(k)} \prod_{pixel=0}^{783} (pLabelPixel_{9,pixel}^{(k)})^{xDataPixel_{i,pixel}} (1-pLabelPixel_{9,pixel}^{(k)})^{(1-xDataPixel_{i,pixel})}, i \in[0,59999] \end{bmatrix}\\\\
> 
> \text{because }&\text{just producting all of the 28*28 pixel to get an image will lead to underflow } 
> (e^{-784*3}~e^{-784*2}) \text{, we need to use log operation}\\
> &log( \lambda Label_{label}^{(k)}  \prod_{pixel=0}^{783} (pLabelPixel_{label,pixel}^{(k)})^{xDataPixel_{i,pixel}} (1-pLabelPixel_{label,pixel}^{(k)})^{(1-xDataPixel_{i,pixel})})\\
> &=log(\lambda Label_{label}^{(k)})+\sum_{pixel=0}^{783} log((pLabelPixel_{label,pixel}^{(k)})^{xDataPixel_{i,pixel}} (1-pLabelPixel_{label,pixel}^{(k)})^{(1-xDataPixel_{i,pixel})})\\
> &=log(\lambda Label_{label}^{(k)})+\sum_{pixel=0}^{783}xDataPixel_{i,pixel}*log(pLabelPixel_{label,pixel}^{(k)})+(1-xDataPixel_{i,pixel})*log(1-pLabelPixel_{label,pixel}^{(k)})\\
> log(w)
> &= \begin{bmatrix} log(\lambda Label_{0}^{(k)}) + \sum_{pixel=0}^{783} xDataPixel_{i,pixel}*log(pLabelPixel_{0,pixel}^{(k)})+(1-xDataPixel_{i,pixel})*log(1-pLabelPixel_{0,pixel}^{(k)}), i \in[0,59999] \\ log(\lambda Label_{1}^{(k)}) + \sum_{pixel=0}^{783} xDataPixel_{i,pixel}*log(pLabelPixel_{1,pixel}^{(k)})+(1-xDataPixel_{i,pixel})*log(1-pLabelPixel_{1,pixel}^{(k)}), i \in[0,59999] \\ \vdots \\ log(\lambda Label_{9}^{(k)}) + \sum_{pixel=0}^{783}xDataPixel_{i,pixel}*log(pLabelPixel_{9,pixel}^{(k)})+(1-xDataPixel_{i,pixel})*log(1-pLabelPixel_{9,pixel}^{(k)}), i \in[0,59999] \end{bmatrix}\\\\
> 
> \text{we }& \text{expand } i \in [0,59999] \text{ in row} \\
> &=\begin{bmatrix} \omega LabelData_{0,0}&\omega LabelData_{0,1}&\dots&\omega LabelData_{0,59999} \\ \omega LabelData_{1,0}&\omega LabelData_{1,1}&\dots&\omega LabelData_{1,59999} \\ \vdots&\vdots&\vdots&\vdots \\ \omega LabelData_{9,0}&\omega LabelData_{9,1}&\dots&\omega LabelData_{9,59999} \end{bmatrix}_{10*60000}\\
> \text{where }& \omega LabelData_{label,data} \text{ the log probability of Data data under Label label}
> \\\\
> \end{align}
> $$
>
> now we normalize $\omega$ to get responsibility $\hat \omega$
> $$
> \begin{align}
> \hat \omega&=\begin{bmatrix} \hat \omega LabelData_{0,0}&\hat \omega LabelData_{0,1}&\dots&\hat \omega LabelData_{0,59999} \\ \hat \omega LabelData_{1,0}&\hat \omega LabelData_{1,1}&\dots&\hat \omega LabelData_{1,59999} \\ \vdots&\vdots&\vdots&\vdots \\ \hat \omega LabelData_{9,0}&\hat \omega LabelData_{9,1}&\dots&\hat \omega LabelData_{9,59999} \end{bmatrix}_{10*60000} \\
> \text{where }& \hat \omega LabelData_{label,data}=\frac{\omega LabelData_{label,data}}{\sum_{label}\omega LabelData_{label,data}}=\frac{\omega LabelData_{label,data}}{np.sum(\omega,axis=0)}
> \end{align}
> $$
> note that 
>
> before we normalize $\omega$ to responsibility $\hat \omega$: 
>
> when $\omega LabelData$ is higher, the probability of the image under the label is "higher"
>
> after we normalize $\omega$ to responsibility $\hat \omega$: 
>
> when $\hat \omega LabelData$ is higher, the probability of the image under the label is "lower"
>
> because $\sum_{label}\omega LabelData_{label,data}$ is negative
>
> 但後來就沒有用 log ，因為不知道為什麼用log搭配negative normalization一直出錯... 
>
> https://stackoverflow.com/questions/51464638/what-is-the-proper-way-to-normalize-negative-values/51464692

**M step : update model paramater $\theta^{(k+1)}=\lambda, P$**

> $$
> \begin{align}
> \lambda &= \begin{bmatrix} \lambda Label_0 \\ \lambda Label_1 \\ \vdots \\ \lambda Label_9 \end{bmatrix}
> = \begin{bmatrix} \frac{\sum_{data}\hat \omega LabelData_{0,data}}{nData(=60000)} \\ \frac{\sum_{data}\hat \omega LabelData_{1,data}}{nData(=60000)} \\ \vdots \\ \frac{\sum_{data}\hat \omega LabelData_{9,data}}{nData(=60000)} \end{bmatrix}_{10*1} \\\\
> 
> \text{becasue }&\text{P will sum up each probability of data, so we normalize all the data}\\
> \hat \omega LabelD&ata_{label,data}=\frac{\hat \omega LabelData_{label,data}}{\sum_{data}\hat \omega LabelData_{label,data}}=\frac{\hat \omega LabelData_{label,data}}{np.sum(\hat \omega,axis=1)}\\
> 
> P&=\begin{bmatrix} pLabel_0 \\ pLabel_1 \\ \vdots \\ pLabel_9 \end{bmatrix} \\
> &=\begin{bmatrix} \sum_{data}\hat \omega LabelData_{0,data}xData_{data} \\ \sum_{data}\hat \omega LabelData_{1,data}xData_{data} \\ \vdots \\ \sum_{data}\hat \omega LabelData_{9,data}xData_{data} \end{bmatrix}\\
> &=\begin{bmatrix} \sum_{data}\hat \omega LabelData_{0,data}xDataPixel_{data,0} & \sum_{data}\hat \omega LabelData_{0,data}xDataPixel_{data,1} & \dots & \sum_{data}\hat \omega LabelData_{0,data}xDataPixel_{data,783} \\ \sum_{data}\hat \omega LabelData_{1,data}xDataPixel_{data,0} & \sum_{data}\hat \omega LabelData_{1,data}xDataPixel_{data,1} & \dots & \sum_{data}\hat \omega LabelData_{1,data}xDataPixel_{data,783} \\ \vdots \\ \sum_{data}\hat \omega LabelData_{9,data}xDataPixel_{data,0} & \sum_{data}\hat \omega LabelData_{9,data}xDataPixel_{data,1} & \dots & \sum_{data}\hat \omega LabelData_{9,data}xDataPixel_{data,783} \end{bmatrix}_{10*784}
> 
> \end{align}
> $$

}



$$
\begin{align}&

\end{align}
$$

注意最後求出來的 logW是沒有正確label的，要搭配training label去做一個matching table，才能知道那些logW背後真正的label是什麼