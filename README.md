# **Asset Allocation System**

## **Preface**

This README documents the concept behind the project, the thought process in each step, how to setup the automated pipeline, and future plan for development. This project was expanded based on stock market analysis with LSTM and intermarket analysis. For more details about initial work, please refer to this [GitHub repo](https://github.com/ianyu93/stock-market-forecast).

The repository consists of 2 folders:

**Production-21:** This folder consists of all the scripts necessary to make a 21-trading-day forecast.

**Production-63:** This folder consists of all the scripts necessary to make a 63-trading-day forecast.

---

## **Table of Content**

1. [Setup](#Setup)
2. [About the Project](#About-the-Project)
    1. [How is AASystem Different?](#How-is-AASystem-Different?)
    2. [Major Concepts](#Major-Concepts)
    3. [System Flow Breakdown](#System-Flow-Breakdown)
    4. [Limitations](#Limitations)
3. [Data](#Data)
4. [Feature Engineering](#Feature-Engineering)
5. [Architecture](#Architecture)
6. [Future Development](#Future-Development)


---

## **Setup**

In order to setup, ensure that you have `conda` installed.

1. Create a conda environement with `conda create --name myenv`
2. Activate the environment with `conda activate myenv`
3. Install the Python package with `conda install python`
4. Install the requirements.txt with `pip install -r requirements.txt`

More on how to manage your conda environment can be read here [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

For more information on how to schedule tasks with crontab, read [here](https://www.cyberciti.biz/faq/how-do-i-add-jobs-to-cron-under-linux-or-unix-oses/)

[Back to Top](#Table-of-Content)

---
## **About the Project**

When it comes to applying AI in investment, the most obvious application would be predicting the stock market. After all, that is "where the money is made". Predicting the stock market, however, is inherently hard and risky. Even if one can create a model that forecasts relatively well, what is the next step? How much money should one invest? How does one control the risks to protect downside? In the financial world, not only does security selection matter, but diversification through asset allocation also matters. Asset allocation is not confined within the stock market world only, but also across different financial markets, such as currency, commodity, and the bond market. 

The AASystem, my Asset Allocation System, does exactly that in two parts:

1. First, AASystem performs monthly and quarterly forecast of the price movement across different financial markets. 
2. Then, based on the predicted values across different markets, the system allocates weightings across different asset classes that maximizes return while minimizing risks. 

**Essentially, the AASystem tells me how much money should I put in stocks, bonds, gold, oil, and dollars market to maximize my return while minimizing risks next month/quarter**. 

<br>
<div>
<Center>
<img src="https://imgur.com/q4wCxCZ.png" height=225 alt="Two parts of the the AASystem">
</div>
<br>

### **How is AASystem Different?**

The AASystem is different in two ways: How the prediction problem is framed and how allocation weigting is decided.

From the many predictive models on the web, there are two approaches. First is to predict specific stock or the S&P 500 itself with their past data. This approach is confined within the stock world and is turning a blind eye to the macro economy and financial trends, making it hard to forecast for a longer timeframe. The second approach is to transform the Closing price 30 days later into binary classes of price increased or not, create technical indicators as additional features, and predicts based on today's prices and the indicators. This approach often ignores the temporal affect on the market. 

For asset allocation, based on Modern Portfolio Theory, the typical approach is to annualize the returns and volatility based on past history, making an assumption of the future. Instead the AASystem attempts to allocate based on the predicted future, controlling the risks that it foresees.

<br>

### **Major Concepts**

The AASystem is built on two major financial concepts: **Intermarket Analysis and Efficient Frontier**.

[Intermarket Analysis](https://www.investopedia.com/terms/i/intermarketanalysis.asp#:~:text=Intermarket%20analysis%20is%20a%20method,be%20beneficial%20to%20the%20trader.) is a method of analyzing markets by examining the correlation between different asset classes. The pioneer of the discipline, John J. Murphy, asserts that the four major markets - Stocks, Bonds, Commodity, and Currency - are intercorrelated and can be examined in parallel to gain long-term insights.



[Efficient Frontier](https://www.investopedia.com/terms/e/efficientfrontier.asp#:~:text=The%20efficient%20frontier%20is%20the,for%20the%20level%20of%20risk.), the cornerstone of the [Modern Portfolio Theory](https://www.investopedia.com/terms/m/modernportfoliotheory.asp), seeks the a set of optimal asset weightings that offer the highest expected return for a defined level of risk or the lowest risk for a given level of expected return. The weighting of different assets does not only consider the expected returns and volatility of each asset, but also the covariance between the assets. 

<br>

<div>
<Center>
<img src="https://school.stockcharts.com/lib/exe/fetch.php?media=market_analysis:intermarket_analysis:im-1-intermarket.png" height=500 alt="Bond Market and the Stock Market have inverse relationship">
<img src="https://imgur.com/pL1ftkb.png" height=500 alt="Bond Market and the Stock Market have inverse relationship">
</div>
<div>
<Center>

*A: Bond Market and the Stock Market have inverse relationship. B: Efficient Frontier*
</div>

<br>

### **System Flow Breakdown**

The AASystem automated workflow is split into 6 stages, from sourcing and cleaning the data all the way to making prediction.

**Stage 1: Sourcing and Cleaning.** At stage 1, the AASystem sources data from the [Yahoo! Finance](http://yahoo.finance/), [Quandl](https://www.quandl.com/), and [FRED](https://fred.stlouisfed.org/) through APIs. The data collected includes daily values for S&P 500, Treasury Yields for various maturity, WTI Oil Sport Price, Gold Spot Price, the Dollar Index, and an annual growth rate for US Consumer Price Index. The system treats missing values, realign timestamps, and concatenates into a single dataframe appropriate to the model. 

**Stage 2: Feature Engineering.** At stage 2, the AASystem takes in the cleaned dataframe, feature engineers multiple technical indicators to measure relative strength between any two assets, momentum, and volatility. The system then creates a training dataset and a final testing dataset for both monthly and quarterly trading day prediction. 

**Stage 3: Hypertuning.** At stage 3, data of the last 20 years are fed into the hypertuner to search the best predictive parameters in the neural network models for each of the market. This step usually takes 6-8 hours and takes place every year.

**Stage 4: Training.** As the parameters are determined at Stage 3, the AASystem will train each predictive model based on the given parameters. This step usually takes 1 hour and takes place every month.

**Stage 5: Forecasts.** Once the models are trained, the final testing dataset from Stage 2 is fed in to make forecast. The forecast includes 63 trading days before the forecast date that the models have never seen before, and predicting one month/quarter after the forecast date. This step takes place every month, after training.

**Stage 6: Asset Allocation.** Based on the predicted values, the AASystem makes a set of recommendations based on the predicted values. This website is updated daily to compare true market value and forecasted values. 

<div>
<Center>
<img src="https://imgur.com/8s3Ht7h.png" height=180 alt="Bond Market and the Stock Market have inverse relationship">

*Tools Used in the Project*
</div>

<br>

### **Limitations**
The AASystem is currently limited to market-level allocation, controlling investment risks from the macro-level. The actual returns and volatility are still subject to securities selection and allocation in each market. The subsystem that performs securities selection and allocation in each market will continue to be developed.

The evaluation metrics on Optimal Portfolio assume a buy-and-hold scenario from investors. In practice, buy-and-hold strategy rarely lasts only a month. I will also continue to optimize the predictive models to improve longer-term forecasts. The model also does not consider weight cap, such as maximum weighting for a given market should be X%. 

[Back to Top](#Table-of-Content)

---

## **Data**


### **S&P 500 Index: Proxy to Stock Market**
The S&P 500 Index dataset is downloaded through [Yahoo! Finance](https://finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) API. S&P 500 Index is a stock index composed by Standard and Poor's to measure the stock performance of 500 large companies listed on stock exchanges in the United States. It is one of the most followed stock index in the world, and is widely regarded as the best single gauge of large-cap U.S. equities. The dataset consists of daily data of the data. The index is a free-float capitalization-weighted index, that is, companies are weighted in the index in proportion to their market capitalizations. 

### **10Y US Government Bond Yields: Proxy to Bond Market**
The data feed comes from [Quandl](https://www.quandl.com/data/USTREASURY-US-Treasury). The US government bonds are generally seen as representative of quality fixed income asset. Here a wide range of maturity is included, ranging from 1 month to 30 Years. For more details on the bond yield, read [here](https://www.investopedia.com/terms/b/bond-yield.asp).

### **Dollar Index or DXY: Proxy to Currency Market**

The dataset also comes from [Yahoo! Finance](https://finance.yahoo.com/quote/DX-Y.NYB/history?p=DX-Y.NYB) API. Because currency is always trades against another, the exchange rate is always relative. The Dollar Index is a trade-weighted index of the dollar's strength against a basket of US's most significant trading partners. Although the trading weight of the index has not been updated since the creation of the Euro, thus seen obsolete sometimes, it is still the most followed indicator regarding to the dollar's strength.

### **WTI Spot Price: Proxy to Commodity**

West Texas Intermediate oil spot price dataset from [Federal Reserve Economic Data](https://fred.stlouisfed.org/series/DCOILWTICO). WTI historically has been used to track US oil. Although now it has gained global adoptoin and does not discriminate origin, it is still more relavent to the US financial market than other oil index. We are using spot price, meaning the price on the day, instead of futures. While the commodity market is enormous, including cocoa, sugar, and metals, the oil market is one of the most representative and influential commodity in the market. 

### **GOLD: Proxy to Commodity-ish**

Gold data feed from [Quandl](https://www.quandl.com/data/LBMA/GOLD-Gold-Price-London-Fixing), based on [London Fixing Price](https://en.wikipedia.org/wiki/Gold_fixing). While gold is a precious metal commodity, it also has a unique status of reserve currency, making it a hedge against bad times. When the market is pessimistic, money often flows from the stock market to gold. Gold is very representative in cross-market sentiment.

### **CPI: Proxy to Inflation Environment**

One of the most important underlying concept of Intermarket Analysis is that the inflation environment also affects how each of the four major markets is correlated with one another. For example, stocks and bonds used to have a positive correlation back in the 70s. Comparing to decades ago, we now are under what is considered a deflationary environment, where we are more worried about not having enough inflation and may even enter deflation. We source the US Consumer Price Index, one of the most prominent index to measure US inflation, from [Federal Reserve Economic Data](https://fred.stlouisfed.org/series/CPALTT01USA657N) of St. Louis Fed. As we are taking annual inflation rate to represent the long-term macro environment, FRED does not have inflation rate for 2021 yet, which we will set to 2% until more comprehensive forecast is made available.

[Back to Top](#Table-of-Content)

---

## **Feature Engineering**

In my dataset, not only did I include market values across asset classes, but I also manually engineered many indicators that blew out into 200+ features. While I'm not going to list everything here, my goal was to measure 4 things: yield curve, momentum, relative strength, and volatility.

### **Yield Curve**

In the dataset, not only did I include treasury yields for multiple maturity, but I also created features to include yield difference between different pairs of maturity. For example, `10YR yields - 2YR yields` is the difference for the 2Y/10Y pair. Replicating to other pairs, the dataset would see whether the differences are getting steeper or flatter, providing a proxy to the yield curve development.

### **Momentum**

To measure momentum, I used Exponential Moving Average, or EMA. While Simple Moving Average (SMA, the regular moving average) smooths out the overall trend line, EMA adds more weight on more recent values. In other words, EMA not only shows the overall trend, it also incorporates latent momentum. We will be applying EMA on other features to learn momentum of other features.

*Note: I used pandas Exponential Weighted Functions, which works slightly different from the traditional EMA in the finance world, but serves our purpose.*

### **Cross-Market Relative Strength**

To measure cross-market relative strength, I used Relative Strength Line. A concept frequently used by John Murphy in Intermarket Analysis, the RSL is simply the ratio between any two given assets/markets. It measures the relative performance of the two assets. For example, SPX/US10Y RSL measures the relative performance of the stock market comparing to the 10 Year Treasury Bond. An increase of this indicator would signal that the stock market's relative performance to the bond market is getting stronger.

### **Volatility**

To measure volatility, I used the Bollinger Band indicator. The Bollinger Band is comprised of a middle band of SMA, typically a 20-Day Moving Average, an upper band of +2 standard deviation from the middle band, and -2 standard deviation from the middle band. Approximately 90% of the price action would happen within the bands, and since standard deviation is a measure of volitility, when the market becomes volitile, the bands would expand, and vice versa when the market becomes more stable.

### **Timeframe**

For each of the ascribed feature, I also incorporated certain timeframes. For moving averages, we will use 5 (trading days) to measure weekly moving average, 20 (4 weeks of trading days) to represent monthly, and 60 (3 months) to represent a quarter. I also employed market convention of 50-day and 200-day moving averages to measure long-term trends, despite the fact that they do not represent an intuitive timeframe. Contrary to popular belief, the heavy focus of the quantitative perspective in the financial world is actually relatively new (starting around the 80s). Many of the convention are just hard-tested rules that lasted for decades.

[Back to Top](#Table-of-Content)

---

## **Architecture**

In a nutshell, our model architecture looks like this:

<div>
<Center>
<img src="https://imgur.com/mUESztw.png" height=350>
</div>

**Recurrent Neural Netowrk** is a type of neural network that allows us to learn backwards in sequence. It takes in a 3D shape array, input the data as (batch size, sequence, features). Batch size would be the number of data points that we are passing into the neural network at once. If we have a batch size of 1024, then we are passing 1024 trading days into the network at a time. We will have the hypertuner to determine what is the optimal batch size. Sequence length is about the past context, where if we set a sequence length of 10, the model would learn how does the previous 10 sequence affect the current input. We will also leave it to the hypertuner to determine the best sequence length parameter. Features dimension would simply be the number of features we have in our dataset.
<div>
<Center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Recurrent_neural_network_unfold.svg/2880px-Recurrent_neural_network_unfold.svg.png" width=700>
</div>
<div>
<Center>

*image from [Wikepedia](https://en.wikipedia.org/wiki/Recurrent_neural_network#/media/File:Recurrent_neural_network_unfold.svg)*
</div>

**Long Short-Term Memory** is a special type of RNN that learns about the long term default behaviour of the dataset. In effect, this would decompose seasonality, trends, and other potential long-term patterns. 

**Bidirectional** is applied to the LSTM RNN for the purpose of learning the future context as well. Not only does the past affect the stock market today, but also the anticipation of tomorrow's environment would affect today's market. Therefore, we would also need to understand how that anticipation affects today's price. The bidirectional element creates a separate network in the same training session that learns forwards in the sequence instead, so that each time the network is learning both backwards and forward in time. 

**Time Distributed Layer** is a special type of output layer that keeps the training input and output one at a time, keeping the timestamps true. Without the layer, the default behaviour of RNN would learn and output in batches instead. 

[Back to Top](#Table-of-Content)

---

## **Future Development**

The AASystem is only the beginning of what I'm building. After pushing the scripts and documentation onto GitHub, I plan to:

1. Further research and develop data, indicators, and architecture to improve the performance for quarterly forecast, up to June 30, 2021.
2. Develop asset selectors to pick assets in each of the market. Essentially, if 30% is allocated to the stock market, I still need a selector to tell me what to buy with that 30%.
3. Sub-Asset Allocator at the asset level, which is confined within a single market. Say I pick 5 stocks with the 30% of resources allocated, how should I allocate the weighting of each stock?

If you would like to get in touch or see my other projects, [here is my Personal Home Page](https://ianyu93.github.io/homepage/).

[Back to Top](#Table-of-Content)