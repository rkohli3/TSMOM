# Disclaimer

- None of the contents constitute an offer to sell, a solicitation to buy, or a recommendation or endorsement for any security or strategy, nor does it constitute an offer to provide investment advisory services
- Past performance is no indicator of future performance
- Provided for informational purposes only
- All investments involve risk, including loss of principal

For access to the jupyter notebook with interactivity containing the CTA strategy, please click on this link: http://nbviewer.jupyter.org/github/rkohli3/TSMOM/blob/master/Momentum.ipynb

To access the jupyter notebook with interactivity for Indian Equities, please click on the link below:
http://nbviewer.jupyter.org/github/rkohli3/TSMOM/blob/master/MomentumIndia.ipynb

# Time Series Momentum (TSMOM)
<br>
<br>

## <center> Summary </center>

The following code blocks are based on the Time Series Momentum strategy, TSMOM, as illustrated in the 2011, Moskowitz, Ooi and Pedersen paper.

1. **What is TSMOM and how is it different from Momentum mentioned by Jegadeesha and Titman, 2001?**<br>
TSMOM is a smarket anomaly that captures strong positive predicitibility from a security's own past returns. That is, if the past returns are positive, they will continue to be positive and vice versa. <br> <br>
This is related to, however, different from "momentum" in finance literature, which refers to the *cross-sectional* performance comparison of a security from its peers, where securities that have outperfermoed their peers in the past three to twelve months, will continue to do so on an average. TSMOM, focuses primarily on the security's *own* returns

2. **Why replicate?**<br><br>
I do this in order to backtest results for the highly liquid ETF securities in the same asset class as the paper mentions. These asset classes are:<b>
    1. Bonds
    2. Equity Index
    3. Currencies
    4. Commodities


<br>
<br><br><br>
Let's get on with it, shall we?!?!
