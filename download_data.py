import yfinance as yf
import pandas as pd
import numpy as np

Dow_Jones_30 = ['AXP','AMGN','AAPL','BA','CAT','CSCO','CVX','GS','HD','HON','IBM','INTC','JNJ','KO','JPM','MCD','MMM','MRK','MSFT','NKE','PG','TRV','UNH','CRM','VZ','V','WBA','WMT','DIS','DOW']
EuroStoxx_50 = ['ADS.DE','ADYEN.AS','AD.AS','AI.PA','AIR.PA','ALV.DE','ABI.BR','ASML.AS','CS.PA','BAS.DE','BAYN.DE','BBVA.MC','SAN.MC','BMW.DE','BNP.PA','CRG.IR','BN.PA','DB1.DE','DPW.DE','DTE.DE','ENEL.MI','ENI.MI','EL.PA','FLTR.IR','RMS.PA','IBE.MC','ITX.MC','IFX.DE','INGA.AS','ISP.MI','KER.PA','KNEBV.HE','OR.PA','LIN.DE','MC.PA','MBG.DE','MUV2.DE','RI.PA','PHIA.AS','PRX.AS','SAF.PA','SAN.PA','SAP.DE','SU.PA','SIE.DE','STLA.MI','TTE.PA','DG.PA','VOW.DE','VNA.DE']
Hang_Seng = ['0005.HK','0011.HK','0388.HK','0939.HK','1299.HK','1398.HK','2318.HK','2388.HK','2628.HK','3968.HK','3988.HK','0002.HK','0003.HK','0006.HK','1038.HK','2688.HK','0012.HK','0016.HK','0017.HK','0101.HK','0688.HK','0823.HK','0960.HK','1109.HK','1113.HK','1997.HK','2007.HK','6098.HK','0001.HK','0027.HK','0066.HK','0175.HK','0241.HK','0267.HK','0288.HK','0291.HK','0316.HK','0386.HK','0669.HK','0700.HK','0762.HK','0857.HK','0868.HK','0881.HK','0883.HK','0941.HK','0968.HK','0981.HK','0992.HK','1044.HK','1088.HK','1093.HK','1177.HK','1211.HK','1378.HK','1810.HK','1876.HK','1928.HK','1929.HK','2020.HK','2269.HK','2313.HK','2319.HK','2331.HK','2382.HK','3690.HK','3692.HK','6862.HK','9618.HK','9633.HK','9888.HK','9988.HK','9999.HK']

for st in Dow_Jones_30:
    st_ticker = yf.Ticker(st)
    data = st_ticker.history(period="max")
    # check if the stock have history of at least 10 years
    if int(str(data.index[0])[:4]) <= 2013:
        print(st)
        data.to_csv('Dow_Jones_30/'+st+'.csv') 

for st in EuroStoxx_50:
    st_ticker = yf.Ticker(st)
    data = st_ticker.history(period="max")
    # check if the stock have history of at least 10 years
    if int(str(data.index[0])[:4]) <= 2013:
        print(st)
        data.to_csv('EuroStoxx_50/'+st+'.csv')

for st in Hang_Seng:
    st_ticker = yf.Ticker(st)
    data = st_ticker.history(period="max")
    # check if the stock have history of at least 10 years
    if int(str(data.index[0])[:4]) <= 2013:
        print(st)
        data.to_csv('Hang_Seng/'+st+'.csv')
        
print('finished')