# Data Dictionary

## StockX-Data-Contest-2019-3 2.csv

Source: StockX Data Contest 2019. ~99K transaction-level records.

| Column | Type | Description |
|---|---|---|
| Order Date | date | Transaction date (MM/DD/YY) |
| Brand | string | Shoe brand (Yeezy, Nike, etc.) |
| Sneaker Name | string | Full sneaker model name |
| Sale Price | string | Resale transaction price (e.g. "$1,097") |
| Retail Price | string | Original retail price (e.g. "$220") |
| Release Date | date | Shoe release date |
| Shoe Size | float | US shoe size |
| Buyer Region | string | US state of buyer |

## sneakers2023.csv

Source: Sneaker market data, 2023. ~2000 product-level records.

| Column | Type | Description |
|---|---|---|
| item | string | Sneaker name |
| brand | string | Brand name |
| retail | float | Retail price |
| release | date | Release date |
| lowestAsk | float | Current lowest ask price |
| numberOfAsks | int | Number of active asks |
| salesThisPeriod | int | Sales in current period |
| highestBid | float | Current highest bid |
| numberOfBids | int | Number of active bids |
| annualHigh | float | 52-week high price |
| annualLow | float | 52-week low price |
| volatility | float | Price volatility |
| deadstockSold | int | Total deadstock units sold |
| pricePremium | float | Premium over retail (ratio) |
| averageDeadstockPrice | float | Average deadstock sale price |
| lastSale | float | Most recent sale price |
| changePercentage | float | Recent price change (%) |
