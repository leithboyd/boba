

We will define our feature calculation look back windows in event space e.g. trade_bin_eth_usdt_p_{N} could mean the time window from now till the Nth most recent trade on the listing bin_eth_usdt_p.

There are many possible event clocks that can be used e.g. we could use a merged event clock by pooling all the trades from every listing we are using to caulcate our features over then define our window as a lookback over that event stream.

