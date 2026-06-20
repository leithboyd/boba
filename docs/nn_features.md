

We will define our feature calculation look back windows in event space e.g. trade_bin_eth_usdt_p_{N} could mean the time window from now till the Nth most recent trade on the listing bin_eth_usdt_p.

There are many possible event clocks that can be used e.g. we could use a merged event clock by pooling all the trades from every listing we are using to caulcate our features over then define our window as a lookback over that event stream.

The research tends to point towards trades as a good event clock. Our features will use the merged trade event clock.

At any point in time let the window {N}t be defined as the wall clock time window [now, time of the Nth most recent trade in the merged trade stream]

The definiton of the clock is the collection of listings that go into the feature set. let L be that set of listing and T be a listing in L that is the target listing we are tring to predict the pice on.

We are going to develop and validate our feature set and neural network using the following L and T.

L: {bin_eth_usdt_p, byb_eth_usdt_p, okx_eth_usdt_p}
T: byb_eth_usdt_p

An important note to consider reguarding features is the practicality of implimentending the feature efficently in a high performance streaming context. For example if a features requires a buffer of the last 100000 events that might not be very practical.

During feature development and exploration we should always only use features that have a practical implimentation. Additionally each feature we impliment we will also impliment a streaming event based version that could be used in a production context i.e. as class that is fed events from all the exchanges and internally calculates and maintains state nessicary for the feature to be cauclated and maintained. We should standarise this interface. 

IFeatureBuilder<TFeatures> {
    on_trade(listing, trade_event_data)
    on_front_levels(listing, front_levels_event_data)
    on_funding(listing, funding_event_data)
    emit() -> TFeatures
}

This implimentation/interface allows multiple event to be ingestested before emitting a feature set to predict on. This is important to allow all event data to be invested before emiting a set of features for prediction.


The feature set is going to rely heavily on moving averages. So its important that we iron how how this is going to be done now before any feature development. instead of a basic average over a window we are going to use EMA's. The merged trade clock makes this a little complicated. If the clock was bin_trades and we wanted a evet trabed ema of trade volume over the last N events thats very simple. We just got an EMA with a fixed decay calculated using the event window size and thats that. the challange is when the event clock and the events your are aggregating using the ema are different. For example byb trade volume over the last N bin trade events. In this vase the implimentation should work as follows.






