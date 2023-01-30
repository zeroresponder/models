# Neural Network for ZeroResponder

    - Trained on roughly 1000 patients with the goal of determing the likely hood of a person being at risk for heart disease
    
    - Identified which datapoints could be dropped without significantly impacting accuracy to improve user experience
    
    - Over-fitting was a huge problem with age, as training data only had people aged between 40-60. Dropout layers were used to force the network to draw connectinos between other inputs
