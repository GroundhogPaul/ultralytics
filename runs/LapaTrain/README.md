176x1000epoch_MileStone
	trained for 1000 epoch
176xPruneStep2_1500epoch_L1shrinkBN
	trained for 1500 epoch
	for future pruning, so the BN gamma are L1 shrinked during training
176xToBeResumed
    trained based on pretrained "176x1000epoch_MileStone"
	TODO: set the train parameter "pretrained" to "True" or "False", any difference (train faster)?
	TODO: set the learning rate low to the end rate of "176x1000epoch_MileStone", any difference (train faster)?