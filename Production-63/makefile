.PHONY: hypertune train prediction

hypertune: param/10YR_best_param.csv param/DXY_best_param.csv param/SPX_best_param.csv param/GOLD_best_param.csv param/WTI_best_param.csv
train: models/10YR.h5 models/DXY.h5 models/SPX.h5 models/GOLD.h5 models/WTI.h5
prediction: prediction/10YR_63Dprediction.csv prediction/DXY_63Dprediction.csv prediction/SPX_63Dprediction.csv prediction/GOLD_63Dprediction.csv prediction/WTI_63Dprediction.csv

# Make prediciton and save to csv
prediction/%_63Dprediction.csv: prediction.py train/%_training_data.csv validate/%_validation_data.csv models/%.h5
	python $^ prediction/$*_63Dprediction.csv
	
# Take parameters suggested by the hypertuner and train model 
models/%.h5: train.py train/%_training_data.csv param/%_best_param.csv param/%_best_param.csv
	python $^ models/$*.h5

# Hypertune each dataset for all asset classes
param/%_best_param.csv: hypertune.py train/%_training_data.csv
	python $^ param/$*_best_param.csv

# Clean the data and models 
clean:
	rm -rf data/*.csv
	rm -rf train/*.csv
	rm -rf validate/*.csv
	rm -rf param/*.csv
	rm -rf models/*.h5
	rm -rf prediction/*.csv

clear:
	rm -rf trials/*
