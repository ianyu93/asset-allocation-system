python clean.py
python feature-engineering.py "SPX Close" "SPX Today" "SPX"
python feature-engineering.py "DXY Close" "DXY Today" "DXY"
python feature-engineering.py "WTI price" "WTI Today" "WTI"
python feature-engineering.py "GOLD price" "GOLD Today" "GOLD"
python feature-engineering.py "10YR yields" "10YR yields Today" "10YR"
make hypertune
make train
make prediction
cp prediction/*.csv ../../app/prediction