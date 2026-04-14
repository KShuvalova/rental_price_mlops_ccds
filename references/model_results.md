# Model Results

## RandomForestRegressor

### Validation
- mae_log: 0.3114387494819858
- rmse_log: 0.43278139037283087
- r2_log: 0.6096890375880852
- mae_price: 55.88384142886965
- rmse_price: 211.53112206779787
- r2_price: 0.16315302081906424

### Test
- mae_log: 0.3046926300761344
- rmse_log: 0.4349523231003128
- r2_log: 0.609708564032948
- mae_price: 58.651600974244374
- rmse_price: 256.6607152172681
- r2_price: 0.12301101098233858

## CatBoostRegressor

### Validation
- mae_log: 0.306419674274405
- rmse_log: 0.4240827414779343
- r2_log: 0.6252213944120355
- mae_price: 55.11381683362576
- rmse_price: 210.07738346792283
- r2_price: 0.17461588620554658

## Current conclusion
- CatBoost performs better than RandomForest on validation.
- Best current validation metric: r2_log = 0.6252
- RandomForest already has independent test evaluation.
- Next step: evaluate CatBoost on test set.
