# **Flight Price Predictor** ✈️

This project predicts U.S. domestic flight fares based on route features like distance, passenger volume, and market share. It includes both:

- A **Jupyter Notebook** with model training, feature analysis, and evaluation
- A **Streamlit web app** that allows users to estimate flight costs interactively

## **Files**
- `Flight_predictor_notebook.ipynb`: Full project analysis
- `Flight_predictor.py`: Streamlit app interface for user inputs and prediction
- `model.pkl`: Trained model file
- `train_model.py`: Random Forest Regressor model for full model
- `domestic.csv`: Dataset containing ~100K U.S. domestic flight routes and fares

## **Features**
- Random Forest Regressor models (Full vs. Minimal)
- Feature importance visualization
- Predicted vs. actual fare plotting
- Performance metrics (R², MSE, RMSE)
- Streamlit-based interactive interface and optional visual insights

## **Model**
| Model | Features Used	| Purpose |
| ----- | ------------- | ------- |
| Full | Distance, quarter, passengers, market share, fares | Maximize Accuracy |
| Minimal |	Distance + quarter only	| Lightweight, low-data fallback |

## Performance
*Table based on most recent training session. See notebook for exact metrics.*
| Model         | R² Score | RMSE  | MSE     |
|---------------|----------|-------|---------|
| Full Model    | 0.991    | 5.90  | 34.82   |
| Minimal Model | 0.564    | 41.17 | 1694.88 |

## **Future Work**
- Expand dataset to include additional domestic data and add international flights
- Integrate real-time airfare APIs for live prediction
- Add more user personalization (holiday season, day-of-week)
- Improve model robustness with hyperparamter tuning

## Dataset
- Source: https://www.kaggle.com/datasets/amitzala/us-airline-flight-routes-and-fares
- Records: ~100K domestic flight routes
- Target Variable: `fare` (average ticket price)

## How to Run
#### Notebook
1. Open `Flight_predictor.ipynb`
2. Ensure `domestic.csv` lives in same directory
3. Run all cells to see analysis and model examples

#### Streamlit App
1. Run "streamlit run Flight_predictor.py" in terminal
2. Follow app UI and toggle features and optional visual insights