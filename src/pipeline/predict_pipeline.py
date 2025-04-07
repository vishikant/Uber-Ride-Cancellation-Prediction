import pickle
from src.pipeline.custom_data import CustomData

class PredictPipeline:
    def __init__(self, 
                 ride_id, 
                 driver_id, 
                 customer_id, 
                 pickup_location, 
                 dropoff_location, 
                 ride_start_time, 
                 ride_end_time, 
                 cancellation_status, 
                 cancellation_reason, 
                 ride_fare, 
                 day_of_week, 
                 hour_of_day, 
                 peak_hours, 
                 driver_rating, 
                 past_cancellations, 
                 weather_condition, 
                 traffic_condition, 
                 driver_name):
        self.custom_data = CustomData(
            ride_id=ride_id,
            driver_id=driver_id,
            customer_id=customer_id,
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            ride_start_time=ride_start_time,
            ride_end_time=ride_end_time,
            cancellation_status=cancellation_status,
            cancellation_reason=cancellation_reason,
            ride_fare=ride_fare,
            day_of_week=day_of_week,
            hour_of_day=hour_of_day,
            peak_hours=peak_hours,
            driver_rating=driver_rating,
            past_cancellations=past_cancellations,
            weather_condition=weather_condition,
            traffic_condition=traffic_condition,
            driver_name=driver_name
        )

    def predict(self):
        # Convert custom data to a DataFrame
        data = self.custom_data.get_data_as_dataframe()
        
        # Make predictions
        prediction = self.make_prediction(data)
        return prediction

    def make_prediction(self, data):
        # Load the model
        with open('/Users/vaishalikant/Downloads/data science project/Uber-Ride-Cancellation-Prediction/artifacts/model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        
        # Make predictions
        prediction = model.predict(data)
        
        # Return the predictions
        return prediction