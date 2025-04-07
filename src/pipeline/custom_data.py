import pandas as pd

class CustomData:
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
        self.ride_id = ride_id
        self.driver_id = driver_id
        self.customer_id = customer_id
        self.pickup_location = pickup_location
        self.dropoff_location = dropoff_location
        self.ride_start_time = ride_start_time
        self.ride_end_time = ride_end_time
        self.cancellation_status = cancellation_status
        self.cancellation_reason = cancellation_reason
        self.ride_fare = ride_fare
        self.day_of_week = day_of_week
        self.hour_of_day = hour_of_day
        self.peak_hours = peak_hours
        self.driver_rating = driver_rating
        self.past_cancellations = past_cancellations
        self.weather_condition = weather_condition
        self.traffic_condition = traffic_condition
        self.driver_name = driver_name

    def get_data_as_dataframe(self):
        custom_data_input_dict = {
            'Ride_ID': [self.ride_id],
            'Driver_ID': [self.driver_id],
            'Customer_ID': [self.customer_id],
            'Pickup_Location': [self.pickup_location],
            'Dropoff_Location': [self.dropoff_location],
            'Ride_Start_Time': [self.ride_start_time],
            'Ride_End_Time': [self.ride_end_time],
            'Cancellation_Status': [self.cancellation_status],
            'Cancellation_Reason': [self.cancellation_reason],
            'Ride_Fare': [self.ride_fare],
            'Day_of_Week': [self.day_of_week],
            'Hour_of_Day': [self.hour_of_day],
            'Peak_Hours': [self.peak_hours],
            'Driver_Rating': [self.driver_rating],
            'Past_Cancellations': [self.past_cancellations],
            'Weather_Condition': [self.weather_condition],
            'Traffic_Condition': [self.traffic_condition],
            'Driver_Name': [self.driver_name]
        }
        return pd.DataFrame(custom_data_input_dict)