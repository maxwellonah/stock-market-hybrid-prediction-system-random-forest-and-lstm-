from django.db import models
import os


class StockData(models.Model):
    """Model to store uploaded stock data files"""
    name = models.CharField(max_length=100)
    file = models.FileField(upload_to='stock_data/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    def delete(self, *args, **kwargs):
        # Delete the file when the model instance is deleted
        if self.file:
            if os.path.isfile(self.file.path):
                os.remove(self.file.path)
        super().delete(*args, **kwargs)


class TrainedModel(models.Model):
    """Model to store information about trained ML models"""
    MODEL_TYPES = (
        ('RF', 'Random Forest'),
        ('LSTM', 'LSTM'),
    )
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=4, choices=MODEL_TYPES)
    stock_symbol = models.CharField(max_length=10)
    file_path = models.CharField(max_length=255)
    accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.name} ({self.model_type}) - {self.stock_symbol}"


class Prediction(models.Model):
    """Model to store prediction results"""
    PREDICTION_TYPES = (
        ('daily', 'Daily Prediction'),
        ('monthly', 'Monthly Prediction'),
    )
    
    stock_symbol = models.CharField(max_length=10)
    prediction_type = models.CharField(max_length=10, choices=PREDICTION_TYPES)
    prediction_date = models.DateField()
    predicted_price = models.FloatField()
    actual_price = models.FloatField(null=True, blank=True)
    model = models.ForeignKey(TrainedModel, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.stock_symbol} {self.prediction_type} prediction for {self.prediction_date}"
