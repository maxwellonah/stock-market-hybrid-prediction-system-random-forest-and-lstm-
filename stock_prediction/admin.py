from django.contrib import admin
from .models import StockData, TrainedModel, Prediction

@admin.register(StockData)
class StockDataAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at')
    search_fields = ('name',)
    list_filter = ('uploaded_at',)

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'stock_symbol', 'created_at')
    list_filter = ('model_type', 'stock_symbol', 'created_at')
    search_fields = ('name', 'stock_symbol')

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('stock_symbol', 'prediction_type', 'prediction_date', 'predicted_price', 'actual_price')
    list_filter = ('prediction_type', 'stock_symbol', 'prediction_date')
    search_fields = ('stock_symbol',)
