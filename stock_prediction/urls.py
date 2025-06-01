from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('fetch-stock-data/', views.fetch_stock_data, name='fetch_stock_data'),
    path('upload-stock-data/', views.upload_stock_data, name='upload_stock_data'),
    path('train-models/', views.train_models, name='train_models'),
    path('make-predictions/', views.make_predictions, name='make_predictions'),
    path('live-prediction/', views.live_prediction, name='live_prediction'),
]
