from django import forms
from .models import StockData

class StockDataUploadForm(forms.ModelForm):
    """Form for uploading stock data files"""
    class Meta:
        model = StockData
        fields = ['name', 'file']
        
    def clean_file(self):
        file = self.cleaned_data.get('file')
        if file:
            if not file.name.endswith('.csv'):
                raise forms.ValidationError("Only CSV files are allowed.")
        return file


class StockSymbolForm(forms.Form):
    """Form for selecting stock symbols from Polygon API"""
    STOCK_CHOICES = [
        ('GOOGL', 'Google (GOOGL)'),
        ('AAPL', 'Apple (AAPL)'),
        ('MSFT', 'Microsoft (MSFT)'),
        ('AMZN', 'Amazon (AMZN)'),
    ]
    
    stock_symbol = forms.ChoiceField(
        choices=STOCK_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    
    start_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        required=False
    )
    
    end_date = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        required=False
    )
