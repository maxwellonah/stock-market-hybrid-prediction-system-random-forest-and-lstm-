// Main JavaScript for Hybrid Stock Prediction System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
    
    // File upload preview
    const fileInput = document.getElementById('id_file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0]?.name || 'No file selected';
            const fileLabel = document.querySelector('.custom-file-label');
            if (fileLabel) {
                fileLabel.textContent = fileName;
            }
        });
    }
    
    // Training status updates
    const trainingForm = document.querySelector('form[action*="train-models"]');
    if (trainingForm) {
        trainingForm.addEventListener('submit', function() {
            const rfStatus = document.querySelector('.rf-status');
            const lstmStatus = document.querySelector('.lstm-status');
            
            if (rfStatus) {
                rfStatus.innerHTML = '<div class="loading-spinner"></div> Training...';
                rfStatus.classList.add('training-status', 'running');
            }
            
            if (lstmStatus) {
                lstmStatus.innerHTML = '<div class="loading-spinner"></div> Waiting...';
                lstmStatus.classList.add('training-status', 'running');
            }
        });
    }
    
    // Prediction form
    const predictionForm = document.querySelector('form[action*="make-predictions"]');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function() {
            const predictionBtn = this.querySelector('button[type="submit"]');
            if (predictionBtn) {
                predictionBtn.innerHTML = '<div class="loading-spinner"></div> Generating Predictions...';
                predictionBtn.disabled = true;
            }
        });
    }
    
    // Date range validation for API fetch
    const startDateInput = document.getElementById('id_start_date');
    const endDateInput = document.getElementById('id_end_date');
    
    if (startDateInput && endDateInput) {
        endDateInput.addEventListener('change', function() {
            const startDate = new Date(startDateInput.value);
            const endDate = new Date(endDateInput.value);
            
            if (endDate < startDate) {
                alert('End date cannot be earlier than start date');
                endDateInput.value = '';
            }
        });
        
        startDateInput.addEventListener('change', function() {
            const startDate = new Date(startDateInput.value);
            const endDate = new Date(endDateInput.value);
            
            if (endDateInput.value && endDate < startDate) {
                alert('Start date cannot be later than end date');
                startDateInput.value = '';
            }
        });
    }
    
    // Toggle between prediction tabs
    const predictionTabs = document.querySelectorAll('[data-bs-toggle="tab"]');
    predictionTabs.forEach(tab => {
        tab.addEventListener('shown.bs.tab', function(event) {
            // Trigger window resize to make sure Plotly charts render correctly
            window.dispatchEvent(new Event('resize'));
        });
    });
});
