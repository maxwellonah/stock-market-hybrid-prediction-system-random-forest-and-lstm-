import os
import subprocess
import sys

def run_command(command):
    """Run a shell command and print output"""
    print(f"Running: {command}")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True
    )
    
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def main():
    """Initialize database and run Django server"""
    print("Initializing Hybrid Stock Prediction System...")
    
    # Make migrations
    if run_command("python manage.py makemigrations") != 0:
        print("Error creating migrations. Exiting.")
        return
    
    # Apply migrations
    if run_command("python manage.py migrate") != 0:
        print("Error applying migrations. Exiting.")
        return
    
    # Create superuser if needed
    try:
        from django.contrib.auth.models import User
        if not User.objects.filter(username='admin').exists():
            print("Creating admin superuser...")
            from django.contrib.auth.models import User
            User.objects.create_superuser('admin', 'admin@example.com', 'admin')
            print("Superuser created successfully.")
    except Exception as e:
        print(f"Error creating superuser: {str(e)}")
    
    # Run server
    print("\nStarting Django server...")
    print("The application will be available at http://127.0.0.1:8000/")
    print("Admin interface available at http://127.0.0.1:8000/admin/ (username: admin, password: admin)")
    print("Press Ctrl+C to stop the server")
    run_command("python manage.py runserver")

if __name__ == "__main__":
    main()
