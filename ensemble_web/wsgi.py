"""
WSGI config for ensemble_web project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ensemble_web.settings')

application = get_wsgi_application()
