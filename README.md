# Jupiter

Jupiter is a Django-based project developed by Rapid Labs to provide robust backend services for various applications. The repository is structured for scalability and includes modular components for easy integration.

## Features
- Scalable Django project structure.
- Modular apps for clean code separation.
- RESTful APIs built with Django REST Framework.
- Configurable settings for different environments (e.g., development, production).
- Integrated user authentication and authorization.

## Prerequisites
- Python 3.10+
- Django 5+
- pip (Python package manager)
- Sqlite3 
- Git

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Rapid-Labs-AI/jupiter.git
   cd jupiter
   ```
2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    Configure environment variables:
    ```
4. Apply migrations:

    ```bash
    python manage.py migrate
    ```
5. Run the development server:

    ```bash
    python manage.py runserver
    ```
6. Access the application:
    ```
    Open your browser and navigate to http://127.0.0.1:8000/.
    ```

## Folder Structure
```
jupiter/
├── core/                # Core functionality and settings
├── apps/                # Django apps
├── static/              # Static files
├── templates/           # HTML templates
├── requirements.txt     # Python dependencies
├── manage.py            # Django management script
└── README.md            # Project documentation

```
