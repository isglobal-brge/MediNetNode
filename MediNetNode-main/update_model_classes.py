"""
Script para actualizar el output_schema de un modelo con las etiquetas de clase.
Ejecutar desde MediNetNode-main/
"""
import os
import sys

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medinet.settings')
import django
django.setup()

from inference.models import DeployedModel


def main():
    # Get the latest approved model
    model = DeployedModel.objects.filter(status='approved').order_by('-created_at').first()

    if not model:
        print("No approved models found!")
        return

    print(f"=" * 60)
    print(f"Model: {model.name} v{model.version}")
    print(f"Domain: {model.domain}")
    print(f"=" * 60)

    print("\nCurrent output_schema:")
    print(model.output_schema)

    # Update output_schema with class labels
    output_schema = model.output_schema or {}

    # Set type to classification if not set
    if 'type' not in output_schema:
        output_schema['type'] = 'classification'

    # Add class labels
    # 0 = Control, 1 = Case (based on your model)
    output_schema['classes'] = {
        '0': 'Control',
        '1': 'Case'
    }

    # Save
    model.output_schema = output_schema
    model.save(update_fields=['output_schema'])

    print("\nUpdated output_schema:")
    print(model.output_schema)
    print("\nModel updated successfully!")
    print("\nYou can customize the class names by editing the model in the admin panel")
    print("or by modifying this script.")


if __name__ == '__main__':
    main()
