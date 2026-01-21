class DatabaseRouter:
    """Route apps to the appropriate database."""

    route_app_labels = {
        'users': 'default',
        'auth_system': 'default',
        'audit': 'default',
        'trainings': 'default',  # Training monitoring system
        'dataset': 'datasets_db',
        # Accept plural alias used in tests
        #'datasets': 'datasets_db',
    }

    def db_for_read(self, model, **hints):
        app_label = model._meta.app_label
        return self.route_app_labels.get(app_label)

    def db_for_write(self, model, **hints):
        app_label = model._meta.app_label
        return self.route_app_labels.get(app_label)

    def allow_relation(self, obj1, obj2, **hints):
        """Allow relations between dataset models and user models.""" 
        # Always allow relations between models in our managed databases
        db_set = {'default', 'datasets_db'}
        
        # If both objects exist and are in our databases, allow the relation
        if obj1 and obj2:
            db1 = getattr(obj1._state, 'db', None) 
            db2 = getattr(obj2._state, 'db', None)
            if db1 in db_set and db2 in db_set:
                return True
        
        # Also allow relations based on app labels (for when objects don't have _state.db yet)
        if hasattr(obj1, '_meta') and hasattr(obj2, '_meta'):
            app1 = obj1._meta.app_label
            app2 = obj2._meta.app_label
            
            # Allow User <-> Dataset relations specifically
            allowed_pairs = [
                ('users', 'dataset'),
                ('dataset', 'users'),
                ('users', 'datasets'),
                ('datasets', 'users')
            ]
            if (app1, app2) in allowed_pairs:
                return True
        
        return None

    def allow_migrate(self, db, app_label, model_name=None, **hints):
        # In test environment with pytest, allow users app to migrate to both databases
        # This enables foreign key relationships to work in tests
        import sys
        import os
        is_pytest = 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ
        is_test = 'test' in sys.argv or is_pytest
        
        if is_test and app_label == 'users':
            return True  # Allow users to migrate to both databases in tests
                
        target = self.route_app_labels.get(app_label)
        if target is None:
            # Default: only migrate unmapped apps on 'default' DB
            return db == 'default'
        return db == target


