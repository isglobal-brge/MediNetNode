"""
Tests for audit log archival and retention system.
"""
import os
import tempfile
import gzip
from datetime import datetime, timedelta
from pathlib import Path
from io import StringIO

from django.test import TestCase
from django.core.management import call_command
from django.utils import timezone

from audit.models import AuditEvent, DataAccessLog, SecurityIncident
from audit.management.commands.archive_audit_logs import Command
from users.models import CustomUser, Role


class AuditArchivalTests(TestCase):
    """Test audit log archival functionality."""
    
    def setUp(self):
        """Set up test data."""
        role = Role.objects.get(name='RESEARCHER')
        self.user = CustomUser.objects.create_user(
            username='test_user', password='StrongPass123!', role=role
        )
        
        now = timezone.now()

        # Old events (should be archived) - create much older to be safe
        self.old_event = AuditEvent.objects.create(
            user=self.user,
            action='OLD_EVENT',
            category='SYSTEM',
            risk_score=10
        )
        # Update timestamp manually (auto_now_add prevents setting during create)
        AuditEvent.objects.filter(id=self.old_event.id).update(
            timestamp=now - timedelta(days=500)  # Much older than 365 days
        )
        
        # Recent events (should not be archived)
        self.recent_event = AuditEvent.objects.create(
            user=self.user,
            action='RECENT_EVENT',
            category='SYSTEM',
            risk_score=15
        )
        # Update timestamp manually
        AuditEvent.objects.filter(id=self.recent_event.id).update(
            timestamp=now - timedelta(days=10)
        )

        self.old_event.refresh_from_db()
        self.recent_event.refresh_from_db()

        DataAccessLog.objects.create(
            audit_event=self.old_event,
            medical_domain='test_domain',
            records_accessed=100,
            data_sensitivity_level=2
        )
        
        self.incident = SecurityIncident.objects.create(
            incident_type='SUSPICIOUS_ACTIVITY',
            description='Test incident',
            severity=2
        )
        self.incident.related_events.add(self.old_event)

        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test data."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_archive_old_events(self):
        """Test archiving of old events."""
        self.assertEqual(AuditEvent.objects.count(), 2)

        call_command(
            'archive_audit_logs',
            '--days=365',
            '--force',
            verbosity=0
        )

        # Old event should be archived (deleted)
        self.assertFalse(AuditEvent.objects.filter(id=self.old_event.id).exists())

        # Recent event should still exist
        self.assertTrue(AuditEvent.objects.filter(id=self.recent_event.id).exists())

        # Related data access log should be deleted
        self.assertFalse(DataAccessLog.objects.filter(audit_event_id=self.old_event.id).exists())

    def test_category_specific_retention(self):
        """Test category-specific retention policies."""
        auth_event = AuditEvent.objects.create(
            user=self.user,
            action='LOGIN',
            category='AUTH'
        )
        AuditEvent.objects.filter(id=auth_event.id).update(
            timestamp=timezone.now() - timedelta(days=500)  # Older than AUTH retention (365 days)
        )
        
        data_event = AuditEvent.objects.create(
            user=self.user,
            action='DATA_ACCESS',
            category='DATA_ACCESS'
        )
        AuditEvent.objects.filter(id=data_event.id).update(
            timestamp=timezone.now() - timedelta(days=500)  # Not older than DATA_ACCESS retention (2555 days)
        )
        
        call_command(
            'archive_audit_logs',
            '--force',
            verbosity=0
        )

        # AUTH event should be archived
        self.assertFalse(AuditEvent.objects.filter(id=auth_event.id).exists())

        # DATA_ACCESS event should still exist (longer retention)
        self.assertTrue(AuditEvent.objects.filter(id=data_event.id).exists())

    def test_security_events_never_archived(self):
        """Test that SECURITY category events are never archived."""
        security_event = AuditEvent.objects.create(
            user=self.user,
            action='SECURITY_BREACH',
            category='SECURITY',
            severity='SECURITY'
        )
        AuditEvent.objects.filter(id=security_event.id).update(
            timestamp=timezone.now() - timedelta(days=3000)  # Very old
        )

        call_command(
            'archive_audit_logs',
            '--force',
            verbosity=0
        )

        # Security event should never be archived
        self.assertTrue(AuditEvent.objects.filter(id=security_event.id).exists())

    def test_export_before_archive(self):
        """Test exporting events before archiving."""
        call_command(
            'archive_audit_logs',
            '--days=365',
            '--export',
            f'--export-path={self.temp_dir}',
            '--force',
            verbosity=0
        )

        export_files = list(Path(self.temp_dir).glob('*_audit_export_*.csv'))
        self.assertGreater(len(export_files), 0)

        if export_files:
            with open(export_files[0], 'r') as f:
                content = f.read()
                self.assertIn('OLD_EVENT', content)  # Should contain our old event

    def test_compressed_export(self):
        """Test compressed export functionality."""
        call_command(
            'archive_audit_logs',
            '--days=365',
            '--export',
            '--compress',
            f'--export-path={self.temp_dir}',
            '--force',
            verbosity=0
        )

        export_files = list(Path(self.temp_dir).glob('*.csv.gz'))
        self.assertGreater(len(export_files), 0)

        if export_files:
            with gzip.open(export_files[0], 'rt') as f:
                content = f.read()
                self.assertIn('timestamp', content)  # Should contain CSV header

    def test_dry_run_mode(self):
        """Test dry run mode doesn't actually archive."""
        initial_count = AuditEvent.objects.count()

        call_command(
            'archive_audit_logs',
            '--days=365',
            '--dry-run',
            verbosity=0
        )

        # No events should be deleted
        self.assertEqual(AuditEvent.objects.count(), initial_count)

        self.assertTrue(AuditEvent.objects.filter(id=self.old_event.id).exists())

    def test_batch_processing(self):
        """Test batch processing of large datasets."""
        bulk_events = []
        for i in range(50):
            bulk_events.append(AuditEvent(
                user=self.user,
                action=f'BULK_EVENT_{i}',
                category='SYSTEM'
            ))
        created_events = AuditEvent.objects.bulk_create(bulk_events)

        old_timestamp = timezone.now() - timedelta(days=500)
        event_ids = [event.id for event in created_events]
        AuditEvent.objects.filter(id__in=event_ids).update(timestamp=old_timestamp)

        initial_count = AuditEvent.objects.count()

        call_command(
            'archive_audit_logs',
            '--days=365',
            '--batch-size=10',
            '--force',
            verbosity=0
        )

        final_count = AuditEvent.objects.count()
        self.assertLess(final_count, initial_count)

    def test_keep_incidents_option(self):
        """Test keeping security incidents when archiving related events."""
        call_command(
            'archive_audit_logs',
            '--days=365',
            '--keep-incidents',
            '--force',
            verbosity=0
        )

        # Old event should be archived
        self.assertFalse(AuditEvent.objects.filter(id=self.old_event.id).exists())

        # But security incident should still exist
        self.assertTrue(SecurityIncident.objects.filter(id=self.incident.id).exists())

    def test_category_filter(self):
        """Test archiving specific categories only."""
        auth_event = AuditEvent.objects.create(
            user=self.user,
            action='OLD_LOGIN',
            category='AUTH'
        )
        AuditEvent.objects.filter(id=auth_event.id).update(
            timestamp=timezone.now() - timedelta(days=500)
        )
        
        system_event = AuditEvent.objects.create(
            user=self.user,
            action='OLD_SYSTEM',
            category='SYSTEM'
        )
        AuditEvent.objects.filter(id=system_event.id).update(
            timestamp=timezone.now() - timedelta(days=500)
        )
        
        # Archive only AUTH category
        call_command(
            'archive_audit_logs',
            '--category=AUTH',
            '--days=365',
            '--force',
            verbosity=0
        )

        # AUTH event should be archived
        self.assertFalse(AuditEvent.objects.filter(id=auth_event.id).exists())

        # SYSTEM events should still exist
        self.assertTrue(AuditEvent.objects.filter(id=system_event.id).exists())
        self.assertTrue(AuditEvent.objects.filter(id=self.old_event.id).exists())

    def test_command_output(self):
        """Test command output and summary."""
        out = StringIO()

        call_command(
            'archive_audit_logs',
            '--days=365',
            '--force',
            verbosity=2,
            stdout=out
        )
        
        output = out.getvalue()

        # Should contain summary information
        self.assertIn('Summary', output)
        self.assertIn('Archived:', output)

    def test_invalid_category(self):
        """Test handling of invalid category."""
        from django.core.management.base import CommandError

        with self.assertRaises(CommandError):
            call_command(
                'archive_audit_logs',
                '--category=INVALID_CATEGORY',
                '--force',
                verbosity=0
            )

        # All events should still exist (command didn't run)
        self.assertEqual(AuditEvent.objects.count(), 2)

    def test_export_file_naming(self):
        """Test export file naming conventions."""
        call_command(
            'archive_audit_logs',
            '--category=SYSTEM',
            '--days=365',
            '--export',
            f'--export-path={self.temp_dir}',
            '--force',
            verbosity=0
        )

        export_files = list(Path(self.temp_dir).glob('system_audit_export_*.csv'))
        if export_files:
            filename = export_files[0].name
            self.assertIn('system', filename)
            self.assertIn('audit_export', filename)
            # Should have timestamp in filename
            self.assertRegex(filename, r'\d{8}_\d{6}')

    def test_data_access_export_fields(self):
        """Test that data access specific fields are exported."""
        call_command(
            'archive_audit_logs',
            '--days=365',
            '--export',
            f'--export-path={self.temp_dir}',
            '--force',
            verbosity=0
        )

        export_files = list(Path(self.temp_dir).glob('*_audit_export_*.csv'))
        if export_files:
            with open(export_files[0], 'r') as f:
                content = f.read()
                self.assertIn('medical_domain', content)
                self.assertIn('patient_count_accessed', content)
                self.assertIn('data_sensitivity_level', content)
                self.assertIn('test_domain', content)  # Our test data

    def test_retention_policy_defaults(self):
        """Test default retention policy values."""
        command = Command()

        expected_policies = {
            'AUTH': 365,
            'DATA_ACCESS': 2555,  # 7 years for medical compliance
            'USER_MGMT': 1095,    # 3 years
            'DATASET_MGMT': 2555, # 7 years for medical compliance
            'TRAINING': 365,
            'API': 90,
            'SYSTEM': 30,
            'SECURITY': -1,       # Never delete
        }
        
        for category, expected_days in expected_policies.items():
            self.assertEqual(
                command.DEFAULT_RETENTION_POLICIES[category],
                expected_days,
                f"Retention policy for {category} should be {expected_days} days"
            )