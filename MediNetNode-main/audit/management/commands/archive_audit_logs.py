"""
Django management command for archiving old audit logs.

Usage:
    python manage.py archive_audit_logs --days=365 --compress --export
    python manage.py archive_audit_logs --category=TRAINING --days=90
    python manage.py archive_audit_logs --dry-run
"""
import json
import gzip
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List

from django.core.management.base import BaseCommand, CommandError
from django.conf import settings
from django.utils import timezone
from django.db import transaction

from audit.models import AuditEvent, DataAccessLog, SecurityIncident


class Command(BaseCommand):
    help = 'Archive old audit logs with configurable retention policies'
    
    # Default retention policies by event category (in days)
    DEFAULT_RETENTION_POLICIES = {
        'AUTH': 365,           # Authentication events - 1 year
        'DATA_ACCESS': 2555,   # Data access events - 7 years (medical compliance)
        'USER_MGMT': 1095,     # User management - 3 years
        'DATASET_MGMT': 2555,  # Dataset management - 7 years (medical compliance)
        'TRAINING': 365,       # Federated training - 1 year
        'API': 90,             # API access - 3 months
        'SYSTEM': 30,          # System operations - 1 month
        'SECURITY': -1,        # Security events - never delete (use -1)
    }
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            help='Archive events older than N days (overrides category-specific retention)',
        )
        parser.add_argument(
            '--category',
            type=str,
            choices=['AUTH', 'DATA_ACCESS', 'USER_MGMT', 'DATASET_MGMT', 'TRAINING', 'API', 'SYSTEM'],
            help='Archive specific category only',
        )
        parser.add_argument(
            '--compress',
            action='store_true',
            help='Compress archived files with gzip',
        )
        parser.add_argument(
            '--export',
            action='store_true',
            help='Export to CSV before archiving',
        )
        parser.add_argument(
            '--export-path',
            type=str,
            default='audit_exports',
            help='Path for exported files (default: audit_exports)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be archived without actually doing it',
        )
        parser.add_argument(
            '--batch-size',
            type=int,
            default=1000,
            help='Process records in batches (default: 1000)',
        )
        parser.add_argument(
            '--keep-incidents',
            action='store_true',
            help='Keep SecurityIncidents even if related events are archived',
        )
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force archiving without confirmation prompts',
        )

    def handle(self, *args, **options):
        """Main command handler."""
        self.verbosity = options['verbosity']
        self.dry_run = options['dry_run']
        self.compress = options['compress']
        self.export = options['export']
        self.export_path = Path(options['export_path'])
        self.batch_size = options['batch_size']
        self.keep_incidents = options['keep_incidents']
        self.force = options['force']
        
        # Create export directory if needed
        if self.export and not self.dry_run:
            self.export_path.mkdir(exist_ok=True)
        
        # Determine retention policies
        if options['days']:
            # Single retention period for all categories
            retention_policies = {cat: options['days'] for cat in self.DEFAULT_RETENTION_POLICIES.keys()}
            retention_policies['SECURITY'] = -1  # Never delete security events
        else:
            # Use default category-specific retention
            retention_policies = self.DEFAULT_RETENTION_POLICIES.copy()
        
        # Filter by category if specified
        categories_to_process = [options['category']] if options['category'] else list(retention_policies.keys())
        
        self.stdout.write(self.style.SUCCESS('=== Audit Log Archiving Process ==='))
        
        total_archived = 0
        total_exported = 0
        
        for category in categories_to_process:
            retention_days = retention_policies[category]
            
            if retention_days == -1:
                self.stdout.write(f"Skipping {category} (permanent retention policy)")
                continue
            
            self.stdout.write(f"\nProcessing category: {category} (retention: {retention_days} days)")
            
            archived, exported = self._process_category(category, retention_days)
            total_archived += archived
            total_exported += exported
        
        # Legacy AuditLog processing removed - all auditing now uses AuditEvent
        
        # Summary
        self.stdout.write(self.style.SUCCESS(f"\n=== Summary ==="))
        if self.dry_run:
            self.stdout.write(f"Would archive: {total_archived} audit events")
            if self.export:
                self.stdout.write(f"Would export: {total_exported} records to CSV")
        else:
            self.stdout.write(f"Archived: {total_archived} audit events")
            if self.export:
                self.stdout.write(f"Exported: {total_exported} records to CSV")

    def _process_category(self, category: str, retention_days: int) -> tuple[int, int]:
        """Process archiving for a specific category."""
        cutoff_date = timezone.now() - timedelta(days=retention_days)
        
        # Find events to archive
        events_query = AuditEvent.objects.filter(
            category=category,
            timestamp__lt=cutoff_date
        )
        
        event_count = events_query.count()
        
        if event_count == 0:
            self.stdout.write(f"  No events to archive for {category}")
            return 0, 0
        
        self.stdout.write(f"  Found {event_count} events older than {retention_days} days")
        
        # Confirmation prompt
        if not self.force and not self.dry_run:
            confirm = input(f"  Archive {event_count} {category} events? [y/N]: ")
            if confirm.lower() not in ['y', 'yes']:
                self.stdout.write("  Skipped by user")
                return 0, 0
        
        archived_count = 0
        exported_count = 0
        
        if self.export:
            exported_count = self._export_events(events_query, category)
        
        if not self.dry_run:
            archived_count = self._archive_events(events_query, category)
        else:
            archived_count = event_count
        
        return archived_count, exported_count

    def _export_events(self, events_query, category: str) -> int:
        """Export events to CSV before archiving."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{category.lower()}_audit_export_{timestamp}.csv"
        filepath = self.export_path / filename
        
        if self.compress:
            filepath = filepath.with_suffix('.csv.gz')
            file_open = gzip.open
        else:
            file_open = open
        
        exported_count = 0
        
        if self.verbosity >= 1:
            self.stdout.write(f"  Exporting to: {filepath}")
        
        if not self.dry_run:
            with file_open(filepath, 'wt', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                headers = [
                    'id', 'timestamp', 'user_id', 'username', 'action', 'resource',
                    'category', 'severity', 'risk_score', 'ip_address', 'success',
                    'session_id', 'user_agent', 'request_size', 'request_duration_ms',
                    'requires_review', 'reviewed_at', 'reviewed_by_id', 'details',
                    'medical_domain', 'patient_count_accessed', 'data_sensitivity_level',
                    'records_accessed', 'columns_accessed', 'query_hash'
                ]
                writer.writerow(headers)
                
                # Write data in batches
                for batch in self._batch_queryset(events_query.select_related('user', 'reviewed_by'), self.batch_size):
                    for event in batch:
                        # Get data access log if exists
                        data_access = getattr(event, 'data_access_log', None)
                        
                        row = [
                            event.id,
                            event.timestamp.isoformat(),
                            event.user.id if event.user else None,
                            event.user.username if event.user else None,
                            event.action,
                            event.resource,
                            event.category,
                            event.severity,
                            event.risk_score,
                            event.ip_address,
                            event.success,
                            event.session_id,
                            event.user_agent,
                            event.request_size,
                            event.request_duration_ms,
                            event.requires_review,
                            event.reviewed_at.isoformat() if event.reviewed_at else None,
                            event.reviewed_by.id if event.reviewed_by else None,
                            json.dumps(event.details) if event.details else None,
                            data_access.medical_domain if data_access else None,
                            data_access.patient_count_accessed if data_access else None,
                            data_access.data_sensitivity_level if data_access else None,
                            data_access.records_accessed if data_access else None,
                            json.dumps(data_access.columns_accessed) if data_access else None,
                            data_access.query_hash if data_access else None,
                        ]
                        writer.writerow(row)
                        exported_count += 1
        
        else:
            # Dry run - just count
            exported_count = events_query.count()
        
        if self.verbosity >= 1:
            self.stdout.write(f"  Exported {exported_count} records")
        
        return exported_count

    def _archive_events(self, events_query, category: str) -> int:
        """Archive (delete) events after optional export."""
        archived_count = 0
        
        # Handle security incidents if needed
        if not self.keep_incidents:
            # Find security incidents related to these events
            incident_ids = SecurityIncident.objects.filter(
                related_events__in=events_query
            ).values_list('id', flat=True)
            
            if incident_ids:
                self.stdout.write(f"  Archiving {len(incident_ids)} related security incidents")
                SecurityIncident.objects.filter(id__in=incident_ids).delete()
        
        # Delete in batches to avoid memory issues
        with transaction.atomic():
            for batch in self._batch_queryset(events_query, self.batch_size):
                batch_ids = [event.id for event in batch]
                
                # Delete DataAccessLog entries first (foreign key constraint)
                DataAccessLog.objects.filter(audit_event_id__in=batch_ids).delete()
                
                # Delete the events
                deleted_count = AuditEvent.objects.filter(id__in=batch_ids).delete()[0]
                archived_count += deleted_count
                
                if self.verbosity >= 2:
                    self.stdout.write(f"  Archived batch of {deleted_count} events")
        
        if self.verbosity >= 1:
            self.stdout.write(f"  Archived {archived_count} events")
        
        return archived_count


    def _batch_queryset(self, queryset, batch_size: int):
        """Yield queryset in batches to avoid memory issues."""
        total = queryset.count()
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            yield queryset[start:end]

    def _format_size(self, size_bytes: int) -> str:
        """Format byte size in human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"