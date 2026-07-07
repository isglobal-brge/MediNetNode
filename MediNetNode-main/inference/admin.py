from django.contrib import admin
from django.utils.html import format_html
from .models import DeployedModel, PredictionAudit


@admin.register(DeployedModel)
class DeployedModelAdmin(admin.ModelAdmin):
    """Admin interface for DeployedModel."""

    list_display = [
        'name',
        'version',
        'domain',
        'status_badge',
        'is_public',
        'total_predictions',
        'uploaded_by',
        'created_at',
    ]

    list_filter = [
        'status',
        'domain',
        'is_public',
        'source',
        'enable_differential_privacy',
        'created_at',
    ]

    search_fields = [
        'name',
        'version',
        'description',
        'uploaded_by__username',
    ]

    readonly_fields = [
        'checksum',
        'file_size',
        'total_predictions',
        'last_prediction_at',
        'created_at',
        'updated_at',
        'approved_at',
        'epsilon_recommendations',
    ]

    fieldsets = (
        ('Identification', {
            'fields': ('name', 'version', 'description', 'domain')
        }),
        ('Model File', {
            'fields': ('model_file', 'file_size', 'checksum')
        }),
        ('Schema', {
            'fields': ('input_schema', 'output_schema'),
            'classes': ('collapse',)
        }),
        ('Visibility & Status', {
            'fields': ('is_public', 'status')
        }),
        ('Audit Trail', {
            'fields': (
                'uploaded_by',
                'approved_by',
                'created_at',
                'updated_at',
                'approved_at',
            ),
            'classes': ('collapse',)
        }),
        ('Source', {
            'fields': ('source', 'training_session_id'),
            'classes': ('collapse',)
        }),
        ('Metrics', {
            'fields': ('accuracy', 'validation_notes')
        }),
        ('Security - Rate Limiting', {
            'fields': ('max_requests_per_minute', 'max_batch_size'),
            'classes': ('collapse',)
        }),
        ('Security - Differential Privacy', {
            'fields': (
                'enable_differential_privacy',
                'dp_epsilon',
                'dp_noise_scale',
                'epsilon_recommendations',
            ),
            'classes': ('collapse',),
            'description': 'Differential Privacy adds calibrated noise to predictions to prevent reverse engineering. '
                          'Lower epsilon = stronger privacy but more noise. See recommendations below.'
        }),
        ('Usage Statistics', {
            'fields': ('total_predictions', 'last_prediction_at'),
            'classes': ('collapse',)
        }),
    )

    actions = ['approve_models', 'reject_models', 'deprecate_models']

    def status_badge(self, obj):
        """Display status as colored badge."""
        colors = {
            'pending': '#ffc107',
            'approved': '#28a745',
            'deprecated': '#6c757d',
            'rejected': '#dc3545',
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            colors.get(obj.status, '#6c757d'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'

    def approve_models(self, request, queryset):
        """Bulk approve models."""
        count = 0
        for model in queryset.filter(status='pending'):
            model.approve(request.user)
            count += 1
        self.message_user(request, f'{count} model(s) approved successfully.')
    approve_models.short_description = 'Approve selected models'

    def reject_models(self, request, queryset):
        """Bulk reject models."""
        count = 0
        for model in queryset.filter(status='pending'):
            model.reject(request.user, reason='Bulk rejection from admin')
            count += 1
        self.message_user(request, f'{count} model(s) rejected.')
    reject_models.short_description = 'Reject selected models'

    def deprecate_models(self, request, queryset):
        """Bulk deprecate models."""
        count = queryset.filter(status='approved').update(status='deprecated')
        self.message_user(request, f'{count} model(s) deprecated.')
    deprecate_models.short_description = 'Deprecate selected models'

    def epsilon_recommendations(self, obj):
        """Display recommended epsilon values with research paper references."""
        return format_html(
            '<div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; '
            'border-left: 4px solid #007bff; margin: 10px 0;">'
            '<h4 style="margin-top: 0; color: #007bff;">📚 Recommended Epsilon Values by Domain</h4>'
            '<p style="margin-bottom: 10px;"><em>Lower epsilon = stronger privacy protection but more noise added to predictions.</em></p>'

            '<table style="width: 100%; border-collapse: collapse; margin: 10px 0;">'
            '<thead>'
            '<tr style="background-color: #e9ecef;">'
            '<th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Domain</th>'
            '<th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Epsilon (ε)</th>'
            '<th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Privacy Level</th>'
            '<th style="padding: 8px; text-align: left; border: 1px solid #dee2e6;">Est. Accuracy Impact</th>'
            '</tr>'
            '</thead>'
            '<tbody>'
            '<tr>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Oncology</strong></td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">1.0</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><span style="color: #28a745;">●</span> High</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">&gt;10%</td>'
            '</tr>'
            '<tr style="background-color: #f8f9fa;">'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Neurology</strong></td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">1.5</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><span style="color: #28a745;">●</span> High</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">5-10%</td>'
            '</tr>'
            '<tr>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Cardiology</strong></td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">2.0</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><span style="color: #ffc107;">●</span> Medium-High</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">5-10%</td>'
            '</tr>'
            '<tr style="background-color: #f8f9fa;">'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Diabetes / General</strong></td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">3.0</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><span style="color: #ffc107;">●</span> Medium</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">2-5%</td>'
            '</tr>'
            '<tr>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Public Datasets</strong></td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">5.0</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><span style="color: #ff9800;">●</span> Medium</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">1-2%</td>'
            '</tr>'
            '<tr style="background-color: #f8f9fa;">'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><strong>Research</strong></td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">8.0</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;"><span style="color: #dc3545;">●</span> Low</td>'
            '<td style="padding: 8px; border: 1px solid #dee2e6;">&lt;1%</td>'
            '</tr>'
            '</tbody>'
            '</table>'

            '<div style="margin-top: 15px; padding: 10px; background-color: #fff3cd; border-radius: 3px;">'
            '<h5 style="margin-top: 0; color: #856404;">🔬 Research References</h5>'
            '<ul style="margin: 5px 0; padding-left: 20px; font-size: 0.9em;">'

            '<li style="margin: 5px 0;">'
            '<strong>Dwork & Roth (2014)</strong> - "The Algorithmic Foundations of Differential Privacy"<br/>'
            '<a href="https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf" target="_blank" '
            'style="color: #007bff; text-decoration: none;">https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf</a><br/>'
            '<em>Foundational theory on epsilon values and privacy guarantees.</em>'
            '</li>'

            '<li style="margin: 5px 0;">'
            '<strong>Abadi et al. (2016)</strong> - "Deep Learning with Differential Privacy" (Google Brain)<br/>'
            '<a href="https://arxiv.org/abs/1607.00133" target="_blank" '
            'style="color: #007bff; text-decoration: none;">https://arxiv.org/abs/1607.00133</a><br/>'
            '<em>Practical epsilon values for ML models: ε=1-10 range with accuracy trade-offs.</em>'
            '</li>'

            '<li style="margin: 5px 0;">'
            '<strong>Jayaraman & Evans (2019)</strong> - "Evaluating Differentially Private Machine Learning in Practice"<br/>'
            '<a href="https://arxiv.org/abs/1902.08874" target="_blank" '
            'style="color: #007bff; text-decoration: none;">https://arxiv.org/abs/1902.08874</a><br/>'
            '<em>Empirical evaluation showing ε&lt;3 provides strong privacy for medical ML.</em>'
            '</li>'

            '<li style="margin: 5px 0;">'
            '<strong>Beaulieu-Jones et al. (2019)</strong> - "Privacy-Preserving Generative Deep Neural Networks Support Clinical Data Sharing"<br/>'
            '<a href="https://www.ahajournals.org/doi/10.1161/CIRCOUTCOMES.118.005122" target="_blank" '
            'style="color: #007bff; text-decoration: none;">https://www.ahajournals.org/doi/10.1161/CIRCOUTCOMES.118.005122</a><br/>'
            '<em>Medical data privacy: recommends ε=1.0 for sensitive health records (oncology, cardiology).</em>'
            '</li>'

            '<li style="margin: 5px 0;">'
            '<strong>Fredrikson et al. (2015)</strong> - "Model Inversion Attacks that Exploit Confidence Information"<br/>'
            '<a href="https://dl.acm.org/doi/10.1145/2810103.2813677" target="_blank" '
            'style="color: #007bff; text-decoration: none;">https://dl.acm.org/doi/10.1145/2810103.2813677</a><br/>'
            '<em>Demonstrates why low epsilon (strong DP) is critical for preventing model inversion in medical domains.</em>'
            '</li>'

            '</ul>'
            '</div>'

            '<div style="margin-top: 10px; padding: 10px; background-color: #d1ecf1; border-radius: 3px; font-size: 0.9em;">'
            '<strong>ℹ️ Note:</strong> These recommendations are based on peer-reviewed research. '
            'Actual epsilon values should be adjusted based on your specific data sensitivity, '
            'regulatory requirements (HIPAA, GDPR), and acceptable accuracy trade-offs. '
            'Consult with your privacy officer or legal team for compliance guidance.'
            '</div>'

            '</div>'
        )
    epsilon_recommendations.short_description = 'Epsilon Recommendations & Research'

    def save_model(self, request, obj, form, change):
        """Set uploaded_by when creating new model."""
        if not change:  # Creating new model
            obj.uploaded_by = request.user
        super().save_model(request, obj, form, change)


@admin.register(PredictionAudit)
class PredictionAuditAdmin(admin.ModelAdmin):
    """Admin interface for PredictionAudit (read-only)."""

    list_display = [
        'timestamp',
        'user',
        'model_info',
        'records_count',
        'execution_time_ms',
        'success_badge',
        'suspicious_badge',
        'ip_address',
    ]

    list_filter = [
        'success',
        'dp_noise_applied',
        'timestamp',
        'model_domain',
        ('suspicious_score', admin.EmptyFieldListFilter),
    ]

    search_fields = [
        'user__username',
        'model_name',
        'ip_address',
        'input_hash',
    ]

    readonly_fields = [
        'id',
        'user',
        'api_key',
        'ip_address',
        'model',
        'model_name',
        'model_version',
        'model_domain',
        'timestamp',
        'records_count',
        'execution_time_ms',
        'rate_limit_remaining',
        'suspicious_score',
        'patterns_detected',
        'input_hash',
        'success',
        'error_message',
        'dp_noise_applied',
    ]

    fieldsets = (
        ('Request Identity', {
            'fields': ('id', 'timestamp', 'user', 'api_key', 'ip_address')
        }),
        ('Model Information', {
            'fields': ('model', 'model_name', 'model_version', 'model_domain')
        }),
        ('Execution Metrics', {
            'fields': ('records_count', 'execution_time_ms', 'success', 'error_message')
        }),
        ('Security Monitoring', {
            'fields': ('rate_limit_remaining', 'suspicious_score', 'patterns_detected'),
            'classes': ('collapse',)
        }),
        ('Privacy', {
            'fields': ('input_hash', 'dp_noise_applied'),
            'classes': ('collapse',)
        }),
    )

    date_hierarchy = 'timestamp'

    def has_add_permission(self, request):
        """Prevent manual creation of audit logs."""
        return False

    def has_delete_permission(self, request, obj=None):
        """Prevent deletion of audit logs."""
        return False

    def model_info(self, obj):
        """Display model name and version."""
        return f"{obj.model_name} v{obj.model_version}"
    model_info.short_description = 'Model'

    def success_badge(self, obj):
        """Display success status as colored badge."""
        if obj.success:
            return format_html(
                '<span style="background-color: #28a745; color: white; padding: 3px 10px; '
                'border-radius: 3px; font-weight: bold;">Success</span>'
            )
        return format_html(
            '<span style="background-color: #dc3545; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">Failed</span>'
        )
    success_badge.short_description = 'Status'

    def suspicious_badge(self, obj):
        """Display suspicious score as colored badge."""
        if obj.suspicious_score == 0.0:
            color = '#28a745'  # Green
            text = 'Normal'
        elif obj.suspicious_score < 0.5:
            color = '#ffc107'  # Yellow
            text = f'Low ({obj.suspicious_score:.1f})'
        elif obj.suspicious_score < 0.8:
            color = '#ff9800'  # Orange
            text = f'Medium ({obj.suspicious_score:.1f})'
        else:
            color = '#dc3545'  # Red
            text = f'High ({obj.suspicious_score:.1f})'

        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; '
            'border-radius: 3px; font-weight: bold;">{}</span>',
            color,
            text
        )
    suspicious_badge.short_description = 'Suspicious'
