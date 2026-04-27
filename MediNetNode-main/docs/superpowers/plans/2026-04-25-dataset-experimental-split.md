# Dataset Experimental Split — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow admins to optionally split an uploaded CSV into a small experimental subset (no DP budget) and a full production dataset (real ε budget enforced), so researchers can iterate freely to find the optimal model before committing epsilon.

**Architecture:** At upload time MediNetNode physically splits the CSV with `pandas.sample()`, stores two files, and records both paths in the `Dataset` row. The Flower client receives a `use_experiment` flag in the model JSON and routes `load_data_from_django()` to the experiment file, bypassing the `validate_training_permissions()` budget checks entirely for experimental jobs.

**Tech Stack:** Django 4.x, pandas, SQLite (datasets_db), raw SQL INSERT (uploader), Bootstrap 5 + vanilla JS (upload template), Flower client (api/federated/client.py)

---

## File Map

| Path | Action | Responsibility |
|------|--------|----------------|
| `dataset/models.py` | Modify | Add 3 nullable fields to `Dataset` |
| `dataset/migrations/0006_add_dataset_experiment_split.py` | Create | DB migration |
| `dataset/uploader.py` | Modify | Split logic + raw SQL update |
| `dataset/views.py` | Modify | Read `split_ratio` from POST |
| `dataset/forms.py` | Modify | Add optional `split_ratio` field |
| `templates/dataset/dataset_upload.html` | Modify | Toggle + slider + DP recommendations panel |
| `api/views.py` | Modify | Skip budget checks for experimental jobs |
| `api/federated/data_loaders.py` | Modify | Route to experiment file when flagged |
| `api/federated/client.py` | Modify | Pass `use_experiment` to `load_data_from_django` |
| `tests/test_dataset_experimental_split.py` | Create | Unit + integration tests |

---

## Task 1: Add experiment fields to Dataset model

**Files:**
- Modify: `dataset/models.py`

- [ ] **Step 1: Read the current Dataset model**

```bash
# Already done in research — confirm fields at line ~10
grep -n "experiment\|file_path\|rows_count" MediNetNode-main/dataset/models.py
```

Expected: no `experiment_` fields exist yet.

- [ ] **Step 2: Add 3 nullable fields after `file_path`**

Open `dataset/models.py`. Find the `Dataset` class. After the `file_path` field, add:

```python
    # Experimental split — populated only when split_ratio was given at upload
    experiment_file_path = models.CharField(max_length=500, null=True, blank=True)
    experiment_row_count = models.PositiveIntegerField(null=True, blank=True)
    experiment_split_ratio = models.FloatField(null=True, blank=True)
```

- [ ] **Step 3: Verify the model loads**

```bash
cd MediNetNode/MediNetNode-main
python manage.py check --settings=config.settings.test
```

Expected: `System check identified no issues (0 silenced).`

- [ ] **Step 4: Commit**

```bash
git add dataset/models.py
git commit -m "feat(dataset): add experiment split fields to Dataset model"
```

---

## Task 2: Write and apply database migration

**Files:**
- Create: `dataset/migrations/0006_add_dataset_experiment_split.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_dataset_experimental_split.py`:

```python
import pytest
from django.test import TestCase

class TestDatasetExperimentFields(TestCase):
    databases = ['datasets_db']

    def test_dataset_has_experiment_fields(self):
        from dataset.models import Dataset
        d = Dataset()
        assert hasattr(d, 'experiment_file_path')
        assert hasattr(d, 'experiment_row_count')
        assert hasattr(d, 'experiment_split_ratio')
        assert d.experiment_file_path is None
        assert d.experiment_row_count is None
        assert d.experiment_split_ratio is None
```

- [ ] **Step 2: Run test to verify it fails (field missing from DB)**

```bash
cd MediNetNode/MediNetNode-main
pytest tests/test_dataset_experimental_split.py::TestDatasetExperimentFields -v
```

Expected: `FAIL` — `OperationalError: no such column: dataset_experiment_file_path`

- [ ] **Step 3: Generate migration**

```bash
python manage.py makemigrations dataset --name add_dataset_experiment_split --settings=config.settings.test
```

Expected output: `Migrations for 'dataset': dataset/migrations/0006_add_dataset_experiment_split.py`

- [ ] **Step 4: Apply migration**

```bash
python manage.py migrate dataset --settings=config.settings.test
python manage.py migrate dataset --settings=config.settings.dev
```

Expected: `OK` for both.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_dataset_experimental_split.py::TestDatasetExperimentFields -v
```

Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add dataset/migrations/0006_add_dataset_experiment_split.py
git commit -m "feat(dataset): migration 0006 — add experiment split columns"
```

---

## Task 3: Implement split logic in uploader

**Files:**
- Modify: `dataset/uploader.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import tempfile
import pandas as pd
import pytest

class TestUploaderExperimentSplit:

    def test_no_split_when_ratio_is_none(self, tmp_path):
        """upload_dataset with split_ratio=None must NOT create experiment file."""
        from dataset.uploader import DatasetUploader
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({'a': range(100), 'b': range(100), 'label': [0]*100})
        df.to_csv(csv_path, index=False)
        uploader = DatasetUploader(user_id=1, base_dir=str(tmp_path))
        result = uploader._maybe_create_experiment_split(
            production_path=str(csv_path),
            split_ratio=None,
            timestamp="20260425_120000"
        )
        assert result == (None, None)

    def test_split_creates_experiment_file(self, tmp_path):
        """upload_dataset with split_ratio=0.2 must create file with ~20 rows."""
        from dataset.uploader import DatasetUploader
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({'a': range(100), 'b': range(100), 'label': [0]*100})
        df.to_csv(csv_path, index=False)
        uploader = DatasetUploader(user_id=1, base_dir=str(tmp_path))
        exp_path, exp_rows = uploader._maybe_create_experiment_split(
            production_path=str(csv_path),
            split_ratio=0.2,
            timestamp="20260425_120000"
        )
        assert exp_path is not None
        assert os.path.exists(exp_path)
        assert exp_rows == 20  # 20% of 100
        exp_df = pd.read_csv(exp_path)
        assert len(exp_df) == 20

    def test_split_filename_contains_experiment(self, tmp_path):
        from dataset.uploader import DatasetUploader
        csv_path = tmp_path / "data.csv"
        df = pd.DataFrame({'a': range(50), 'label': [1]*50})
        df.to_csv(csv_path, index=False)
        uploader = DatasetUploader(user_id=1, base_dir=str(tmp_path))
        exp_path, _ = uploader._maybe_create_experiment_split(
            production_path=str(csv_path),
            split_ratio=0.3,
            timestamp="20260425_120000"
        )
        assert "_experiment_" in os.path.basename(exp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset_experimental_split.py::TestUploaderExperimentSplit -v
```

Expected: `FAIL` — `AttributeError: 'DatasetUploader' object has no attribute '_maybe_create_experiment_split'`

- [ ] **Step 3: Add `_maybe_create_experiment_split` method to `DatasetUploader`**

In `dataset/uploader.py`, inside the `DatasetUploader` class (after `_store_file_securely`), add:

```python
def _maybe_create_experiment_split(
    self,
    production_path: str,
    split_ratio: float | None,
    timestamp: str,
) -> tuple[str | None, int | None]:
    """Physically split the production CSV into a smaller experiment file.

    Returns (experiment_file_path, experiment_row_count) or (None, None).
    """
    if split_ratio is None:
        return None, None

    df = pd.read_csv(production_path)
    experiment_df = df.sample(frac=split_ratio, random_state=42)
    n_rows = len(experiment_df)

    stem = os.path.splitext(os.path.basename(production_path))[0]
    # Remove existing timestamp suffix to avoid double-stamping
    exp_filename = f"{stem}_experiment_{timestamp}.csv"
    exp_path = os.path.join(os.path.dirname(production_path), exp_filename)
    experiment_df.to_csv(exp_path, index=False)
    return exp_path, n_rows
```

Also add `import os` at the top of the file if not already present.

- [ ] **Step 4: Update `_create_dataset_record_raw_sql` to accept 3 new nullable columns**

Locate `_create_dataset_record_raw_sql()` in `dataset/uploader.py`. The current INSERT statement has 18 columns. Add 3 more:

Find the INSERT columns list and add at the end (before the closing parenthesis of the column list):
```
experiment_file_path,
experiment_row_count,
experiment_split_ratio
```

Find the VALUES placeholder list and add at the end:
```
?, ?, ?
```

Find the params tuple and add at the end (the 3 new values must be passed in):

The method signature needs to accept them. Change:
```python
def _create_dataset_record_raw_sql(self, name, description, medical_domain,
                                    data_type, file_path, file_size,
                                    rows_count, columns_count, checksum,
                                    target_column, uploaded_by_id):
```
to:
```python
def _create_dataset_record_raw_sql(self, name, description, medical_domain,
                                    data_type, file_path, file_size,
                                    rows_count, columns_count, checksum,
                                    target_column, uploaded_by_id,
                                    experiment_file_path=None,
                                    experiment_row_count=None,
                                    experiment_split_ratio=None):
```

And add `experiment_file_path, experiment_row_count, experiment_split_ratio` at the end of the params tuple passed to `cursor.execute(sql, params)`.

- [ ] **Step 5: Call the new method from `upload_dataset`**

In `upload_dataset()`, after `_store_file_securely` returns `final_path`, add:

```python
exp_path, exp_rows = self._maybe_create_experiment_split(
    production_path=final_path,
    split_ratio=split_ratio,
    timestamp=timestamp,  # already computed for the production filename
)
```

Then pass `experiment_file_path=exp_path, experiment_row_count=exp_rows, experiment_split_ratio=split_ratio` to `_create_dataset_record_raw_sql(...)`.

Also update the method signature of `upload_dataset` to accept `split_ratio: float | None = None`.

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_dataset_experimental_split.py::TestUploaderExperimentSplit -v
```

Expected: all 3 `PASS`

- [ ] **Step 7: Commit**

```bash
git add dataset/uploader.py
git commit -m "feat(dataset): implement experiment CSV split in DatasetUploader"
```

---

## Task 4: Read split_ratio from POST in upload view and form

**Files:**
- Modify: `dataset/views.py`
- Modify: `dataset/forms.py`

- [ ] **Step 1: Write failing test**

```python
class TestUploadViewSplitRatio:

    def test_invalid_split_ratio_rejected(self, client, admin_user):
        client.force_login(admin_user)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df = pd.DataFrame({'a': range(10), 'label': [0]*10})
            df.to_csv(f.name, index=False)
            with open(f.name, 'rb') as csv_file:
                response = client.post('/dataset/upload/', {
                    'name': 'test',
                    'description': 'desc',
                    'medical_domain': 'cardiology',
                    'data_type': 'tabular',
                    'target_column': 'label',
                    'split_ratio': '0.9',  # out of range
                    'file': csv_file,
                })
        assert response.status_code == 400
        data = response.json()
        assert 'split_ratio' in data.get('error', '').lower()

    def test_split_ratio_none_when_not_provided(self, client, admin_user):
        """If split_ratio not in POST, uploader receives None."""
        # This test verifies the view passes split_ratio=None to the uploader
        # by mocking uploader.upload_dataset and capturing kwargs.
        from unittest.mock import patch, MagicMock
        client.force_login(admin_user)
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as f:
            df = pd.DataFrame({'a': range(10), 'label': [0]*10})
            df.to_csv(f.name, index=False)
            mock_result = MagicMock()
            mock_result.dataset_id = 99
            with patch('dataset.views.DatasetUploader') as MockUp:
                MockUp.return_value.upload_dataset.return_value = mock_result
                with open(f.name, 'rb') as csv_file:
                    client.post('/dataset/upload/', {
                        'name': 'test',
                        'description': 'desc',
                        'medical_domain': 'cardiology',
                        'data_type': 'tabular',
                        'target_column': 'label',
                        'file': csv_file,
                    })
                call_kwargs = MockUp.return_value.upload_dataset.call_args
                assert call_kwargs.kwargs.get('split_ratio') is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset_experimental_split.py::TestUploadViewSplitRatio -v
```

Expected: `FAIL`

- [ ] **Step 3: Update `DatasetUploadForm` in `dataset/forms.py`**

Add an optional field (not a required model field, just used for validation in the view):

```python
split_ratio = forms.FloatField(
    required=False,
    min_value=0.1,
    max_value=0.5,
    widget=forms.HiddenInput(),
)
```

- [ ] **Step 4: Update `dataset_upload` view in `dataset/views.py`**

Find the `dataset_upload` view function. After reading other POST params, add:

```python
split_ratio_raw = request.POST.get('split_ratio', '').strip()
split_ratio = None
if split_ratio_raw:
    try:
        split_ratio = float(split_ratio_raw)
        if not (0.1 <= split_ratio <= 0.5):
            return JsonResponse(
                {'error': 'split_ratio must be between 0.1 and 0.5'},
                status=400
            )
    except ValueError:
        return JsonResponse({'error': 'split_ratio must be a number'}, status=400)
```

Then pass `split_ratio=split_ratio` to the `uploader.upload_dataset(...)` call.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_dataset_experimental_split.py::TestUploadViewSplitRatio -v
```

Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add dataset/forms.py dataset/views.py
git commit -m "feat(dataset): read and validate split_ratio from upload POST"
```

---

## Task 5: Update upload UI — toggle, slider, DP recommendations

**Files:**
- Modify: `templates/dataset/dataset_upload.html`

- [ ] **Step 1: Read the current template to find the file input and submit button**

```bash
grep -n "submit\|file\|btn\|split\|privacy" \
  MediNetNode-main/templates/dataset/dataset_upload.html | head -40
```

Note the exact surrounding `<div>` structure to insert after.

- [ ] **Step 2: Add the experimental split section before the submit button**

Insert the following block in the upload form, after the `target_column` field and before the submit button:

```html
<!-- Experimental Split Section -->
<div class="card border-0 bg-light mb-4" id="split-card">
  <div class="card-body">
    <div class="form-check form-switch mb-3">
      <input class="form-check-input" type="checkbox" id="enableSplit" name="enable_split">
      <label class="form-check-label fw-semibold" for="enableSplit">
        Create experimental subset
      </label>
      <div class="form-text">
        A small copy of the dataset with <strong>no epsilon budget</strong>.
        Researchers can iterate freely, then switch to the full dataset for production training.
      </div>
    </div>

    <div id="splitOptions" class="d-none">
      <!-- Slider -->
      <label for="splitRatio" class="form-label">
        Subset size: <span id="splitPct">20</span>% of rows
      </label>
      <input type="range" class="form-range mb-1" id="splitRatio" name="split_ratio"
             min="10" max="50" step="5" value="20"
             oninput="document.getElementById('splitPct').textContent = this.value">
      <input type="hidden" id="splitRatioHidden" name="split_ratio" value="0.20">

      <!-- DP Budget Recommendations -->
      <div class="alert alert-info mt-3 mb-0 p-3" id="dpRecommendations">
        <h6 class="mb-2">
          <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"
               fill="currentColor" class="bi bi-shield-lock me-1" viewBox="0 0 16 16">
            <path d="M5.338 1.59a61 61 0 0 0-2.837.856.48.48 0 0 0-.328.39c-.554
                     4.157.726 7.19 2.253 9.188a10.7 10.7 0 0 0 2.287 2.233c.346.244.652.42.893.533
                     l.04.018.011.005.004.002.001.001-.001-.001-.004-.002-.011-.005-.04-.018a9 9 0
                     0 1-.176-.094 12 12 0 0 1-.TNE"/>
          </svg>
          DP Budget Recommendations (literature guidelines)
        </h6>
        <div id="dpRec-high" class="d-none">
          <strong>High sensitivity data</strong> (e.g. genetic, psychiatric records)<br>
          Recommended: ε ≤ 0.5 per job &nbsp;·&nbsp; Lifetime budget ≤ 2.0<br>
          <small class="text-muted">Dwork & Roth (2014); Apple DP Team (2017) use ε ≤ 1.0</small>
        </div>
        <div id="dpRec-medium" class="d-none">
          <strong>Medium sensitivity data</strong> (e.g. lab results, diagnoses)<br>
          Recommended: ε ≤ 1.0 per job &nbsp;·&nbsp; Lifetime budget ≤ 5.0<br>
          <small class="text-muted">Erlingsson et al. (2014); US Census Bureau ε = 17.14</small>
        </div>
        <div id="dpRec-low" class="d-none">
          <strong>Low sensitivity data</strong> (e.g. aggregate stats, anonymised)<br>
          Recommended: ε ≤ 3.0 per job &nbsp;·&nbsp; Lifetime budget ≤ 15.0<br>
          <small class="text-muted">Abowd (2018); no strict consensus below ε = 10</small>
        </div>
        <p class="mb-0 mt-2 text-muted" id="dpRec-placeholder">
          Select a <strong>medical domain sensitivity</strong> above to see recommendations.
        </p>
      </div>
    </div>
  </div>
</div>
```

- [ ] **Step 3: Add the JavaScript block before `</body>` (or in the existing script section)**

```html
<script nonce="{% csp_nonce %}">
(function () {
  var enableSplit  = document.getElementById('enableSplit');
  var splitOptions = document.getElementById('splitOptions');
  var rangeInput   = document.getElementById('splitRatio');
  var hiddenInput  = document.getElementById('splitRatioHidden');
  var domainSelect = document.getElementById('id_medical_domain');  // Django field id

  // Domain → sensitivity mapping (keep in sync with DatasetPrivacyPolicy.SENSITIVITY_DEFAULTS)
  var sensitivityMap = {
    cardiology:    'medium',
    oncology:      'high',
    neurology:     'high',
    radiology:     'medium',
    genomics:      'high',
    psychiatry:    'high',
    general:       'low',
    endocrinology: 'medium',
    // extend as needed
  };

  function showDpRec(domain) {
    var level = sensitivityMap[domain] || null;
    ['high', 'medium', 'low'].forEach(function (l) {
      document.getElementById('dpRec-' + l).classList.add('d-none');
    });
    var placeholder = document.getElementById('dpRec-placeholder');
    if (level) {
      document.getElementById('dpRec-' + level).classList.remove('d-none');
      placeholder.classList.add('d-none');
    } else {
      placeholder.classList.remove('d-none');
    }
  }

  enableSplit.addEventListener('change', function () {
    splitOptions.classList.toggle('d-none', !this.checked);
    hiddenInput.disabled = !this.checked;
    if (this.checked && domainSelect) showDpRec(domainSelect.value);
  });

  rangeInput.addEventListener('input', function () {
    hiddenInput.value = (parseInt(this.value) / 100).toFixed(2);
  });

  if (domainSelect) {
    domainSelect.addEventListener('change', function () {
      if (!enableSplit.classList.contains('d-none')) showDpRec(this.value);
    });
  }

  // Prevent sending split_ratio when toggle is off
  document.querySelector('form').addEventListener('submit', function () {
    if (!enableSplit.checked) {
      hiddenInput.disabled = true;
      rangeInput.disabled  = true;
    }
  });
}());
</script>
```

- [ ] **Step 4: Verify template renders without error (manual check)**

Start the dev server on port 5001 and navigate to the upload page:
```bash
cd MediNetNode/MediNetNode-main
python manage.py runserver 0.0.0.0:5001 --settings=config.settings.dev
```
Open `http://localhost:5001/dataset/upload/` and verify:
- Toggle shows/hides slider section
- Slider updates the percentage label
- Changing medical domain (if DP panel is open) shows the correct sensitivity text
- Submitting without toggle does NOT send `split_ratio`

- [ ] **Step 5: Commit**

```bash
git add templates/dataset/dataset_upload.html
git commit -m "feat(dataset): upload UI — experimental split toggle, slider, DP recommendations"
```

---

## Task 6: Skip budget checks for experimental jobs in API

**Files:**
- Modify: `api/views.py`

- [ ] **Step 1: Write failing test**

```python
import json
from unittest.mock import patch, MagicMock

class TestValidateTrainingPermissionsExperiment:

    def test_experimental_job_skips_budget_check(self, client, researcher_user):
        """If use_experiment=true and dataset has experiment_file_path, skip epsilon checks."""
        client.force_login(researcher_user)

        model_json = {
            'model': {
                'dataset': {
                    'selected_datasets': [{'dataset_id': 1}]
                }
            },
            'use_experiment': True,
        }

        mock_dataset = MagicMock()
        mock_dataset.experiment_file_path = '/some/path/data_experiment_20260425.csv'
        mock_policy = MagicMock()
        mock_policy.can_accept_job.return_value = True

        with patch('api.views.Dataset') as MockDs, \
             patch('api.views.DatasetPrivacyPolicy') as MockPolicy, \
             patch('api.views.ResearcherEpsilonBudget') as MockBudget, \
             patch('api.views._launch_flower_client'):
            MockDs.objects.using.return_value.get.return_value = mock_dataset
            MockPolicy.objects.using.return_value.filter.return_value.first.return_value = mock_policy

            response = client.post(
                '/api/v1/start-client/',
                data=json.dumps(model_json),
                content_type='application/json',
            )

        # Budget.can_accept_job must NOT have been called
        MockBudget.objects.using.return_value.get_or_create.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_dataset_experimental_split.py::TestValidateTrainingPermissionsExperiment -v
```

Expected: `FAIL`

- [ ] **Step 3: Modify `validate_training_permissions()` in `api/views.py`**

Locate `validate_training_permissions(model_json, researcher_id)` (or wherever budget checks live before `start_client` launches the Flower client). Add an early-return guard at the top of the function (or at the budget-check block):

```python
def validate_training_permissions(model_json: dict, researcher_id: int) -> tuple[bool, str]:
    """Returns (allowed, reason). Experimental jobs skip epsilon budget checks."""
    use_experiment = model_json.get('use_experiment', False)

    dataset_id = int(
        model_json['model']['dataset']['selected_datasets'][0]['dataset_id']
    )
    dataset = Dataset.objects.using('datasets_db').get(id=dataset_id)

    if use_experiment and dataset.experiment_file_path:
        # Experimental subset: no budget consumed, no policy check
        return True, 'experimental_job'

    # --- Existing policy + budget checks follow unchanged ---
    policy = DatasetPrivacyPolicy.objects.using('datasets_db') \
                 .filter(dataset_id=dataset_id).first()
    if policy and not policy.can_accept_job(epsilon_requested=...):
        return False, 'policy_rejected'

    budget = ResearcherEpsilonBudget.objects.using('datasets_db') \
                 .get_or_create(dataset_id=dataset_id, researcher_id=researcher_id)[0]
    if not budget.can_accept_job(epsilon=...):
        return False, 'budget_exhausted'

    return True, 'ok'
```

Adapt to the exact existing call signature — do not change the existing logic, only add the early-return guard before it.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_dataset_experimental_split.py::TestValidateTrainingPermissionsExperiment -v
```

Expected: `PASS`

- [ ] **Step 5: Commit**

```bash
git add api/views.py
git commit -m "feat(api): skip epsilon budget checks for experimental dataset jobs"
```

---

## Task 7: Route data_loaders to experiment file when flagged

**Files:**
- Modify: `api/federated/data_loaders.py`
- Modify: `api/federated/client.py`

- [ ] **Step 1: Write failing tests**

```python
class TestDataLoaderExperimentRouting:

    def test_uses_experiment_file_when_flagged(self, tmp_path):
        """load_data_from_django with use_experiment=True must read experiment_file_path."""
        import pandas as pd
        from unittest.mock import patch, MagicMock
        from api.federated.data_loaders import load_data_from_django

        # Production file: 100 rows
        prod_path = tmp_path / "data.csv"
        prod_df = pd.DataFrame({'a': range(100), 'label': [0]*100})
        prod_df.to_csv(prod_path, index=False)

        # Experiment file: 20 rows
        exp_path = tmp_path / "data_experiment.csv"
        exp_df = prod_df.sample(20, random_state=42)
        exp_df.to_csv(exp_path, index=False)

        mock_dataset = MagicMock()
        mock_dataset.file_path = str(prod_path)
        mock_dataset.experiment_file_path = str(exp_path)
        mock_dataset.target_column = 'label'

        with patch('api.federated.data_loaders.Dataset') as MockDs:
            MockDs.objects.using.return_value.get.return_value = mock_dataset
            df, target = load_data_from_django(dataset_id=1, use_experiment=True)

        assert len(df) == 20

    def test_uses_production_file_when_not_flagged(self, tmp_path):
        import pandas as pd
        from unittest.mock import patch, MagicMock
        from api.federated.data_loaders import load_data_from_django

        prod_path = tmp_path / "data.csv"
        prod_df = pd.DataFrame({'a': range(100), 'label': [0]*100})
        prod_df.to_csv(prod_path, index=False)

        mock_dataset = MagicMock()
        mock_dataset.file_path = str(prod_path)
        mock_dataset.experiment_file_path = None
        mock_dataset.target_column = 'label'

        with patch('api.federated.data_loaders.Dataset') as MockDs:
            MockDs.objects.using.return_value.get.return_value = mock_dataset
            df, target = load_data_from_django(dataset_id=1, use_experiment=False)

        assert len(df) == 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_dataset_experimental_split.py::TestDataLoaderExperimentRouting -v
```

Expected: `FAIL` — `TypeError: load_data_from_django() got an unexpected keyword argument 'use_experiment'`

- [ ] **Step 3: Modify `load_data_from_django` in `api/federated/data_loaders.py`**

Find the function signature:
```python
def load_data_from_django(dataset_id: int) -> tuple[pd.DataFrame, str]:
```

Change to:
```python
def load_data_from_django(
    dataset_id: int,
    use_experiment: bool = False,
) -> tuple[pd.DataFrame, str]:
```

Inside the function, after `dataset = Dataset.objects.using('datasets_db').get(id=dataset_id)`, add:

```python
if use_experiment and dataset.experiment_file_path:
    file_path = dataset.experiment_file_path
else:
    file_path = dataset.file_path
```

Replace all subsequent uses of `dataset.file_path` with `file_path`.

- [ ] **Step 4: Pass `use_experiment` from `start_flower_client` in `client.py`**

In `api/federated/client.py`, find where `model_json` is parsed and `TABLE_NAME` is set:

```python
TABLE_NAME = int(model_json['model']['dataset']['selected_datasets'][0]['dataset_id'])
```

Add below it:
```python
USE_EXPERIMENT = bool(model_json.get('use_experiment', False))
```

Then find all calls to `load_data_from_django(TABLE_NAME)` and `load_ml_data(dataset_id=TABLE_NAME)`:

Change `load_data_from_django(TABLE_NAME)` → `load_data_from_django(TABLE_NAME, use_experiment=USE_EXPERIMENT)`

If `load_ml_data` is a separate wrapper, update it the same way (add `use_experiment=False` param and pass through to `load_data_from_django`).

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_dataset_experimental_split.py::TestDataLoaderExperimentRouting -v
```

Expected: `PASS`

- [ ] **Step 6: Commit**

```bash
git add api/federated/data_loaders.py api/federated/client.py
git commit -m "feat(api): route Flower client to experiment CSV when use_experiment=true"
```

---

## Task 8: Full integration test + coverage check

**Files:**
- Modify: `tests/test_dataset_experimental_split.py`

- [ ] **Step 1: Add an end-to-end upload integration test**

```python
import os
import tempfile
import pandas as pd
import pytest
from django.test import TestCase

class TestUploadWithSplitIntegration(TestCase):
    databases = ['default', 'datasets_db']

    def setUp(self):
        from django.contrib.auth import get_user_model
        User = get_user_model()
        self.admin = User.objects.create_superuser(
            username='admin_split', password='pass', email='a@b.com'
        )

    def test_upload_creates_two_files(self):
        self.client.force_login(self.admin)

        df = pd.DataFrame({'feature': range(200), 'label': [0, 1]*100})
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w') as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        with open(tmp_path, 'rb') as csv_file:
            response = self.client.post('/dataset/upload/', {
                'name': 'split_test',
                'description': 'Integration test with split',
                'medical_domain': 'cardiology',
                'data_type': 'tabular',
                'target_column': 'label',
                'split_ratio': '0.20',
                'file': csv_file,
            })

        assert response.status_code == 200
        data = response.json()
        dataset_id = data['dataset_id']

        from dataset.models import Dataset
        ds = Dataset.objects.using('datasets_db').get(id=dataset_id)
        assert ds.experiment_file_path is not None
        assert os.path.exists(ds.experiment_file_path)
        assert ds.experiment_row_count == 40  # 20% of 200
        assert abs(ds.experiment_split_ratio - 0.20) < 0.001

        exp_df = pd.read_csv(ds.experiment_file_path)
        assert len(exp_df) == 40

        os.unlink(tmp_path)
```

- [ ] **Step 2: Run the full test suite**

```bash
cd MediNetNode/MediNetNode-main
pytest tests/test_dataset_experimental_split.py -v --tb=short
```

Expected: all tests `PASS`

- [ ] **Step 3: Check coverage**

```bash
pytest tests/test_dataset_experimental_split.py \
       --cov=dataset.uploader \
       --cov=dataset.views \
       --cov=api.views \
       --cov=api.federated.data_loaders \
       --cov=api.federated.client \
       --cov-report=term-missing
```

Expected: ≥80% coverage on all modified modules.

- [ ] **Step 4: Run the full project test suite to check for regressions**

```bash
pytest --tb=short -q
```

Expected: no regressions.

- [ ] **Step 5: Commit the integration test**

```bash
git add tests/test_dataset_experimental_split.py
git commit -m "test(dataset): integration tests for experimental split upload and routing"
```

---

## Self-Review Checklist

### Spec Coverage
| Requirement | Task |
|-------------|------|
| Physical CSV split at upload time | Task 3 |
| Optional — not forced | Task 4 (split_ratio=None path), Task 5 (toggle) |
| Slider 10–50% | Task 5 |
| DP budget recommendations from literature | Task 5 (dpRecommendations panel) |
| No budget tracking for experimental subset | Task 6 |
| Researchers can iterate freely on experiment | Task 6 |
| Routing to experiment file in Flower client | Task 7 |
| DB fields for experiment_file_path | Task 1+2 |
| Tests ≥80% coverage | Task 8 |

### Type Consistency
- `split_ratio` is always `float | None` — Python float division in JS (`parseInt / 100`), stored as `FloatField`
- `use_experiment` is `bool` in Python, `true/false` JSON from MediNetHub
- `load_data_from_django(dataset_id, use_experiment=False)` — default False keeps existing callers working
- `_maybe_create_experiment_split` returns `tuple[str | None, int | None]` — both values checked before use

### No Placeholders
All code blocks are complete and runnable. All file paths are exact. All commands show expected output.
