
function switchTab(tabId) {
    // Buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.innerText.toLowerCase().includes(tabId)) btn.classList.add('active');
    });
    // Content
    document.querySelectorAll('.tab-content').forEach(section => {
        section.classList.remove('active');
    });
    document.getElementById(tabId + '-section').classList.add('active');
}

// Setup Tag Inputs for Training
function setupTagInput(containerId) {
    const container = document.getElementById(containerId);
    const input = container.querySelector('input');
    const tagsDiv = container.querySelector('.tags');

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            const val = input.value.trim();
            if (val && !isNaN(val)) {
                addTag(tagsDiv, val);
                input.value = '';
            }
        }
    });

    function addTag(parent, text) {
        const tag = document.createElement('span');
        tag.className = 'tag';
        tag.innerHTML = `${text} <span class="remove-tag">&times;</span>`;
        tag.querySelector('.remove-tag').addEventListener('click', () => {
            tag.remove();
        });
        parent.appendChild(tag);
    }
}

['mnb_alphas', 'cnb_alphas', 'bnb_alphas'].forEach(setupTagInput);

// UI Toggles for Config
document.getElementById('algorithm').addEventListener('change', function () {
    const val = this.value;
    const isNB = ['mnb', 'cnb', 'bnb', 'best_naive_bayes'].includes(val);

    // Show/Hide NB specific settings
    document.getElementById('norm-container').style.display = (val === 'cnb') ? 'block' : 'none';
    document.getElementById('binarize-container').style.display = (val === 'bnb') ? 'block' : 'none';
    document.getElementById('alpha').closest('.config-item').style.display = isNB ? 'block' : 'none';
    document.getElementById('fit_prior').closest('.config-item').style.display = isNB ? 'block' : 'none';
});

// Single Analysis
document.getElementById('analyze-btn').addEventListener('click', async () => {
    const title = document.getElementById('title').value;
    const text = document.getElementById('text').value;

    if (!title || !text) {
        alert("Please fill in both fields.");
        return;
    }

    const analyzeBtn = document.getElementById('analyze-btn');
    const resultContainer = document.getElementById('result-container');

    analyzeBtn.innerText = "Analyzing...";
    analyzeBtn.disabled = true;

    try {
        const payload = {
            title: title,
            text: text,
            algorithm: document.getElementById('algorithm').value,
            alpha: document.getElementById('alpha').value,
            fit_prior: document.getElementById('fit_prior').checked,
            norm: document.getElementById('norm').checked,
            binarize: document.getElementById('binarize').value
        };

        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        // Update UI
        resultContainer.classList.remove('hidden');
        const badge = document.getElementById('result-badge');
        badge.innerText = data.label;
        badge.className = 'badge ' + data.label.toLowerCase();

        const confidence = (data.confidence * 100).toFixed(1);
        document.getElementById('confidence-text').innerText = confidence + "%";
        document.getElementById('confidence-fill').style.width = confidence + "%";
        document.getElementById('algo-name-display').innerText = data.algorithm_used;

        // Features
        const wordCloud = document.getElementById('top-words');
        wordCloud.innerHTML = '';
        data.top_features.forEach(feat => {
            const chip = document.createElement('span');
            chip.className = 'word-chip';
            chip.innerHTML = `${feat.word} <small>${feat.weight.toFixed(2)}</small>`;
            wordCloud.appendChild(chip);
        });

    } catch (err) {
        alert("Server error. Make sure app.py is running.");
    } finally {
        analyzeBtn.innerText = "Analyze Credibility";
        analyzeBtn.disabled = false;
    }
});

// Batch Analysis
document.getElementById('batch-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('batch-file');
    if (!fileInput.files[0]) {
        alert("Please select a CSV file.");
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('algorithm', document.getElementById('algorithm').value);

    const statusMsg = document.getElementById('batch-status');
    statusMsg.innerText = "Processing batch... (this may take a few seconds)";

    try {
        const response = await fetch('/predict_batch', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (data.error) throw new Error(data.error);

        statusMsg.innerText = `Success! Processed ${data.summary.total} records.`;

        // Update Analytics UI
        const analytics = document.getElementById('batch-analytics');
        analytics.classList.remove('hidden');

        document.getElementById('batch-real-perc').innerText = data.summary.real_percentage + "%";
        document.getElementById('batch-real-fill').style.width = data.summary.real_percentage + "%";

        document.getElementById('batch-fake-perc').innerText = data.summary.fake_percentage + "%";
        document.getElementById('batch-fake-fill').style.width = data.summary.fake_percentage + "%";

        document.getElementById('batch-total-count').innerText = data.summary.total;

        // Show brief result preview
        const resultsDiv = document.getElementById('batch-results');
        resultsDiv.innerHTML = '<h4>Results (Preview)</h4>';
        data.results.slice(0, 10).forEach(res => {
            const item = document.createElement('div');
            item.className = 'batch-item';
            item.innerHTML = `<strong>${res.prediction}:</strong> ${res.title.substring(0, 50)}...`;
            resultsDiv.appendChild(item);
        });

    } catch (err) {
        statusMsg.innerText = "Error: " + err.message;
    }
});

// Model Training
document.getElementById('train-trigger-btn').addEventListener('click', async () => {
    const consoleBox = document.getElementById('train-console');
    consoleBox.innerHTML += "<p>> Requesting training...</p>";

    try {
        const getGrid = (containerId) => {
            const tags = document.querySelectorAll(`#${containerId} .tag`);
            const values = Array.from(tags).map(t => parseFloat(t.innerText));
            return values.length > 0 ? { alpha: values, fit_prior: [true, false] } : null;
        };

        const payload = {
            mnb_grid: getGrid('mnb_alphas'),
            cnb_grid: getGrid('cnb_alphas'),
            bnb_grid: getGrid('bnb_alphas')
        };

        // Remove nulls
        Object.keys(payload).forEach(k => payload[k] == null && delete payload[k]);

        const resp = await fetch('/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await resp.json();
        consoleBox.innerHTML += `<p>> ${data.status}</p>`;

        // Start polling
        const poll = setInterval(async () => {
            const statusResp = await fetch('/train/status');
            const status = await statusResp.json();

            if (status.ongoing) {
                consoleBox.innerHTML = "<p>> Training in progress... Please wait.</p>";
            } else {
                clearInterval(poll);
                consoleBox.innerHTML = "<h4>Training Complete!</h4><pre>" + status.last_result.substring(0, 500) + "...</pre>";
            }
        }, 3000);

    } catch (err) {
        consoleBox.innerHTML += `<p class="error">> Fatal: ${err.message}</p>`;
    }
});
