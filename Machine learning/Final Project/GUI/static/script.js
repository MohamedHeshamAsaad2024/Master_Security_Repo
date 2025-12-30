document.addEventListener('DOMContentLoaded', () => {
    const analyzeBtn = document.getElementById('analyze-btn');
    const modelSelector = document.getElementById('model-selector');
    const titleInput = document.getElementById('news-title');
    const textInput = document.getElementById('news-text');
    const resultSection = document.getElementById('result-section');
    const loader = document.querySelector('.loader');
    const btnText = document.querySelector('.btn-text');

    // Result Elements
    const statusBadge = document.getElementById('status-badge');
    const confidenceBar = document.getElementById('confidence-bar');
    const confidenceText = document.getElementById('confidence-text');
    const featureTags = document.getElementById('feature-tags');

    analyzeBtn.addEventListener('click', async () => {
        const title = titleInput.value.trim();
        const text = textInput.value.trim();
        const modelType = modelSelector.value;

        if (!title && !text) {
            alert("Please enter a headline or article content.");
            return;
        }

        // UI Loading State
        setLoading(true);
        resultSection.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ title, text, model_type: modelType })
            });

            const data = await response.json();

            if (data.error) {
                alert("Error: " + data.error);
                return;
            }

            displayResults(data);

        } catch (error) {
            console.error("Error:", error);
            alert("Failed to connect to the server.");
        } finally {
            setLoading(false);
        }
    });

    function setLoading(isLoading) {
        if (isLoading) {
            analyzeBtn.disabled = true;
            loader.classList.remove('hidden');
            btnText.classList.add('hidden');
        } else {
            analyzeBtn.disabled = false;
            loader.classList.add('hidden');
            btnText.classList.remove('hidden');
        }
    }

    function displayResults(data) {
        // 1. Update Badge
        statusBadge.textContent = data.label;
        statusBadge.className = `status-badge ${data.label}`; // Adds .REAL or .FAKE class

        // 2. Update Confidence
        const pct = (data.confidence * 100).toFixed(1) + "%";
        confidenceText.textContent = pct;

        // Timeout to allow animation
        confidenceBar.style.width = "0%";
        setTimeout(() => {
            confidenceBar.style.width = pct;
            // Color based on prediction
            confidenceBar.style.backgroundColor = data.label === 'REAL' ? 'var(--real-color)' : 'var(--fake-color)';
        }, 100);

        // 3. Update Features
        featureTags.innerHTML = '';
        if (data.top_features && data.top_features.length > 0) {
            data.top_features.forEach(feat => {
                const tag = document.createElement('div');
                tag.className = 'feature-tag';
                // Show word and its influence (weight)
                tag.textContent = `${feat.word} (${feat.weight.toFixed(2)})`;
                featureTags.appendChild(tag);
            });
        } else {
            featureTags.innerHTML = '<span style="color:var(--text-secondary)">No specific key words found.</span>';
        }

        // Show Section
        resultSection.classList.remove('hidden');
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
});
