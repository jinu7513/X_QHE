document.getElementById('analyze-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    const fileInput = document.getElementById('file-input');
    const targetInput = document.getElementById('target-columns-input');
    const xInput = document.getElementById('x-column-input');
    const errorDiv = document.getElementById('error-message');
    const loadingDiv = document.getElementById('loading');
    const resultsSection = document.getElementById('results-section');
    const resultsContainer = document.getElementById('results-container');
    const submitBtn = document.getElementById('submit-btn');

    if (!fileInput.files[0]) {
        errorDiv.textContent = "Please select a file first.";
        errorDiv.style.display = 'block';
        return;
    }

    errorDiv.style.display = 'none';
    resultsSection.style.display = 'none';
    loadingDiv.style.display = 'block';
    submitBtn.disabled = true;
    submitBtn.textContent = 'Analyzing...';

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    if (targetInput.value) formData.append('target_columns', targetInput.value);
    if (xInput.value) formData.append('x_column', xInput.value);

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Server error occurred");
        }

        renderResults(data.results, resultsContainer);
        resultsSection.style.display = 'block';
    } catch (err) {
        errorDiv.textContent = err.message;
        errorDiv.style.display = 'block';
    } finally {
        loadingDiv.style.display = 'none';
        submitBtn.disabled = false;
        submitBtn.textContent = 'Analyze Data';
    }
});

function renderResults(results, container) {
    container.innerHTML = '';
    if (Object.keys(results).length === 0) {
        container.innerHTML = '<p>No results to display.</p>';
        return;
    }

    for (const [columnName, columnData] of Object.entries(results)) {
        const targetDiv = document.createElement('div');
        targetDiv.style.marginBottom = '2rem';

        const header = document.createElement('h3');
        header.style.marginBottom = '1rem';
        header.style.borderBottom = '1px solid rgba(255,255,255,0.1)';
        header.style.paddingBottom = '0.5rem';
        header.textContent = `Target: ${columnName}`;
        targetDiv.appendChild(header);

        if (columnData.error) {
            const err = document.createElement('div');
            err.className = 'error-message';
            err.textContent = columnData.error;
            targetDiv.appendChild(err);
        } else {
            if (columnData.plot) {
                targetDiv.appendChild(createRegressionVisualization(columnName, columnData.plot));
            }

            const grid = document.createElement('div');
            grid.className = 'results-grid';

            grid.appendChild(createMethodResult('Classical', columnData.classical));
            grid.appendChild(createMethodResult('Quantum HHL', columnData.hhl));
            grid.appendChild(createMethodResult('HE + HHL', columnData.he_hhl));

            targetDiv.appendChild(grid);
        }

        container.appendChild(targetDiv);
    }
}

function createMethodResult(title, data) {
    const card = document.createElement('div');
    card.className = 'result-card';
    if (!data || data.error) return card;

    const h3 = document.createElement('h3');
    h3.textContent = title;
    card.appendChild(h3);

    if (data.mean_y !== undefined && !Number.isNaN(data.mean_y)) card.appendChild(createMetric('Mean (y)', data.mean_y));
    if (data.var_y !== undefined && !Number.isNaN(data.var_y)) card.appendChild(createMetric('Variance (y)', data.var_y));
    card.appendChild(createMetric('Intercept', data.intercept));
    card.appendChild(createMetric('Slope', data.slope));
    card.appendChild(createMetric('R²', data.r2));
    if (data.encrypted_preview) card.appendChild(createEncryptedPreview(data.encrypted_preview));

    return card;
}

function createMetric(label, value) {
    const div = document.createElement('div');
    div.className = 'metric';
    
    const spanLabel = document.createElement('span');
    spanLabel.className = 'label';
    spanLabel.textContent = label;

    const spanValue = document.createElement('span');
    spanValue.className = 'value';
    spanValue.textContent = typeof value === 'number' ? value.toFixed(6) : value;

    div.appendChild(spanLabel);
    div.appendChild(spanValue);
    return div;
}

function createRegressionVisualization(columnName, plot) {
    const section = document.createElement('div');
    section.className = 'regression-visualization';

    const title = document.createElement('h3');
    title.textContent = 'Actual Data vs Regression Lines';
    section.appendChild(title);

    const canvas = document.createElement('canvas');
    canvas.className = 'chart-canvas';
    canvas.setAttribute('aria-label', `Regression chart for ${columnName}`);
    section.appendChild(canvas);

    const legend = document.createElement('div');
    legend.className = 'chart-legend';
    [
        ['Actual data', '#f8fafc'],
        ['Classical', '#38bdf8'],
        ['Quantum HHL', '#a78bfa'],
        ['HE + HHL', '#f59e0b'],
    ].forEach(([label, color]) => {
        const item = document.createElement('span');
        item.className = 'legend-item';
        const swatch = document.createElement('span');
        swatch.className = 'legend-swatch';
        swatch.style.background = color;
        item.appendChild(swatch);
        item.appendChild(document.createTextNode(label));
        legend.appendChild(item);
    });
    section.appendChild(legend);

    requestAnimationFrame(() => drawRegressionChart(canvas, plot));
    window.addEventListener('resize', () => drawRegressionChart(canvas, plot));

    return section;
}

function drawRegressionChart(canvas, plot) {
    const points = plot.points || [];
    if (!points.length) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(rect.width, 320);
    const height = 360;
    const dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.height = `${height}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const lineSeries = Object.entries(plot.lines || {});
    const allX = points.map((point) => point.x);
    const allY = points.map((point) => point.y);
    lineSeries.forEach(([, series]) => {
        (series.points || []).forEach((point) => {
            allX.push(point.x);
            allY.push(point.y);
        });
    });

    let minX = Math.min(...allX);
    let maxX = Math.max(...allX);
    let minY = Math.min(...allY);
    let maxY = Math.max(...allY);

    const xRange = maxX - minX || 1;
    const yRange = maxY - minY || 1;
    minX -= xRange * 0.04;
    maxX += xRange * 0.04;
    minY -= yRange * 0.12;
    maxY += yRange * 0.12;

    const padding = { top: 26, right: 24, bottom: 48, left: 72 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const toCanvasX = (x) => padding.left + ((x - minX) / (maxX - minX)) * plotWidth;
    const toCanvasY = (y) => padding.top + (1 - (y - minY) / (maxY - minY)) * plotHeight;

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = 'rgba(15, 23, 42, 0.62)';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = 'rgba(148, 163, 184, 0.22)';
    ctx.lineWidth = 1;
    ctx.font = '12px Inter, sans-serif';
    ctx.fillStyle = '#94a3b8';

    for (let i = 0; i <= 4; i += 1) {
        const x = padding.left + (plotWidth / 4) * i;
        const y = padding.top + (plotHeight / 4) * i;

        ctx.beginPath();
        ctx.moveTo(x, padding.top);
        ctx.lineTo(x, padding.top + plotHeight);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(padding.left + plotWidth, y);
        ctx.stroke();

        const yValue = maxY - ((maxY - minY) / 4) * i;
        ctx.fillText(formatCompactNumber(yValue), 12, y + 4);
    }

    ctx.strokeStyle = 'rgba(248, 250, 252, 0.42)';
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    const firstPoint = points[0];
    const lastPoint = points[points.length - 1];
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText(firstPoint.x_label || formatCompactNumber(firstPoint.x), padding.left, height - 18);
    ctx.textAlign = 'right';
    ctx.fillText(lastPoint.x_label || formatCompactNumber(lastPoint.x), padding.left + plotWidth, height - 18);
    ctx.textAlign = 'start';

    const lineStyles = {
        classical: '#38bdf8',
        hhl: '#a78bfa',
        he_hhl: '#f59e0b',
    };

    lineSeries.forEach(([key, series]) => {
        const seriesPoints = series.points || [];
        if (seriesPoints.length < 2) return;

        ctx.strokeStyle = lineStyles[key] || '#e2e8f0';
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        seriesPoints.forEach((point, index) => {
            const x = toCanvasX(point.x);
            const y = toCanvasY(point.y);
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        ctx.stroke();
    });

    ctx.fillStyle = '#f8fafc';
    ctx.strokeStyle = 'rgba(15, 23, 42, 0.9)';
    points.forEach((point) => {
        const x = toCanvasX(point.x);
        const y = toCanvasY(point.y);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
}

function formatCompactNumber(value) {
    if (typeof value !== 'number' || Number.isNaN(value)) return '-';
    if (Math.abs(value) >= 100000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
        return value.toExponential(3);
    }
    return value.toLocaleString(undefined, {
        maximumFractionDigits: 4,
    });
}

function createEncryptedPreview(preview) {
    const section = document.createElement('div');
    section.className = 'encrypted-preview';

    const details = document.createElement('details');

    const summary = document.createElement('summary');
    summary.textContent = 'CKKS Encrypted Data';
    details.appendChild(summary);

    const groups = [
        ['Input vectors', preview.vectors || []],
        ['Encrypted aggregates', preview.aggregates || []],
    ];

    groups.forEach(([title, items]) => {
        if (!items.length) return;

        const groupTitle = document.createElement('h4');
        groupTitle.textContent = title;
        details.appendChild(groupTitle);

        items.forEach((item) => {
            details.appendChild(createCiphertextItem(item));
        });
    });

    section.appendChild(details);
    return section;
}

function createCiphertextItem(item) {
    const wrapper = document.createElement('div');
    wrapper.className = 'ciphertext-item';

    const title = document.createElement('div');
    title.className = 'ciphertext-title';
    title.textContent = item.name;
    wrapper.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'ciphertext-meta';
    meta.textContent = `${item.scheme} | ${item.value_count} value(s) | ${item.ciphertext_bytes} bytes | sha256 ${item.sha256.slice(0, 12)}`;
    wrapper.appendChild(meta);

    const snippet = document.createElement('code');
    snippet.className = 'ciphertext-snippet';
    snippet.textContent = `${item.base64_preview}${item.truncated ? '...' : ''}`;
    wrapper.appendChild(snippet);

    return wrapper;
}
