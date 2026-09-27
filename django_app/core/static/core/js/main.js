// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const previewSection = document.getElementById('preview-section');
const previewImage = document.getElementById('preview-image');
const analyzeBtn = document.getElementById('analyze-btn');
const cancelBtn = document.getElementById('cancel-btn');
const loadingIndicator = document.getElementById('loading-indicator');
const resultsSection = document.getElementById('results-section');
const newAnalysisBtn = document.getElementById('new-analysis-btn');

let selectedFile = null;

// Event Listeners
dropZone.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-indigo-400', 'bg-indigo-50/50');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-indigo-400', 'bg-indigo-50/50');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-indigo-400', 'bg-indigo-50/50');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
});

analyzeBtn.addEventListener('click', analyzeImage);
cancelBtn.addEventListener('click', resetUpload);
newAnalysisBtn.addEventListener('click', resetUpload);

function handleFile(file) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }

    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        dropZone.classList.add('hidden');
        previewSection.classList.remove('hidden');
    };
    reader.readAsDataURL(file);
}

async function analyzeImage() {
    if (!selectedFile) return;

    previewSection.classList.add('hidden');
    loadingIndicator.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('Error: ' + (data.error || 'Unknown error'));
            resetUpload();
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image: ' + error.message);
        resetUpload();
    }
}

function displayResults(data) {
    loadingIndicator.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    const results = data.results;

    document.getElementById('interpretation-text').textContent = results.interpretation;
    document.getElementById('res-scene-type').textContent = results.primary_scene; // Use primary_scene for more detail
    document.getElementById('res-object-count').textContent = results.object_count;

    // Risk level styling
    const riskEl = document.getElementById('res-risk-level');
    riskEl.textContent = results.risk_level;
    riskEl.className = 'text-xl font-bold uppercase ' + (
        results.risk_level === 'high' ? 'text-rose-200' :
            results.risk_level === 'medium' ? 'text-amber-200' : 'text-emerald-200'
    );

    document.getElementById('res-confidence').textContent = (results.confidence * 100).toFixed(1) + '%';
    document.getElementById('res-original-image').src = '/media/uploads/' + data.filename;
    document.getElementById('res-annotated-image').src = data.annotated_image;

    // Objects list
    const objectsList = document.getElementById('objects-list');
    objectsList.innerHTML = '';
    const counts = {};
    data.detections.forEach(d => counts[d.class_name] = (counts[d.class_name] || 0) + 1);

    Object.entries(counts).forEach(([name, count]) => {
        const span = document.createElement('span');
        span.className = 'px-4 py-2 bg-gray-50 text-gray-700 rounded-xl text-xs font-bold border border-gray-100 shadow-sm capitalize';
        span.innerHTML = `${name} <span class="ml-2 text-indigo-600">${count}</span>`;
        objectsList.appendChild(span);
    });

    // Activities
    const actList = document.getElementById('activities-list');
    actList.innerHTML = '';
    results.activities.forEach(act => {
        const li = document.createElement('li');
        li.className = 'text-sm text-gray-500 font-medium flex items-center gap-3 bg-gray-50 p-3 rounded-xl border border-gray-100';
        li.innerHTML = `<div class="w-2 h-2 rounded-full bg-indigo-400"></div> ${act}`;
        actList.appendChild(li);
    });

    // Use cases
    const useList = document.getElementById('use-cases-list');
    useList.innerHTML = '';
    results.use_cases.forEach(uc => {
        const div = document.createElement('div');
        div.className = 'bg-white/10 border border-white/5 p-4 rounded-xl flex items-center gap-3 hover:bg-white/20 transition-colors';
        div.innerHTML = `<i class="fas fa-check-circle text-indigo-400"></i> <span class="text-sm font-semibold">${uc}</span>`;
        useList.appendChild(div);
    });

    // Performance
    const perfGrid = document.getElementById('performance-metrics');
    perfGrid.innerHTML = '';
    Object.entries(data.performance).forEach(([name, time]) => {
        const div = document.createElement('div');
        div.className = 'bg-indigo-50 md:bg-transparent md:border-l border-indigo-100 p-4 md:pl-6 text-center md:text-left rounded-2xl md:rounded-none';
        div.innerHTML = `<div class="text-[10px] text-gray-400 font-black uppercase tracking-widest mb-1">${name}</div><div class="text-base font-bold text-indigo-600">${time}</div>`;
        perfGrid.appendChild(div);
    });

    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function resetUpload() {
    selectedFile = null;
    fileInput.value = '';
    dropZone.classList.remove('hidden');
    previewSection.classList.add('hidden');
    loadingIndicator.classList.add('hidden');
    resultsSection.classList.add('hidden');
    previewImage.src = '';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
