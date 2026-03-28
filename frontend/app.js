// Constants
const API_BASE = '/api';
let liveTriageBaselineFeatures = {};

// ==========================================
// NAVIGATION LOGIC
// ==========================================
function switchTab(tabId) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    // Un-highlight buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show active tab
    document.getElementById(`tab-${tabId}`).classList.add('active');
    // Highlight active button mapped to tab argument
    event.currentTarget.classList.add('active');
    
    // Load Dashboard metrics smoothly
    if (tabId === 'dashboard') {
        fetchDashboardMetrics();
    }
}

// ==========================================
// DASHBOARD LOGIC
// ==========================================
async function fetchDashboardMetrics() {
    try {
        const res = await fetch(`${API_BASE}/metrics`);
        const data = await res.json();
        document.getElementById('val-patients').innerText = data.total_patients.toLocaleString();
        document.getElementById('val-admissions').innerText = data.total_admissions.toLocaleString();
        document.getElementById('val-rate').innerText = data.readmission_rate_pct.toFixed(1) + '%';
    } catch (e) {
        console.error("Failed to load metrics", e);
    }
}

// ==========================================
// HISTORICAL LOOKUP LOGIC
// ==========================================
async function populateHistoricalDropdown() {
    try {
        const res = await fetch(`${API_BASE}/patients`);
        const data = await res.json();
        const select = document.getElementById('patientSelect');
        data.hadm_ids.forEach(id => {
            const opt = document.createElement('option');
            opt.value = id;
            opt.innerText = id;
            select.appendChild(opt);
        });
    } catch(e) { console.error(e); }
}

let histShapChartInstance = null;

async function loadHistoricalPatient() {
    const hadmId = document.getElementById('patientSelect').value;
    if (!hadmId) return;
    
    document.getElementById('historical-results').style.display = 'grid';
    
    try {
        const res = await fetch(`${API_BASE}/patient/${hadmId}`);
        const data = await res.json();
        
        const llm = data.llm_insights;
        const alertLvl = llm.risk_level || "UNKNOWN";
        let riskPct = llm.predicted_risk_pct || "--";
        if (typeof riskPct === 'number') riskPct = riskPct.toFixed(1);
        const alertBox = document.getElementById('hist-risk-badge');
        
        // CSS alert assignment
        alertBox.className = 'alert-box ' + (alertLvl.includes("HIGH") ? "high-risk" : alertLvl.includes("MEDIUM") ? "med-risk" : "low-risk");
        if(alertLvl.includes("HIGH")) alertBox.style.borderColor = 'var(--risk-high)';
        if(alertLvl.includes("LOW")) alertBox.style.borderColor = 'var(--risk-low)';
        if(alertLvl.includes("MEDIUM")) alertBox.style.borderColor = 'var(--risk-med)';

        document.getElementById('hist-risk-level').innerText = `${alertLvl} (${riskPct}%)`;
        document.getElementById('hist-doctor-summary').innerText = llm.doctor_alert?.risk_summary || '';
        
        const ul = document.getElementById('hist-precautions');
        ul.innerHTML = '';
        (llm.patient_precautions || []).forEach(p => {
            const li = document.createElement('li');
            li.innerText = p;
            ul.appendChild(li);
        });
        
        document.getElementById('hist-followup').innerText = llm.follow_up_recommendations || '';
        document.getElementById('hist-original-summary').innerText = data.original_summary;
        
        // Render SHAP
        renderShapChart('hist-shap-chart', data.shap_explanation.shap_waterfall, histShapChartInstance, (chart) => { histShapChartInstance = chart; });
        
    } catch(e) {
        alert("Failed to load data.");
        console.error(e);
    }
}

// ==========================================
// LIVE TRIAGE LOGIC
// ==========================================
async function loadBaselineTriageData() {
    try {
        const res = await fetch(`${API_BASE}/triage/baseline`);
        liveTriageBaselineFeatures = await res.json(); // Dict [feature -> value]
        
        const container = document.getElementById('triage-features-container');
        container.innerHTML = '';
        
        const table = document.createElement('table');
        table.className = 'triage-table';
        
        // Render only top 15 features to save UI space
        let count = 0;
        for (const [feat, val] of Object.entries(liveTriageBaselineFeatures)) {
            if(count > 15) break; 
            if(typeof val === 'number') {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>${feat}</td>
                    <td><input type="number" step="any" id="livetriage-${feat}" data-key="${feat}" value="${val.toFixed(2)}"></td>
                `;
                table.appendChild(tr);
                count++;
            }
        }
        container.appendChild(table);
        
    } catch(e) { console.error(e); }
}

let triageShapChartInstance = null;

async function runLiveTriage() {
    const btn = document.getElementById('btn-run-triage');
    btn.disabled = true;
    btn.innerText = "Processing LLM Inference...";
    
    document.getElementById('triage-waiting-state').style.display = 'none';
    document.getElementById('triage-output-content').style.display = 'none';
    document.getElementById('triage-loader').style.display = 'block';
    document.getElementById('triage-results').style.opacity = '1';

    // Harvest inputs
    const note = document.getElementById('triage-note').value;
    const inputs = document.querySelectorAll('input[id^="livetriage-"]');
    inputs.forEach(inp => {
        liveTriageBaselineFeatures[inp.getAttribute('data-key')] = parseFloat(inp.value);
    });

    const payload = { doctors_note: note, features: liveTriageBaselineFeatures };

    try {
        const res = await fetch(`${API_BASE}/triage/live`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        document.getElementById('triage-loader').style.display = 'none';
        document.getElementById('triage-output-content').style.display = 'block';
        
        document.getElementById('triage-risk-pct').innerText = data.risk_probability_pct.toFixed(1) + '%';
        
        const g = data.gemini_insights || {};
        document.getElementById('triage-doctor-summary').innerText = g.doctor_alert?.risk_summary || g.error || 'N/A';
        
        const ul = document.getElementById('triage-precautions');
        ul.innerHTML = '';
        (g.patient_precautions || []).forEach(p => {
            const li = document.createElement('li');
            li.innerText = p;
            ul.appendChild(li);
        });

        // Render SHAP
        renderShapChart('triage-shap-chart', data.shap_waterfall, triageShapChartInstance, (chart) => { triageShapChartInstance = chart; });

    } catch(e) {
        alert("LLM Processing Failed");
        document.getElementById('triage-loader').style.display = 'none';
        document.getElementById('triage-waiting-state').style.display = 'block';
    }
    
    btn.disabled = false;
    btn.innerText = "🚀 Analyze Live Patient";
}

// ==========================================
// SHAP CHART RENDERING
// ==========================================
function renderShapChart(canvasId, shapDataList, existingChart, setChartCallback) {
    if(!shapDataList) return;
    
    // Take top 10 impacts
    const top = shapDataList.slice(0, 10);
    const labels = top.map(x => x.feature);
    const data = top.map(x => x.value);
    
    const colors = data.map(v => v > 0 ? 'rgba(239, 68, 68, 0.8)' : 'rgba(16, 185, 129, 0.8)'); // Red positive, Green negative

    if (existingChart) existingChart.destroy();
    
    const ctx = document.getElementById(canvasId).getContext('2d');
    const newChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{ label: 'SHAP Value (Impact on Risk)', data: data, backgroundColor: colors }]
        },
        options: {
            indexAxis: 'y', // horizontal bar chart perfectly mimics SHAP summary
            responsive: true,
            scales: {
                x: { title: {display: true, text: 'SHAP Value (Log Odds)', color: '#fff'}, ticks: {color: '#aaa'}, grid:{color:'rgba(255,255,255,0.05)'} },
                y: { ticks: {color: '#fff'}, grid:{display:false} }
            },
            plugins: { legend: { display: false } }
        }
    });
    setChartCallback(newChart);
}

// ==========================================
// INITIALIZATION
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    fetchDashboardMetrics();
    populateHistoricalDropdown();
    loadBaselineTriageData();
});
