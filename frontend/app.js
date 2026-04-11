const API = '/api';
let shapChartInst = null;
let triageShapInst = null;
let liveFeatures = {};
let currentPatientContext = null;
let currentHadmId = null;
let chatOpen = false;

// ── TAB SWITCHING ──────────────────────────────────────────────
function switchTab(id, el) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + id).classList.add('active');
    if (el) {
        el.classList.add('active');
    } else {
        const btns = document.querySelectorAll('.nav-btn');
        const map = {home:0, patient:1, triage:2, model:3};
        if (map[id] !== undefined) btns[map[id]].classList.add('active');
    }
    if (id === 'model') loadModelInfo();
}

// ── HOME ───────────────────────────────────────────────────────
async function loadHomeStats() {
    try {
        const d = await fetch(`${API}/metrics`).then(r => r.json());
        document.getElementById('stat-patients').textContent = d.total_patients?.toLocaleString() || '—';
        document.getElementById('stat-rate').textContent = d.readmission_rate_pct ? d.readmission_rate_pct.toFixed(1) + '%' : '—';
        document.getElementById('stat-alerts').textContent = d.ai_alerts_generated?.toLocaleString() || '—';
    } catch(e) { console.error(e); }
}

function searchFromHome() {
    const val = document.getElementById('home-search').value.trim();
    if (!val) return;
    switchTab('patient', document.querySelectorAll('.nav-btn')[1]);
    document.getElementById('inline-search').value = val;
    loadPatient(val);
}

async function pickRandomPatient() {
    try {
        const d = await fetch(`${API}/patients`).then(r => r.json());
        const ids = d.hadm_ids;
        const id = ids[Math.floor(Math.random() * ids.length)];
        document.getElementById('home-search').value = id;
        searchFromHome();
    } catch(e) { console.error(e); }
}

// ── PATIENT ANALYSIS ───────────────────────────────────────────
function searchInline() {
    const val = document.getElementById('inline-search').value.trim();
    if (val) loadPatient(val);
}

async function loadPatient(hadmId) {
    // Reset chat state for new patient
    currentPatientContext = null;
    currentHadmId = null;
    document.getElementById('chat-bubble-btn').style.display = 'none';
    document.getElementById('chat-messages').innerHTML = '';
    document.getElementById('chat-quick-questions').style.display = 'flex';
    if (chatOpen) toggleChat();

    document.getElementById('patient-empty').style.display = 'none';
    document.getElementById('patient-results').style.display = 'none';
    document.getElementById('patient-loading').style.display = 'flex';

    try {
        const data = await fetch(`${API}/patient/${hadmId}`).then(r => {
            if (!r.ok) throw new Error('Patient not found');
            return r.json();
        });
        renderPatient(data);
    } catch(e) {
        document.getElementById('patient-loading').style.display = 'none';
        document.getElementById('patient-empty').style.display = 'flex';
        const stateText = document.getElementById('patient-empty').querySelector('.state-text');
        if (stateText) stateText.textContent = e.message || 'Patient not found';
    }
}

function renderPatient(data) {
    document.getElementById('patient-loading').style.display = 'none';
    document.getElementById('patient-results').style.display = 'block';

    const llm = data.llm_insights || {};
    const risk = parseFloat(llm.predicted_risk_pct || 0);
    const level = (llm.risk_level || 'UNKNOWN').toUpperCase().trim();

    // HADM tag
    document.getElementById('res-hadm').textContent = 'HADM ' + data.hadm_id;

    // Gauge
    updateGauge(risk, level);

    // Risk badge
    const rb = document.getElementById('risk-badge-lg');
    rb.textContent = level;
    rb.className = 'risk-badge-lg ' + (level === 'HIGH' ? 'high' : level === 'MEDIUM' ? 'med' : level === 'LOW' ? 'low' : 'unknown');


    // SHAP chart
    if (shapChartInst) { shapChartInst.destroy(); shapChartInst = null; }
    renderShap('shap-chart', data.shap_explanation?.shap_waterfall, inst => { shapChartInst = inst; });

    // Mental health burden
    renderMentalHealth(data.mental_health_burden || {});

    // Clinical context + RAG source
    document.getElementById('clinical-context').textContent = data.original_summary || 'No clinical summary available.';
    const ragBadge = document.getElementById('rag-source-badge');
    if (data.rag_source === 'retrieved') {
        ragBadge.className = 'rag-badge retrieved';
        ragBadge.innerHTML = '<i data-lucide="file-search"></i> Retrieved';
    } else {
        ragBadge.className = 'rag-badge synthesized';
        ragBadge.innerHTML = '<i data-lucide="sparkles"></i> Synthesized';
    }

    // ── DOCTOR VIEW ──────────────────────────────────────────────
    const strip = document.getElementById('alert-strip');
    strip.className = 'alert-strip' + (level === 'MEDIUM' ? ' med' : level === 'LOW' ? ' low' : '');
    document.getElementById('alert-level-tag').textContent = level + ' RISK';
    document.getElementById('alert-text').textContent = llm.doctor_alert?.risk_summary || '—';

    // Doctor precautions — clinical language; fall back to patient_precautions for old records
    const doctorPrecs = llm.doctor_precautions?.length ? llm.doctor_precautions : llm.patient_precautions || [];
    const dp = document.getElementById('doctor-prec');
    dp.innerHTML = '';
    doctorPrecs.forEach(p => {
        const li = document.createElement('li');
        li.textContent = p;
        dp.appendChild(li);
    });
    document.getElementById('followup-box').textContent = llm.follow_up_recommendations || '—';

    // ── PATIENT VIEW ─────────────────────────────────────────────
    // Update banner color based on risk
    const banner = document.getElementById('patient-banner');
    banner.className = 'patient-banner ' + (level === 'HIGH' ? 'high' : level === 'MEDIUM' ? 'med' : 'low');

    // Patient-friendly health summary — always plain language, never clinical
    const riskLabel = level === 'HIGH' ? 'HIGH' : level === 'MEDIUM' ? 'MODERATE' : 'LOW';
    const riskColor = level === 'HIGH' ? '#ef4444' : level === 'MEDIUM' ? '#f59e0b' : '#10b981';
    const patientIntro = level === 'HIGH'
        ? `Your doctor has identified a HIGH risk that you may need to return to the hospital within 30 days. Please follow these steps carefully — they are important for your recovery.`
        : level === 'MEDIUM'
        ? `Your doctor has identified a MODERATE risk of returning to the hospital. Following these steps will help keep you safe and healthy at home.`
        : `Your readmission risk is LOW. Keep following these steps to stay healthy and avoid any complications.`;
    document.getElementById('patient-summary-text').textContent = patientIntro;
    document.getElementById('patient-risk-pct').textContent = risk.toFixed(1) + '%';
    document.getElementById('patient-risk-pct').style.color = riskColor;

    const ps = document.getElementById('patient-steps');
    ps.innerHTML = '';
    (llm.patient_precautions || []).forEach((p, i) => {
        const div = document.createElement('div');
        div.className = 'patient-step-card';
        div.innerHTML = '<div class="step-num-badge">STEP ' + (i + 1) + '</div><div class="step-text">' + p + '</div>';
        ps.appendChild(div);
    });

    // Patient follow-up — patient-friendly language, separate from doctor follow-up
    const followupEl = document.getElementById('patient-followup');
    if (followupEl) {
        const fu = llm.patient_follow_up || llm.follow_up_recommendations || '';
        followupEl.textContent = fu || 'Your care team will contact you to schedule a follow-up appointment. Please attend all scheduled visits and call your doctor if symptoms worsen.';
    }

    // Default to doctor view
    setView('doctor');

    // Build patient context for chat
    currentHadmId = String(data.hadm_id);
    const shapWaterfall = data.shap_explanation?.shap_waterfall || [];
    const shap_top3 = shapWaterfall.slice(0, 3).map(s => s.feature.replace(/_/g, ' '));
    currentPatientContext = {
        risk_level: level,
        risk_pct: risk,
        precautions: llm.patient_precautions || [],
        shap_top3: shap_top3,
        follow_up: llm.follow_up_recommendations || ''
    };
    document.getElementById('chat-bubble-btn').style.display = 'flex';

    lucide.createIcons();
}

function updateGauge(pct, level) {
    const totalLen = 251.3;
    const filled = (pct / 100) * totalLen;
    const color = level === 'HIGH' ? '#ef4444' : level === 'MEDIUM' ? '#f59e0b' : '#10b981';
    const arc = document.getElementById('gauge-arc');
    if (!arc) return;
    arc.setAttribute('stroke-dasharray', filled + ' ' + totalLen);
    arc.setAttribute('stroke', color);
    document.getElementById('gauge-pct').textContent = pct.toFixed(1) + '%';
    const lbl = document.getElementById('gauge-label');
    lbl.textContent = level + ' RISK';
    lbl.setAttribute('fill', color);
}

function renderMentalHealth(mh) {
    const badge = document.getElementById('mh-badge');
    badge.textContent = mh.level || 'LOW';
    badge.className = 'mh-badge ' + (mh.level || 'LOW');

    const signals = document.getElementById('mh-signals');
    signals.innerHTML = '';
    (mh.signals || []).forEach(s => {
        const li = document.createElement('li');
        li.textContent = s;
        signals.appendChild(li);
    });

    const pct = ((mh.score || 0) / (mh.max_score || 3)) * 100;
    const fill = document.getElementById('mh-fill');
    fill.style.width = pct + '%';
    fill.style.background = mh.level === 'HIGH' ? '#ef4444' : mh.level === 'MEDIUM' ? '#f59e0b' : '#10b981';
}

function setView(view) {
    document.getElementById('doctor-view').style.display = view === 'doctor' ? 'block' : 'none';
    document.getElementById('patient-view').style.display = view === 'patient' ? 'block' : 'none';
    document.getElementById('tog-doctor').classList.toggle('active', view === 'doctor');
    document.getElementById('tog-patient').classList.toggle('active', view === 'patient');
}

// ── SHAP CHART ─────────────────────────────────────────────────
function renderShap(canvasId, shapData, setInst) {
    if (!shapData || !shapData.length) return;
    const top = shapData.slice(0, 10);
    const labels = top.map(x =>
        x.feature.replace(/_/g, ' ')
                 .replace(/\b\w/g, c => c.toUpperCase())
    );
    const vals = top.map(x => x.value);
    const colors = vals.map(v => v > 0 ? 'rgba(239,68,68,0.75)' : 'rgba(16,185,129,0.75)');
    const borders = vals.map(v => v > 0 ? '#ef4444' : '#10b981');
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    const chart = new Chart(ctx.getContext('2d'), {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                data: vals,
                backgroundColor: colors,
                borderColor: borders,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {display:true, text:'SHAP Value (impact on risk)', color:'#71717a', font:{size:11}},
                    ticks: {color:'#71717a'},
                    grid: {color:'rgba(255,255,255,0.04)'},
                    border: {color:'rgba(255,255,255,0.06)'}
                },
                y: {
                    ticks: {color:'#a1a1aa', font:{size:11}},
                    grid: {display:false}
                }
            },
            plugins: {
                legend: {display:false},
                tooltip: {
                    callbacks: {
                        label: ctx => ' ' + (ctx.parsed.x > 0 ? '+' : '') + ctx.parsed.x.toFixed(4)
                    }
                }
            }
        }
    });
    setInst(chart);
}

// ── LIVE TRIAGE ────────────────────────────────────────────────
async function loadTriageBaseline() {
    try {
        liveFeatures = await fetch(`${API}/triage/baseline`).then(r => r.json());
        const cont = document.getElementById('triage-features-container');
        const label = document.createElement('p');
        label.style.cssText = 'color:#71717a;font-size:11px;margin:0 0 8px 0;';
        label.textContent = 'Showing top features by model importance — changes here directly affect the risk score and SHAP chart.';
        const table = document.createElement('table');
        table.className = 'triage-table';
        table.innerHTML = '<thead><tr><th>Feature</th><th>Value</th></tr></thead>';
        const tbody = document.createElement('tbody');
        let count = 0;
        for (const [k, v] of Object.entries(liveFeatures)) {
            if (count >= 20) break;
            if (typeof v === 'number') {
                const tr = document.createElement('tr');
                tr.innerHTML = '<td>' + k.replace(/_/g, ' ') + '</td><td><input type="number" step="any" id="tf-' + k + '" data-key="' + k + '" value="' + v.toFixed(2) + '"></td>';
                tbody.appendChild(tr);
                count++;
            }
        }
        table.appendChild(tbody);
        cont.innerHTML = '';
        cont.appendChild(label);
        cont.appendChild(table);
    } catch(e) { console.error(e); }
}

async function runLiveTriage() {
    const btn = document.getElementById('btn-triage');
    btn.disabled = true;
    btn.innerHTML = '<i data-lucide="loader-2"></i> Analyzing...';
    lucide.createIcons();

    document.getElementById('triage-waiting').style.display = 'none';
    document.getElementById('triage-loading').style.display = 'block';
    document.getElementById('triage-output').style.display = 'none';

    document.querySelectorAll('input[id^="tf-"]').forEach(inp => {
        liveFeatures[inp.getAttribute('data-key')] = parseFloat(inp.value);
    });

    const payload = {
        doctors_note: document.getElementById('triage-note').value,
        features: liveFeatures
    };

    try {
        const data = await fetch(`${API}/triage/live`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        }).then(r => r.json());

        document.getElementById('triage-loading').style.display = 'none';
        document.getElementById('triage-output').style.display = 'block';

        const risk = parseFloat(data.risk_probability_pct || 0);
        const level = risk >= 60 ? 'HIGH' : risk >= 30 ? 'MEDIUM' : 'LOW';

        document.getElementById('triage-big-score').textContent = risk.toFixed(1) + '%';
        document.getElementById('triage-big-score').style.color = level === 'HIGH' ? '#ef4444' : level === 'MEDIUM' ? '#f59e0b' : '#10b981';

        const tb = document.getElementById('triage-risk-badge');
        tb.textContent = level + ' RISK';
        tb.className = 'triage-risk-badge ' + (level === 'HIGH' ? 'high' : level === 'MEDIUM' ? 'med' : 'low');

        const g = data.gemini_insights || {};
        const alertLevel = ((g.doctor_alert?.risk_level) || level).toUpperCase();
        const strip = document.getElementById('triage-alert-strip');
        strip.className = 'alert-strip' + (alertLevel === 'MEDIUM' ? ' med' : alertLevel === 'LOW' ? ' low' : '');
        document.getElementById('triage-alert-level').textContent = alertLevel + ' RISK';
        document.getElementById('triage-alert-text').textContent = g.doctor_alert?.risk_summary || g.error || '—';

        const tp = document.getElementById('triage-prec');
        tp.innerHTML = '';
        (g.patient_precautions || []).forEach(p => {
            const li = document.createElement('li');
            li.textContent = p;
            tp.appendChild(li);
        });

        // Follow-up recommendations
        const triageFollowup = document.getElementById('triage-followup');
        if (triageFollowup) triageFollowup.textContent = g.follow_up_recommendations || '—';

        if (triageShapInst) { triageShapInst.destroy(); triageShapInst = null; }
        renderShap('triage-shap-chart', data.shap_waterfall, inst => { triageShapInst = inst; });
        lucide.createIcons();

    } catch(e) {
        document.getElementById('triage-loading').style.display = 'none';
        document.getElementById('triage-waiting').style.display = 'flex';
        console.error(e);
    }

    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="zap"></i> Analyze Live Patient';
    lucide.createIcons();
}

// ── MODEL INFO ─────────────────────────────────────────────────
async function loadModelInfo() {
    try {
        const d = await fetch(`${API}/model/info`).then(r => r.json());
        document.getElementById('model-name').textContent = d.model_type || '—';
        document.getElementById('model-class').textContent = d.model_class || '—';
        document.getElementById('kpi-features').textContent = d.feature_count || '—';
        document.getElementById('model-pipeline').textContent = d.pipeline || '—';
        document.getElementById('model-med-count').textContent = (d.medication_features?.length || 0) + ' medication features';

        const grid = document.getElementById('feat-grid');
        const countTag = document.getElementById('feat-count-tag');
        if (d.selected_features) {
            grid.innerHTML = '';
            countTag.textContent = d.selected_features.length;
            const medKw = ['loop','ace_arb','beta','aldo','digoxin','anticoag','unique_drugs','furosemide','gdmt'];
            d.selected_features.forEach(f => {
                const isMed = medKw.some(k => f.includes(k));
                const span = document.createElement('span');
                span.className = 'feat-tag ' + (isMed ? 'med' : 'clin');
                span.textContent = f.replace(/_/g, ' ');
                span.title = isMed ? 'Medication feature' : 'Clinical/lab feature';
                grid.appendChild(span);
            });
        }
    } catch(e) { console.error(e); }
}

// ── CHAT WIDGET ────────────────────────────────────────────────
function toggleChat() {
    chatOpen = !chatOpen;
    const panel = document.getElementById('chat-panel');
    panel.style.display = chatOpen ? 'flex' : 'none';
    if (chatOpen) {
        const msgs = document.getElementById('chat-messages');
        msgs.scrollTop = msgs.scrollHeight;
        document.getElementById('chat-input').focus();
        // Welcome message on first open
        if (msgs.children.length === 0) {
            appendChatMessage('bot', `Hi! I'm your health assistant. I can help you understand your ${currentPatientContext?.risk_level || ''} readmission risk. Feel free to ask me anything or use the quick questions below.`);
        }
    }
}

function sendQuickQuestion(question) {
    document.getElementById('chat-input').value = question;
    document.getElementById('chat-quick-questions').style.display = 'none';
    sendChatMessage();
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    if (!message || !currentPatientContext) return;

    input.value = '';
    const sendBtn = document.getElementById('chat-send-btn');
    sendBtn.disabled = true;

    appendChatMessage('user', message);

    const typingId = appendTypingIndicator();

    try {
        const res = await fetch(`${API}/chat`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                hadm_id: currentHadmId,
                message: message,
                patient_context: currentPatientContext
            })
        });

        removeTypingIndicator(typingId);

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            appendChatMessage('bot', err.detail || 'Sorry, I could not get a response. Please try again.');
        } else {
            const data = await res.json();
            appendChatMessage('bot', data.reply || 'Sorry, I did not receive a response.');
        }
    } catch(e) {
        removeTypingIndicator(typingId);
        appendChatMessage('bot', 'Connection error. Please check your network and try again.');
    }

    sendBtn.disabled = false;
    input.focus();
}

function appendChatMessage(role, text) {
    const msgs = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = 'chat-msg ' + (role === 'user' ? 'chat-msg-user' : 'chat-msg-bot');
    div.textContent = text;
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return div;
}

function appendTypingIndicator() {
    const msgs = document.getElementById('chat-messages');
    const div = document.createElement('div');
    const id = 'typing-' + Date.now();
    div.id = id;
    div.className = 'chat-msg chat-msg-bot chat-typing';
    div.innerHTML = '<span></span><span></span><span></span>';
    msgs.appendChild(div);
    msgs.scrollTop = msgs.scrollHeight;
    return id;
}

function removeTypingIndicator(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ── INIT ───────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadHomeStats();
    loadTriageBaseline();
    lucide.createIcons();
});
