const form = document.getElementById('predictionForm');
const tabs = [...document.querySelectorAll('.tab')];
const panels = [...document.querySelectorAll('.tab-content')];
const numericFields = ['age','bp','sg','al','su','bgr','bu','sc','sod','pot','hemo','pcv','wbcc','rbcc'];
const categoricalFields = ['rbc','pc','pcc','ba','htn','dm','cad','appet','pe','ane'];
const allFields = [...numericFields, ...categoricalFields];
let activeTab = 0;

function showTab(index) {
  activeTab = Math.max(0, Math.min(3, index));
  tabs.forEach((tab, i) => {
    const active = i === activeTab;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
  });
  panels.forEach((panel, i) => {
    panel.hidden = i !== activeTab;
    panel.classList.toggle('active', i === activeTab);
  });
  document.getElementById('backTab').disabled = activeTab === 0;
  document.getElementById('nextTab').textContent = activeTab === 3 ? 'Last Tab' : 'Next Tab →';
}
tabs.forEach((tab, i) => tab.addEventListener('click', () => showTab(i)));
document.getElementById('backTab').addEventListener('click', () => showTab(activeTab - 1));
document.getElementById('nextTab').addEventListener('click', () => showTab(activeTab + 1));

function updateConfigured() {
  const count = allFields.filter(id => document.getElementById(id).value.trim() !== '').length;
  document.getElementById('configuredCount').textContent = `${count} / 24 Configured`;
  document.getElementById('configuredBar').style.width = `${count / 24 * 100}%`;
}
allFields.forEach(id => {
  const el = document.getElementById(id);
  el.addEventListener('input', updateConfigured);
  el.addEventListener('change', updateConfigured);
});

const samples = {
  ckd: {age:48,bp:80,sg:1.020,al:3,su:0,bgr:121,bu:36,sc:1.2,sod:138,pot:4.5,hemo:11.2,pcv:32,wbcc:7800,rbcc:3.8,rbc:'abnormal',pc:'abnormal',pcc:'present',ba:'notpresent',htn:'yes',dm:'yes',cad:'no',appet:'poor',pe:'yes',ane:'yes'},
  healthy: {age:35,bp:70,sg:1.020,al:0,su:0,bgr:99,bu:15,sc:0.8,sod:140,pot:4.2,hemo:15,pcv:48,wbcc:7500,rbcc:5.2,rbc:'normal',pc:'normal',pcc:'notpresent',ba:'notpresent',htn:'no',dm:'no',cad:'no',appet:'good',pe:'no',ane:'no'}
};
document.getElementById('sampleCase').addEventListener('change', event => {
  const sample = samples[event.target.value];
  if (!sample) return;
  Object.entries(sample).forEach(([id, value]) => { document.getElementById(id).value = value; });
  updateConfigured();
});

function collectData() {
  const data = {};
  numericFields.forEach(id => { data[id] = Number(document.getElementById(id).value); });
  categoricalFields.forEach(id => { data[id] = document.getElementById(id).value; });
  return data;
}
function validateAllFields() {
  for (let i = 0; i < panels.length; i++) {
    const required = [...panels[i].querySelectorAll('input[required],select[required]')];
    const invalid = required.find(el => !el.value || !el.checkValidity());
    if (invalid) {
      showTab(i);
      invalid.reportValidity();
      invalid.focus();
      return false;
    }
  }
  return form.reportValidity();
}
function renderPrediction(result) {
  const ckd = Math.max(0, Math.min(100, Number(result.ckd_probability) * 100));
  const healthy = Math.max(0, Math.min(100, Number(result.not_ckd_probability) * 100));
  const isCkd = String(result.prediction).toLowerCase().includes('ckd') && !String(result.prediction).toLowerCase().includes('not');
  document.getElementById('predictionResult').textContent = isCkd ? 'CHRONIC KIDNEY DISEASE DETECTED' : 'NO CKD DETECTED';
  document.getElementById('classificationText').textContent = `The global aggregated model classified this assessment as ${isCkd ? 'CKD' : 'not CKD'}. Review the model probabilities and feature contributions.`;
  const classification = document.getElementById('classification');
  classification.className = `classification ${isCkd ? 'detected' : 'healthy'}`;
  document.getElementById('classificationIcon').textContent = isCkd ? '!' : '✓';
  document.getElementById('ckdProbability').textContent = `${ckd.toFixed(2)}%`;
  document.getElementById('notCkdProbability').textContent = `${healthy.toFixed(2)}%`;
  document.getElementById('ckdBar').style.width = `${ckd}%`;
  document.getElementById('notCkdBar').style.width = `${healthy}%`;
  document.getElementById('riskScore').textContent = `${ckd.toFixed(1)}%`;
  const deg = ckd * 3.6;
  document.getElementById('riskGauge').style.background = `conic-gradient(${isCkd ? '#d17b00' : '#078b69'} ${deg}deg,#dce4eb ${deg}deg)`;
  document.getElementById('riskLabel').textContent = 'Model probability (not a clinical risk grade)';
  document.getElementById('recommendationText').textContent = 'Research output only. This prediction is not a diagnosis and must not be used to make treatment decisions. Consult a qualified healthcare professional.';
}
function renderShap(items) {
  const target = document.getElementById('xaiResult');
  const empty = document.getElementById('xaiPlaceholder');
  target.replaceChildren();
  if (!Array.isArray(items) || !items.length) { empty.hidden = false; target.hidden = true; return; }
  const max = Math.max(...items.map(x => Math.abs(Number(x.shap_value))), 1e-8);
  items.forEach(item => {
    const value = Number(item.shap_value);
    const row = document.createElement('div'); row.className = 'shap-row';
    const name = document.createElement('span'); name.className = 'shap-name'; name.textContent = item.feature;
    const track = document.createElement('span'); track.className = 'shap-track';
    const fill = document.createElement('i'); fill.style.width = `${Math.max(2, Math.abs(value) / max * 100)}%`;
    fill.style.background = value >= 0 ? '#d52f3b' : '#078b69'; track.appendChild(fill);
    const amount = document.createElement('span'); amount.className = 'shap-value'; amount.textContent = value.toFixed(4);
    row.append(name, track, amount); target.appendChild(row);
  });
  empty.hidden = true; target.hidden = false;
}
form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!validateAllFields()) return;
  const button = document.getElementById('predictButton');
  button.disabled = true; button.textContent = 'Generating prediction…';
  try {
    const response = await fetch('/predict', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(collectData())});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Prediction failed.');
    renderPrediction(result); renderShap(result.shap_explanation);
    document.querySelector('.assessment-panel').scrollIntoView({behavior:'smooth',block:'start'});
  } catch (error) {
    alert(`Unable to generate prediction: ${error.message}`);
    console.error(error);
  } finally {
    button.disabled = false; button.innerHTML = '✓ &nbsp; Predict CKD Risk';
  }
});
form.addEventListener('reset', () => setTimeout(() => {
  updateConfigured(); document.getElementById('sampleCase').value = '';
  document.getElementById('predictionResult').textContent = 'Awaiting Assessment';
  document.getElementById('classificationText').textContent = 'Populate clinical markers and click Predict CKD Risk.';
  document.getElementById('classification').className = 'classification';
  document.getElementById('classificationIcon').textContent = 'i';
  ['ckdProbability','notCkdProbability','riskScore'].forEach(id => document.getElementById(id).textContent = '--');
  ['ckdBar','notCkdBar'].forEach(id => document.getElementById(id).style.width = '0');
  document.getElementById('riskGauge').style.background = 'conic-gradient(#dce4eb 0deg,#dce4eb 360deg)';
  document.getElementById('riskLabel').textContent = 'Risk Undetermined';
  document.getElementById('recommendationText').textContent = 'Run a prediction to view the model output and its SHAP feature contributions.';
  document.getElementById('xaiResult').replaceChildren(); document.getElementById('xaiResult').hidden = true;
  document.getElementById('xaiPlaceholder').hidden = false; showTab(0);
}, 0));
showTab(0); updateConfigured();
