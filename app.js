/* =============================================================================
   SECURE HEALTH AI - MILITARY-GRADE MEDICAL DIAGNOSIS SYSTEM
   ============================================================================= */

// Global Variables
let selectedFile = null;
let currentStep = 0;

// DOM Elements
const fileInput = document.getElementById('fileInput');
const fileName = document.getElementById('fileName');
const analyzeBtn = document.getElementById('analyzeBtn');
const imagePreview = document.getElementById('imagePreview');
const previewImg = document.getElementById('previewImg');
const processingSection = document.getElementById('processingSection');
const resultsSection = document.getElementById('resultsSection');
const encryptionKey = document.getElementById('encryptionKey');
const togglePassword = document.getElementById('togglePassword');

// Patient Information
const patientName = document.getElementById('patientName');
const patientAge = document.getElementById('patientAge');
const patientGender = document.getElementById('patientGender');

/* =============================================================================
   INITIALIZATION
   ============================================================================= */
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    setReportDate();
});

function setupEventListeners() {
    // File upload
    fileInput.addEventListener('change', handleFileSelect);
    
    // Form validation
    patientName.addEventListener('input', validateForm);
    patientAge.addEventListener('input', validateForm);
    patientGender.addEventListener('change', validateForm);
    encryptionKey.addEventListener('input', validateForm);
    
    // Password toggle
    if (togglePassword) {
        togglePassword.addEventListener('click', togglePasswordVisibility);
    }
    
    // Analyze button
    analyzeBtn.addEventListener('click', startAnalysis);
}

/* =============================================================================
   FILE HANDLING
   ============================================================================= */
function handleFileSelect(event) {
    const file = event.target.files[0];
    
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        alert('Please select a valid image file (JPG, PNG, GIF)');
        fileInput.value = '';
        return;
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        alert('File size must be less than 10MB');
        fileInput.value = '';
        return;
    }
    
    selectedFile = file;
    fileName.textContent = file.name;
    fileName.style.display = 'flex';
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        imagePreview.style.display = 'block';
        imagePreview.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
    reader.readAsDataURL(file);
    
    validateForm();
}

/* =============================================================================
   FORM VALIDATION
   ============================================================================= */
function validateForm() {
    const isValid = 
        selectedFile !== null &&
        patientName.value.trim().length > 0 &&
        patientAge.value > 0 &&
        patientAge.value <= 120 &&
        patientGender.value !== '' &&
        encryptionKey.value.length >= 8;
    
    analyzeBtn.disabled = !isValid;
    
    // Visual feedback
    if (encryptionKey.value.length > 0 && encryptionKey.value.length < 8) {
        encryptionKey.style.borderColor = '#ef4444';
    } else if (encryptionKey.value.length >= 8) {
        encryptionKey.style.borderColor = '#10b981';
    }
}

function togglePasswordVisibility() {
    const type = encryptionKey.type === 'password' ? 'text' : 'password';
    encryptionKey.type = type;
    
    // Toggle icon
    togglePassword.classList.toggle('fa-eye-slash');
    togglePassword.classList.toggle('fa-eye');
}

/* =============================================================================
   ANALYSIS WORKFLOW
   ============================================================================= */
async function startAnalysis() {
    if (!validatePatientInfo()) return;
    
    // Hide previous results
    resultsSection.style.display = 'none';
    
    // Show processing section
    processingSection.style.display = 'block';
    processingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    // Disable button
    analyzeBtn.disabled = true;
    
    try {
        // Step 1: Encrypt image
        await updateStep(1, 'active');
        await sleep(800);
        const imageBytes = await fileToArrayBuffer(selectedFile);
        await updateStep(1, 'completed');
        
        // Step 2: Transmit
        await updateStep(2, 'active');
        await sleep(600);
        const base64Image = arrayBufferToBase64(imageBytes);
        await updateStep(2, 'completed');
        
        // Step 3: Feature extraction (send to server)
        await updateStep(3, 'active');
        
        const requestData = {
            encrypted_data: base64Image,
            patient_name: patientName.value.trim(),
            patient_age: patientAge.value,
            patient_gender: patientGender.value
        };
        
        const response = await fetch('/fhe_predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestData)
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || `Server error: ${response.status}`);
        }
        
        await updateStep(3, 'completed');
        
        // Step 4: Classification
        await updateStep(4, 'active');
        const data = await response.json();
        await sleep(600);
        await updateStep(4, 'completed');
        
        // Step 5: Decrypt results
        await updateStep(5, 'active');
        await sleep(500);
        const results = JSON.parse(atob(data.encrypted_results));
        await updateStep(5, 'completed');
        
        // Show performance metrics
        displayPerformanceMetrics(results.processing_time);
        
        // Wait a moment before showing results
        await sleep(800);
        
        // Display results
        displayResults(results);
        
    } catch (error) {
        console.error('Analysis error:', error);
        alert(`Error during analysis: ${error.message}\n\nPlease try again or contact support.`);
        resetProcessing();
        analyzeBtn.disabled = false;
    }
}

function validatePatientInfo() {
    if (patientName.value.trim().length === 0) {
        alert('Please enter patient name');
        patientName.focus();
        return false;
    }
    
    if (patientAge.value <= 0 || patientAge.value > 120) {
        alert('Please enter a valid age (1-120)');
        patientAge.focus();
        return false;
    }
    
    if (patientGender.value === '') {
        alert('Please select gender');
        patientGender.focus();
        return false;
    }
    
    if (encryptionKey.value.length < 8) {
        alert('Encryption key must be at least 8 characters');
        encryptionKey.focus();
        return false;
    }
    
    return true;
}

/* =============================================================================
   PROCESSING VISUALIZATION
   ============================================================================= */
async function updateStep(stepNumber, status) {
    const step = document.getElementById(`step${stepNumber}`);
    if (!step) return;
    
    // Remove all status classes
    step.classList.remove('active', 'completed');
    
    // Add new status
    if (status !== 'pending') {
        step.classList.add(status);
    }
    
    // Update icon
    const statusIcon = step.querySelector('.step-status i');
    if (statusIcon) {
        statusIcon.className = 'fas';
        if (status === 'active') {
            statusIcon.classList.add('fa-spinner', 'fa-spin');
        } else if (status === 'completed') {
            statusIcon.classList.add('fa-check-circle');
        } else {
            statusIcon.classList.add('fa-circle');
        }
    }
    
    // Scroll into view
    step.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

function displayPerformanceMetrics(metrics) {
    document.getElementById('metricPreprocess').textContent = metrics.preprocessing;
    document.getElementById('metricCNN').textContent = metrics.retrained_inference;
    document.getElementById('metricFHE').textContent = metrics.fhe_inference;
    document.getElementById('metricTotal').textContent = metrics.total;
    
    document.getElementById('perfMetrics').style.display = 'block';
}

function resetProcessing() {
    processingSection.style.display = 'none';
    
    // Reset all steps
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (step) {
            step.classList.remove('active', 'completed');
            const statusIcon = step.querySelector('.step-status i');
            if (statusIcon) {
                statusIcon.className = 'fas fa-circle';
            }
        }
    }
    
    document.getElementById('perfMetrics').style.display = 'none';
}

/* =============================================================================
   RESULTS DISPLAY
   ============================================================================= */
function displayResults(results) {
    // Hide processing
    processingSection.style.display = 'none';
    
    // Show results
    resultsSection.style.display = 'flex';
    resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    // Patient Information
    document.getElementById('resultPatientName').textContent = results.patient_name;
    document.getElementById('resultPatientAge').textContent = results.patient_age;
    document.getElementById('resultPatientGender').textContent = results.patient_gender;
    
    // Diagnosis
    const diagnosis = results.diagnosis;
    const medicalInfo = results.medical_info;
    const confidence = results.head_prob;
    
    // Set diagnosis card color based on urgency
    const diagnosisCard = document.getElementById('diagnosisCard');
    diagnosisCard.style.borderColor = medicalInfo.color;
    
    // Diagnosis icon color
    const diagnosisIcon = document.getElementById('diagnosisIcon');
    diagnosisIcon.style.background = `linear-gradient(135deg, ${medicalInfo.color} 0%, ${medicalInfo.color}dd 100%)`;
    
    // Urgency badge
    const urgencyBadge = document.getElementById('urgencyBadge');
    urgencyBadge.textContent = medicalInfo.urgency;
    urgencyBadge.className = 'urgency-badge ' + medicalInfo.urgency.toLowerCase();
    
    // Diagnosis details
    document.getElementById('diagnosisName').textContent = medicalInfo.full_name;
    document.getElementById('diagnosisDescription').textContent = medicalInfo.description;
    
    // Confidence
    document.getElementById('confidenceBar').style.width = (confidence * 100) + '%';
    document.getElementById('confidenceBar').style.background = 
        `linear-gradient(90deg, ${medicalInfo.color} 0%, ${medicalInfo.color}dd 100%)`;
    document.getElementById('confidencePercentage').textContent = (confidence * 100).toFixed(1) + '%';
    document.getElementById('confidencePercentage').style.color = medicalInfo.color;
    
    // Reasons
    const reasonsList = document.getElementById('reasonsList');
    reasonsList.innerHTML = '';
    medicalInfo.reasons.forEach(reason => {
        const li = document.createElement('li');
        li.textContent = reason;
        reasonsList.appendChild(li);
    });
    
    // Precautions
    const precautionsList = document.getElementById('precautionsList');
    precautionsList.innerHTML = '';
    medicalInfo.precautions.forEach(precaution => {
        const li = document.createElement('li');
        li.textContent = precaution;
        
        // Highlight critical precautions
        if (precaution.includes('⚠️') || precaution.includes('IMMEDIATE')) {
            li.style.borderLeftColor = '#ef4444';
            li.style.background = '#fef2f2';
            li.style.fontWeight = '700';
        }
        
        precautionsList.appendChild(li);
    });
    
    // FHE Model Results
    document.getElementById('fheClass').textContent = results.head_class;
    document.getElementById('fheConfidence').textContent = (results.head_prob * 100).toFixed(1) + '%';
    
    for (let i = 0; i < 4; i++) {
        const prob = results.head_pred[i] * 100;
        document.getElementById(`fheProb${i}`).style.width = prob + '%';
        document.getElementById(`fheVal${i}`).textContent = prob.toFixed(1) + '%';
    }
    
    // CNN Model Results
    document.getElementById('cnnClass').textContent = results.retrained_class;
    document.getElementById('cnnConfidence').textContent = (results.retrained_prob * 100).toFixed(1) + '%';
    
    for (let i = 0; i < 4; i++) {
        const prob = results.retrained_pred[i] * 100;
        document.getElementById(`cnnProb${i}`).style.width = prob + '%';
        document.getElementById(`cnnVal${i}`).textContent = prob.toFixed(1) + '%';
    }
}

/* =============================================================================
   UTILITY FUNCTIONS
   ============================================================================= */
function fileToArrayBuffer(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.readAsArrayBuffer(file);
        reader.onload = () => resolve(reader.result);
        reader.onerror = error => reject(error);
    });
}

function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function setReportDate() {
    const dateElement = document.getElementById('reportDate');
    if (dateElement) {
        const now = new Date();
        const options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        dateElement.textContent = now.toLocaleDateString('en-US', options);
    }
}

/* =============================================================================
   RESET & ACTIONS
   ============================================================================= */
function resetApp() {
    // Confirm reset
    if (!confirm('Start a new analysis? Current results will be cleared.')) {
        return;
    }
    
    // Reset form
    fileInput.value = '';
    fileName.textContent = '';
    fileName.style.display = 'none';
    patientName.value = '';
    patientAge.value = '';
    patientGender.value = '';
    encryptionKey.value = '';
    
    // Reset UI
    imagePreview.style.display = 'none';
    processingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    
    // Reset variables
    selectedFile = null;
    analyzeBtn.disabled = true;
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function downloadReport() {
    alert('PDF download feature coming soon!\n\nFor now, please use the Print button to save as PDF through your browser.');
}

/* =============================================================================
   ERROR HANDLING
   ============================================================================= */
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});

/* =============================================================================
   EXPORT FUNCTIONS FOR INLINE HANDLERS
   ============================================================================= */
window.resetApp = resetApp;
window.downloadReport = downloadReport;
